"""Main crawl orchestrator: coordinates URL discovery, extraction, and output.

The ``Crawler`` class is the top-level entry point for crawling operations.
It loads per-site configurations from sources.yaml, runs the URL discovery
pipeline, fetches and extracts article content, and writes JSONL output
files to ``data/raw/YYYY-MM-DD/{source_id}.jsonl``.

Integration points:
    - NetworkGuard: resilient HTTP client (``src/crawling/network_guard.py``)
    - URLDiscovery: 3-tier URL discovery (``src/crawling/url_discovery.py``)
    - ArticleExtractor: multi-library extraction (``src/crawling/article_extractor.py``)
    - DedupEngine: URL + content dedup (``src/crawling/dedup.py``) -- integration point
    - UAManager: user-agent rotation (``src/crawling/ua_manager.py``) -- integration point

Reference:
    Step 5 Architecture Blueprint, Section 6 (Conductor Pattern).
    Step 6 Crawling Strategies (Per-Site configurations).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.constants import (
    DATA_RAW_DIR,
    MAX_ARTICLES_PER_SITE_PER_DAY,
    DEFAULT_RATE_LIMIT_SECONDS,
    ENABLED_DEFAULT,
)
from src.crawling.contracts import RawArticle, CrawlResult, DiscoveredURL
from src.crawling.network_guard import NetworkGuard
from src.crawling.url_discovery import URLDiscovery
from src.crawling.article_extractor import ArticleExtractor
from src.utils.error_handler import (
    CrawlError,
    NetworkError,
    ParseError,
    BlockDetectedError,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSONL output writer
# ---------------------------------------------------------------------------

class JSONLWriter:
    """Atomic JSONL file writer for crawl output.

    Writes articles to a temporary file first, then atomically moves
    to the final path on close. This prevents partial writes from
    corrupting output files.

    Args:
        output_path: Final path for the JSONL file.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._temp_path: Path | None = None
        self._file = None
        self._count = 0

    def open(self) -> None:
        """Open the temporary file for writing.

        Creates parent directories if they do not exist.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_str = tempfile.mkstemp(
            suffix=".jsonl.tmp",
            dir=str(self._output_path.parent),
        )
        self._temp_path = Path(temp_str)
        self._file = os.fdopen(fd, "w", encoding="utf-8")
        self._count = 0

    def write_article(self, article: RawArticle) -> None:
        """Write a single article as one JSONL line.

        Args:
            article: RawArticle to serialize and write.

        Raises:
            RuntimeError: If the writer has not been opened.
        """
        if self._file is None:
            raise RuntimeError("JSONLWriter not opened. Call open() first.")
        line = article.to_jsonl_line()
        self._file.write(line + "\n")
        self._count += 1

    def close(self) -> int:
        """Close the file and atomically move to the final path.

        Returns:
            Number of articles written.
        """
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None

        if self._temp_path is not None and self._temp_path.exists():
            if self._count > 0:
                if self._output_path.exists():
                    # Append new articles to existing JSONL (multi-run support)
                    with open(self._output_path, "a", encoding="utf-8") as dest:
                        with open(self._temp_path, "r", encoding="utf-8") as src:
                            dest.write(src.read())
                        dest.flush()
                        os.fsync(dest.fileno())
                    self._temp_path.unlink(missing_ok=True)
                else:
                    # First run — atomic rename
                    os.replace(str(self._temp_path), str(self._output_path))
                logger.info(
                    "jsonl_written",
                    path=str(self._output_path),
                    articles=self._count,
                )
            else:
                # No articles -- remove temp file
                self._temp_path.unlink(missing_ok=True)

        return self._count

    def __enter__(self) -> JSONLWriter:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def count(self) -> int:
        """Number of articles written so far."""
        return self._count


# ---------------------------------------------------------------------------
# Crawl state persistence
# ---------------------------------------------------------------------------

class CrawlState:
    """Persists crawl progress for resume-on-restart.

    Tracks which URLs have been processed for each site, allowing the
    crawler to resume from the last processed URL after a restart.

    State is stored as a JSON file at:
    ``data/raw/YYYY-MM-DD/.crawl_state.json``

    Args:
        state_dir: Directory to store the state file.
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_path = state_dir / ".crawl_state.json"
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load state from disk if it exists."""
        if self._state_path.exists():
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
                logger.info("crawl_state_loaded path=%s", self._state_path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("crawl_state_load_error: %s", e)
                self._state = {}

    def save(self) -> None:
        """Save state to disk atomically."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(
            suffix=".json.tmp",
            dir=str(self._state_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
            os.replace(temp, str(self._state_path))
        except OSError:
            if os.path.exists(temp):
                os.unlink(temp)
            raise

    def is_url_processed(self, source_id: str, url: str) -> bool:
        """Check if a URL has already been processed.

        Args:
            source_id: Site identifier.
            url: Article URL.

        Returns:
            True if the URL was already processed in this crawl run.
        """
        processed = self._state.get(source_id, {}).get("processed_urls", [])
        return url in processed

    def mark_url_processed(self, source_id: str, url: str) -> None:
        """Mark a URL as processed.

        Args:
            source_id: Site identifier.
            url: Article URL.
        """
        if source_id not in self._state:
            self._state[source_id] = {"processed_urls": [], "last_updated": ""}
        self._state[source_id]["processed_urls"].append(url)
        self._state[source_id]["last_updated"] = datetime.now(timezone.utc).isoformat()

    def get_processed_count(self, source_id: str) -> int:
        """Get the number of URLs processed for a site.

        Args:
            source_id: Site identifier.

        Returns:
            Number of processed URLs.
        """
        return len(self._state.get(source_id, {}).get("processed_urls", []))

    def mark_site_complete(self, source_id: str) -> None:
        """Mark a site as fully crawled.

        Args:
            source_id: Site identifier.
        """
        if source_id not in self._state:
            self._state[source_id] = {"processed_urls": []}
        self._state[source_id]["complete"] = True
        self._state[source_id]["completed_at"] = datetime.now(timezone.utc).isoformat()

    def is_site_complete(self, source_id: str) -> bool:
        """Check if a site has already been fully crawled.

        Args:
            source_id: Site identifier.

        Returns:
            True if the site is marked as complete.
        """
        return self._state.get(source_id, {}).get("complete", False)


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class Crawler:
    """Main crawl orchestrator.

    Coordinates URL discovery, article extraction, deduplication, and
    JSONL output for all configured sites.

    Pipeline per site:
        1. Load site config from sources.yaml
        2. Configure NetworkGuard (rate limit, circuit breaker)
        3. Run URL discovery (RSS -> Sitemap -> DOM -> Playwright)
        4. Dedup check against crawl state
        5. Fetch and extract each article
        6. Write to JSONL output
        7. Update crawl state

    Args:
        network_guard: Pre-configured NetworkGuard instance.
        crawl_date: Date for this crawl run (default: today).
        output_dir: Base output directory (default: data/raw/).
        max_articles_per_site: Maximum articles to extract per site.
        browser_renderer: Optional BrowserRenderer for paywall/JS sites.
        adaptive_extractor: Optional AdaptiveExtractor for CSS fallback.
    """

    def __init__(
        self,
        network_guard: NetworkGuard | None = None,
        crawl_date: str | None = None,
        output_dir: Path | None = None,
        max_articles_per_site: int = MAX_ARTICLES_PER_SITE_PER_DAY,
        browser_renderer: Any | None = None,
        adaptive_extractor: Any | None = None,
    ) -> None:
        self._guard = network_guard or NetworkGuard()
        self._date = crawl_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._output_base = output_dir or DATA_RAW_DIR
        self._output_dir = self._output_base / self._date
        self._max_articles = max_articles_per_site

        self._url_discovery = URLDiscovery(self._guard)
        self._extractor = ArticleExtractor(
            self._guard,
            browser_renderer=browser_renderer,
            adaptive_extractor=adaptive_extractor,
        )
        self._crawl_state = CrawlState(self._output_dir)

        # Integration points for external modules (set by caller)
        self._dedup_checker: Any = None  # DedupEngine integration point
        self._ua_manager: Any = None  # UAManager integration point

    def set_dedup_engine(self, dedup_engine: Any) -> None:
        """Set the deduplication engine (integration point for @dedup-dev).

        Args:
            dedup_engine: Object with ``is_duplicate(url, content_hash) -> bool`` method.
        """
        self._dedup_checker = dedup_engine

    def set_ua_manager(self, ua_manager: Any) -> None:
        """Set the user-agent manager (integration point for @ua-rotation-dev).

        Args:
            ua_manager: Object with ``get_headers(source_id) -> dict`` method.
        """
        self._ua_manager = ua_manager

    def crawl_site(
        self,
        source_id: str,
        site_config: dict[str, Any],
    ) -> CrawlResult:
        """Crawl a single site: discover URLs, extract articles, write output.

        This is the core per-site crawl pipeline. It never raises exceptions
        to the caller -- all errors are caught, logged, and reported in
        the CrawlResult.

        Args:
            source_id: Site identifier (key in sources.yaml).
            site_config: Site configuration dictionary.

        Returns:
            CrawlResult with crawl statistics and extracted articles.
        """
        start_time = time.monotonic()
        result = CrawlResult(source_id=source_id)

        # Check if site was already completed in a previous run
        if self._crawl_state.is_site_complete(source_id):
            logger.info("site_already_complete source_id=%s", source_id)
            return result

        # Check if site is enabled — D-7 (13): ENABLED_DEFAULT from constants.py (SOT)
        if not site_config.get("meta", {}).get("enabled", ENABLED_DEFAULT):
            logger.info("site_disabled source_id=%s", source_id)
            return result

        logger.info("crawl_site_start source_id=%s date=%s", source_id, self._date)

        # Configure NetworkGuard for this site
        crawl_config = site_config.get("crawl", {})
        self._guard.configure_site(
            source_id,
            rate_limit_seconds=crawl_config.get("rate_limit_seconds", DEFAULT_RATE_LIMIT_SECONDS),
            jitter_seconds=crawl_config.get("jitter_seconds", 0),
        )

        # Phase 1: URL Discovery
        try:
            discovered = self._url_discovery.discover(site_config, source_id)
            result.discovered_urls = len(discovered)
        except Exception as e:
            logger.error("discovery_failed source_id=%s: %s", source_id, e)
            result.errors.append(f"Discovery failed: {e}")
            result.elapsed_seconds = time.monotonic() - start_time
            return result

        if not discovered:
            logger.warning("no_urls_discovered source_id=%s", source_id)
            result.elapsed_seconds = time.monotonic() - start_time
            return result

        # Phase 2: Dedup + Extraction
        output_path = self._output_dir / f"{source_id}.jsonl"

        try:
            with JSONLWriter(output_path) as writer:
                for url_obj in discovered:
                    if writer.count >= self._max_articles:
                        logger.info(
                            "max_articles_reached source_id=%s limit=%d",
                            source_id, self._max_articles,
                        )
                        break

                    # Skip already-processed URLs (resume support)
                    if self._crawl_state.is_url_processed(source_id, url_obj.url):
                        result.skipped_dedup_count += 1
                        continue

                    # External dedup check (if available)
                    if self._dedup_checker is not None:
                        try:
                            if self._dedup_checker.is_duplicate(url_obj.url, ""):
                                result.skipped_dedup_count += 1
                                continue
                        except Exception as e:
                            logger.debug("dedup_check_error url=%s: %s", url_obj.url, e)

                    # Get UA headers (if available)
                    extra_headers: dict[str, str] | None = None
                    if self._ua_manager is not None:
                        try:
                            extra_headers = self._ua_manager.get_headers(source_id)
                        except Exception:
                            pass

                    # Extract article
                    try:
                        article = self._extractor.extract(
                            url=url_obj.url,
                            source_id=source_id,
                            site_config=site_config,
                            title_hint=url_obj.title_hint,
                        )
                        writer.write_article(article)
                        result.articles.append(article)
                        result.extracted_count += 1

                        # Content-based dedup (if available)
                        if self._dedup_checker is not None and article.content_hash:
                            try:
                                self._dedup_checker.register(
                                    url_obj.url, article.content_hash
                                )
                            except Exception:
                                pass

                    except ParseError as e:
                        result.failed_count += 1
                        result.errors.append(f"Parse error {url_obj.url}: {e}")
                        logger.warning(
                            "article_extraction_failed",
                            url=url_obj.url,
                            source_id=source_id,
                            error=str(e),
                        )
                    except NetworkError as e:
                        result.failed_count += 1
                        result.errors.append(f"Network error {url_obj.url}: {e}")
                        logger.warning(
                            "article_fetch_failed",
                            url=url_obj.url,
                            source_id=source_id,
                            error=str(e),
                        )
                    except BlockDetectedError as e:
                        result.failed_count += 1
                        result.errors.append(f"Blocked at {url_obj.url}: {e}")
                        logger.warning(
                            "article_blocked",
                            url=url_obj.url,
                            source_id=source_id,
                            block_type=e.block_type,
                        )
                        # If blocked, escalate tier
                        result.tier_used = max(result.tier_used, 2)
                    except Exception as e:
                        result.failed_count += 1
                        result.errors.append(f"Unexpected error {url_obj.url}: {e}")
                        logger.error(
                            "article_extraction_unexpected",
                            url=url_obj.url,
                            source_id=source_id,
                            error=str(e),
                            error_type=type(e).__name__,
                        )

                    # Mark URL as processed (for resume support)
                    self._crawl_state.mark_url_processed(source_id, url_obj.url)

        except Exception as e:
            logger.error(
                "jsonl_writer_error",
                source_id=source_id,
                error=str(e),
            )
            result.errors.append(f"JSONL writer error: {e}")

        # Phase 3: Finalize
        self._crawl_state.mark_site_complete(source_id)
        self._crawl_state.save()

        result.elapsed_seconds = time.monotonic() - start_time
        logger.info(
            "crawl_site_complete",
            source_id=source_id,
            discovered=result.discovered_urls,
            extracted=result.extracted_count,
            failed=result.failed_count,
            skipped=result.skipped_dedup_count,
            elapsed=round(result.elapsed_seconds, 1),
            tier=result.tier_used,
        )

        return result

    def crawl_sites(
        self,
        sites: dict[str, dict[str, Any]],
    ) -> list[CrawlResult]:
        """Crawl multiple sites sequentially.

        Never raises on individual site failures -- each site is independent.

        Args:
            sites: Dictionary mapping source_id -> site_config.

        Returns:
            List of CrawlResult objects, one per site.
        """
        results: list[CrawlResult] = []
        total = len(sites)

        for idx, (source_id, site_config) in enumerate(sites.items(), 1):
            logger.info(
                "crawl_progress",
                current=idx,
                total=total,
                source_id=source_id,
            )
            try:
                result = self.crawl_site(source_id, site_config)
            except Exception as e:
                # Should never happen (crawl_site catches all), but safety net
                logger.error(
                    "crawl_site_fatal",
                    source_id=source_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                result = CrawlResult(
                    source_id=source_id,
                    errors=[f"Fatal: {e}"],
                    elapsed_seconds=0.0,
                )
            results.append(result)

        # Log summary
        total_discovered = sum(r.discovered_urls for r in results)
        total_extracted = sum(r.extracted_count for r in results)
        total_failed = sum(r.failed_count for r in results)
        total_elapsed = sum(r.elapsed_seconds for r in results)

        logger.info(
            "crawl_complete",
            sites=total,
            total_discovered=total_discovered,
            total_extracted=total_extracted,
            total_failed=total_failed,
            total_elapsed=round(total_elapsed, 1),
            date=self._date,
        )

        return results

    def close(self) -> None:
        """Release resources held by the crawler."""
        self._guard.close()

    def __enter__(self) -> Crawler:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
