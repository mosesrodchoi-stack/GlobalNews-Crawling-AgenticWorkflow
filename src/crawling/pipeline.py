"""Crawling Pipeline Orchestrator: end-to-end crawl coordination with 4-level retry.

Integrates all crawling subsystems into a unified pipeline:
    - sources.yaml loading and site iteration
    - Per-site adapter selection via ``get_adapter(site_id)``
    - Circuit breaker checks before site processing
    - 3-Tier URL Discovery (RSS -> Sitemap -> DOM)
    - Article extraction via ArticleExtractor + site adapters
    - 3-level deduplication (URL + Title + SimHash)
    - JSONL output to ``data/raw/YYYY-MM-DD/all_articles.jsonl``
    - 4-level retry system (NetworkGuard x Strategy x Round x Restart)
    - Structured crawl report generation

Pipeline stages (per site):
    1. Load     -- Read site adapter config, init NetworkGuard per site
    2. Iterate  -- Loop through target sites
    3. Select   -- Choose adapter via get_adapter(site_id)
    4. Discover -- Run URL discovery (RSS -> Sitemap -> DOM fallback)
    5. Extract  -- Fetch and extract article content
    6. Dedup    -- Apply URL + content-hash dedup
    7. JSONL    -- Write validated articles to consolidated output

CLI: ``python3 main.py --mode crawl --date 2026-02-25 [--sites chosun,donga] [--groups A,B]``

Reference:
    Step 5 Architecture Blueprint, Section 6 (Pipeline Orchestration).
    Step 3 Crawling Feasibility (4-Level Retry Architecture).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config.constants import (
    CRAWL_LOOKBACK_HOURS,
    CRAWL_NEVER_ABANDON,
    DATA_RAW_DIR,
    MAX_ARTICLES_PER_SITE_PER_DAY,
    DEFAULT_RATE_LIMIT_SECONDS,
    ENABLED_DEFAULT,
)
from src.crawling.contracts import RawArticle, CrawlResult, DiscoveredURL
from src.crawling.network_guard import NetworkGuard
from src.crawling.url_discovery import URLDiscovery
from src.crawling.article_extractor import ArticleExtractor
from src.crawling.browser_renderer import BrowserRenderer
from src.crawling.adaptive_extractor import AdaptiveExtractor
from src.crawling.crawler import JSONLWriter, CrawlState
from src.crawling.dedup import DedupEngine
from src.crawling.ua_manager import UAManager
from src.crawling.circuit_breaker import CircuitBreakerCoordinator
from src.crawling.anti_block import AntiBlockEngine
from src.crawling.dynamic_bypass import DynamicBypassEngine
from src.crawling.retry_manager import (
    RetryManager,
    StrategyMode,
    L3_MAX_ROUNDS,
    L4_MAX_RESTARTS,
)
from src.crawling.crawl_report import (
    generate_crawl_report,
    print_crawl_summary,
)
from src.utils.config_loader import (
    load_sources_config,
    get_enabled_sites,
    get_site_config,
    get_sites_by_group,
)
from src.utils.error_handler import (
    CrawlError,
    NetworkError,
    ParseError,
    BlockDetectedError,
    RateLimitError,
    CircuitState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONCURRENCY = 5  # Not used yet (sequential), placeholder for async
TOTALWAR_DELAY_MULTIPLIER = 2.0  # Multiply rate limit in TotalWar mode


# ---------------------------------------------------------------------------
# Crawling Pipeline
# ---------------------------------------------------------------------------

class CrawlingPipeline:
    """End-to-end crawling pipeline orchestrator with 4-level retry.

    Coordinates all crawling subsystems to crawl configured news sites,
    extract articles, deduplicate, and produce consolidated JSONL output.

    The pipeline supports:
    - Selective crawling by site IDs or group letters
    - Resume-on-restart via CrawlState persistence
    - 4-level retry with Standard/TotalWar strategy escalation
    - Per-site circuit breakers and anti-block escalation
    - Consolidated output: all articles in one JSONL file per day
    - Structured crawl report generation

    Args:
        crawl_date: Target date string (YYYY-MM-DD).
        output_dir: Base output directory (default: data/raw/).
        sites_filter: Optional list of specific site IDs to crawl.
        groups_filter: Optional list of group letters to crawl.
        max_articles_per_site: Cap on articles per site.
        dry_run: If True, validate config without making requests.
    """

    def __init__(
        self,
        crawl_date: str | None = None,
        output_dir: Path | None = None,
        sites_filter: list[str] | None = None,
        groups_filter: list[str] | None = None,
        max_articles_per_site: int = MAX_ARTICLES_PER_SITE_PER_DAY,
        dry_run: bool = False,
    ) -> None:
        self._date = crawl_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._output_base = output_dir or DATA_RAW_DIR
        self._output_dir = self._output_base / self._date
        self._sites_filter = sites_filter
        self._groups_filter = groups_filter
        self._max_articles = max_articles_per_site
        self._dry_run = dry_run

        # Core subsystems -- initialized lazily in run()
        self._guard: NetworkGuard | None = None
        self._url_discovery: URLDiscovery | None = None
        self._extractor: ArticleExtractor | None = None
        self._dedup: DedupEngine | None = None
        self._ua_manager: UAManager | None = None
        self._circuit_breakers: CircuitBreakerCoordinator | None = None
        self._anti_block: AntiBlockEngine | None = None
        self._retry_manager: RetryManager | None = None
        self._crawl_state: CrawlState | None = None

        # Results tracking
        self._results: list[CrawlResult] = []
        self._pipeline_start_time: float = 0.0

        # 24-hour lookback cutoff: articles with published_at before this
        # timestamp are dropped after extraction (absolute rule for daily runs).
        self._crawl_start_utc: datetime = datetime.now(timezone.utc)
        self._lookback_cutoff: datetime = (
            self._crawl_start_utc - timedelta(hours=CRAWL_LOOKBACK_HOURS)
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full crawling pipeline.

        This is the main entry point. It:
        1. Loads sources.yaml and determines target sites
        2. Initializes all subsystems
        3. Iterates through sites with 4-level retry
        4. Writes consolidated JSONL output
        5. Generates and returns the crawl report

        Returns:
            Crawl report dictionary (also written to disk).
        """
        self._pipeline_start_time = time.monotonic()
        # Refresh the 24h lookback cutoff at actual run time
        self._crawl_start_utc = datetime.now(timezone.utc)
        self._lookback_cutoff = (
            self._crawl_start_utc - timedelta(hours=CRAWL_LOOKBACK_HOURS)
        )

        logger.info(
            "pipeline_start date=%s sites_filter=%s groups_filter=%s dry_run=%s "
            "lookback_cutoff=%s",
            self._date,
            self._sites_filter or "all",
            self._groups_filter or "all",
            self._dry_run,
            self._lookback_cutoff.isoformat(),
        )

        # Step 1: Load configuration and determine target sites
        sources_config = load_sources_config(validate=False)
        target_sites = self._resolve_target_sites(sources_config)

        if not target_sites:
            logger.warning("no_target_sites No sites matched the filter criteria.")
            return generate_crawl_report(
                results=[],
                crawl_date=self._date,
                elapsed_seconds=0.0,
                output_dir=self._output_dir,
            )

        logger.info("target_sites count=%s sites=%s", len(target_sites), list(target_sites.keys()))

        if self._dry_run:
            return self._run_dry(target_sites)

        # Step 2: Initialize subsystems
        self._init_subsystems(sources_config)

        # Step 3: Run the crawl pipeline with Level 4 restart support
        self._results = self._run_with_restarts(target_sites)

        # Step 4: Generate report
        elapsed = time.monotonic() - self._pipeline_start_time
        retry_stats = (
            self._retry_manager.get_retry_stats().get("total_retry_counts", {})
            if self._retry_manager
            else {}
        )

        report = generate_crawl_report(
            results=self._results,
            crawl_date=self._date,
            elapsed_seconds=elapsed,
            retry_stats=retry_stats,
            output_dir=self._output_dir,
        )

        # Print summary to console
        print_crawl_summary(report)

        logger.info(
            "pipeline_complete date=%s articles=%s elapsed=%s",
            self._date, report["total_articles"], round(elapsed, 1),
        )

        return report

    def close(self) -> None:
        """Release all resources held by the pipeline."""
        if self._guard is not None:
            self._guard.close()
        if self._dedup is not None:
            self._dedup.close()

    def __enter__(self) -> CrawlingPipeline:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------

    def _init_subsystems(self, sources_config: dict[str, Any]) -> None:
        """Initialize all crawling subsystems.

        Args:
            sources_config: Loaded sources.yaml configuration.
        """
        # NetworkGuard -- shared HTTP client
        self._guard = NetworkGuard()

        # URL Discovery
        self._url_discovery = URLDiscovery(self._guard)

        # Browser Renderer (optional — for paywall/JS sites)
        self._browser_renderer: BrowserRenderer | None = None
        try:
            renderer = BrowserRenderer()
            if renderer.is_available():
                self._browser_renderer = renderer
                logger.info("browser_renderer_available engine=patchright/playwright")
            else:
                logger.info("browser_renderer_unavailable — paywall sites will use title-only")
        except Exception as e:
            logger.debug("browser_renderer_init_error error=%s", str(e))

        # Adaptive Extractor (CSS selector fallback for rendered HTML)
        self._adaptive_extractor = AdaptiveExtractor()

        # Article Extractor
        self._extractor = ArticleExtractor(
            self._guard,
            browser_renderer=self._browser_renderer,
            adaptive_extractor=self._adaptive_extractor,
        )

        # Deduplication engine (SQLite-backed)
        self._dedup = DedupEngine()

        # User-Agent manager
        self._ua_manager = UAManager(sources_config=sources_config)

        # Circuit breaker coordinator
        self._circuit_breakers = CircuitBreakerCoordinator()

        # Anti-block engine
        self._anti_block = AntiBlockEngine(auto_load=True)

        # 4-Level retry manager
        self._retry_manager = RetryManager(crawl_date=self._date)

        # Dynamic Bypass Engine (block-type-aware strategy dispatch)
        # Used by the Never-Abandon loop for targeted bypass selection
        self._bypass_engine = DynamicBypassEngine(
            proxy_pool=[],  # Populated from sources.yaml proxy config if available
            enable_browser=self._browser_renderer is not None,
        )

        # Crawl state (resume support)
        self._crawl_state = CrawlState(self._output_dir)

        logger.info("subsystems_initialized")

    def _resolve_target_sites(
        self,
        sources_config: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Determine which sites to crawl based on filters.

        Opt-out pattern: sites are included by default unless explicitly
        disabled with ``meta.enabled: false``.  This matches the project
        directive that full-site crawling is the default mode.

        Priority:
        1. --sites filter: explicit site IDs (no enabled check)
        2. --groups filter: non-disabled sites in specified groups
        3. Default: all non-disabled sites

        D-7: ``meta.enabled`` default mirrors ``_SOURCE_DEFAULTS`` in
        ``config_loader.py`` (both default to ``True``).

        Args:
            sources_config: Loaded sources.yaml configuration.

        Returns:
            Dictionary mapping site_id -> site_config for target sites.
        """
        all_sources = sources_config.get("sources", {})
        target: dict[str, dict[str, Any]] = {}

        if self._sites_filter:
            # Explicit site IDs
            for site_id in self._sites_filter:
                if site_id in all_sources:
                    target[site_id] = all_sources[site_id]
                else:
                    logger.warning(
                        "site_not_found site_id=%s available=%s",
                        site_id, sorted(all_sources.keys())[:10],
                    )
        elif self._groups_filter:
            # Sites in specified groups — D-7 (13): ENABLED_DEFAULT from constants.py (SOT)
            for site_id, site_cfg in all_sources.items():
                if site_cfg.get("group") in self._groups_filter:
                    if site_cfg.get("meta", {}).get("enabled", ENABLED_DEFAULT):
                        target[site_id] = site_cfg
        else:
            # All non-disabled sites — D-7 (13): ENABLED_DEFAULT from constants.py (SOT)
            for site_id, site_cfg in all_sources.items():
                if site_cfg.get("meta", {}).get("enabled", ENABLED_DEFAULT):
                    target[site_id] = site_cfg

        return target

    # -----------------------------------------------------------------------
    # Dry Run
    # -----------------------------------------------------------------------

    def _run_dry(self, target_sites: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Execute a dry run -- validate config and show plan.

        Args:
            target_sites: Sites that would be crawled.

        Returns:
            Dry-run report.
        """
        logger.info("DRY RUN: Validating configuration for %s sites", len(target_sites))

        for site_id, site_cfg in sorted(target_sites.items()):
            method = site_cfg.get("crawl", {}).get("primary_method", "?")
            group = site_cfg.get("group", "?")
            estimate = site_cfg.get("meta", {}).get("daily_article_estimate", 0)
            difficulty = site_cfg.get("meta", {}).get("difficulty_tier", "?")
            logger.info(
                "  %s: %s (Group %s, %s, ~%s articles/day)",
                site_id, method, group, difficulty, estimate,
            )

        total_estimate = sum(
            s.get("meta", {}).get("daily_article_estimate", 0)
            for s in target_sites.values()
        )
        logger.info("DRY RUN complete. Would crawl ~%s articles from %s sites.",
                     total_estimate, len(target_sites))

        return {
            "date": self._date,
            "dry_run": True,
            "total_sites": len(target_sites),
            "estimated_articles": total_estimate,
            "sites": list(target_sites.keys()),
        }

    # -----------------------------------------------------------------------
    # Pipeline Execution (with Level 4 restarts)
    # -----------------------------------------------------------------------

    def _run_with_restarts(
        self,
        target_sites: dict[str, dict[str, Any]],
    ) -> list[CrawlResult]:
        """Run the pipeline with Level 4 restart support.

        On catastrophic failure (network outage, disk full), restarts the
        entire pipeline. Already-collected articles are preserved via
        CrawlState deduplication.

        Args:
            target_sites: Sites to crawl.

        Returns:
            Aggregated CrawlResult list from all restarts.
        """
        all_results: dict[str, CrawlResult] = {}

        for restart in range(1, L4_MAX_RESTARTS + 1):
            logger.info(
                "pipeline_run restart=%s/%s sites=%s",
                restart, L4_MAX_RESTARTS, len(target_sites),
            )

            try:
                results = self._run_single_pass(target_sites, restart)

                # Merge results (latest overwrites previous if re-crawled)
                for result in results:
                    existing = all_results.get(result.source_id)
                    if existing is None or result.extracted_count > existing.extracted_count:
                        all_results[result.source_id] = result

                # Check if any sites need re-processing
                sites_needing_restart = self._check_restart_needed(target_sites, all_results)
                if not sites_needing_restart:
                    logger.info("pipeline_all_sites_complete restart=%s", restart)
                    break

                if restart < L4_MAX_RESTARTS:
                    # Apply restart delay
                    assert self._retry_manager is not None
                    for site_id in sites_needing_restart:
                        if self._retry_manager.should_restart_pipeline(site_id):
                            delay = self._retry_manager.restart_pipeline(site_id)
                            logger.info(
                                "pipeline_restart_delay site_id=%s delay=%ss",
                                site_id, delay,
                            )
                            time.sleep(min(delay, 60.0))
                            break  # One delay per restart cycle
                else:
                    # Final restart exhausted -- escalate remaining failures
                    self._escalate_remaining_failures(sites_needing_restart)

            except KeyboardInterrupt:
                logger.warning("pipeline_interrupted restart=%s", restart)
                raise
            except Exception as e:
                logger.error(
                    "pipeline_restart_error restart=%s error=%s error_type=%s",
                    restart, str(e), type(e).__name__,
                )
                if restart >= L4_MAX_RESTARTS:
                    raise

        return list(all_results.values())

    def _check_restart_needed(
        self,
        target_sites: dict[str, dict[str, Any]],
        results: dict[str, CrawlResult],
    ) -> list[str]:
        """Check which sites need re-processing.

        A site needs restart if:
        - It was targeted but has no result
        - It has 0 articles AND errors
        - Its retry manager says restart is needed

        Args:
            target_sites: All target sites.
            results: Current results per site.

        Returns:
            List of site IDs that need restart.
        """
        needs_restart: list[str] = []

        for site_id in target_sites:
            result = results.get(site_id)
            if result is None:
                needs_restart.append(site_id)
                continue

            if result.extracted_count == 0 and result.errors:
                assert self._retry_manager is not None
                retry_state = self._retry_manager.get_state(site_id)
                if not retry_state.exhausted:
                    needs_restart.append(site_id)

        return needs_restart

    def _escalate_remaining_failures(self, site_ids: list[str]) -> None:
        """Escalate remaining failed sites to Never-Abandon persistence loop.

        크롤링 절대 원칙: NEVER abandon. After standard 90 attempts exhaust,
        activate the Never-Abandon loop — cycle through alternative source
        strategies (RSS, cache, AMP, etc.) with exponential backoff.

        Args:
            site_ids: Sites that failed all retry levels.
        """
        assert self._retry_manager is not None
        for site_id in site_ids:
            if self._retry_manager.is_exhausted(site_id):
                report_path = self._retry_manager.escalate_tier6(site_id)
                logger.warning(
                    "never_abandon_loop_start site_id=%s report=%s",
                    site_id, str(report_path),
                )

        # Never-Abandon persistence loop for ALL escalated sites
        if not CRAWL_NEVER_ABANDON:
            return

        never_abandon_sites = [
            sid for sid in site_ids
            if self._retry_manager.get_state(sid).never_abandon_active
        ]

        if not never_abandon_sites:
            return

        logger.warning(
            "never_abandon_loop_entering sites=%s",
            [s for s in never_abandon_sites],
        )

        for site_id in never_abandon_sites:
            self._run_never_abandon_loop(site_id)

    def _run_never_abandon_loop(self, site_id: str) -> None:
        """Run the Never-Abandon persistence loop for a single site.

        Uses the DynamicBypassEngine to select targeted bypass strategies
        based on the detected block type. Each cycle tries a different
        strategy from the engine's recommendation, with exponential
        backoff between cycles.

        Strategy selection flow:
        1. Get the last known block type for this site (from BlockDetector)
        2. Ask DynamicBypassEngine for strategies effective against that block
        3. Try strategies in order (cheapest first, learned success rate)
        4. On success → done. On failure → advance cycle, try next strategy.

        Args:
            site_id: Site identifier.
        """
        assert self._retry_manager is not None
        assert self._circuit_breakers is not None

        state = self._retry_manager.get_state(site_id)

        # Get last known block type from anti-block engine's profile
        last_block_type = None
        if self._anti_block is not None:
            profile = self._anti_block.get_profile(site_id)
            if profile.last_block_type:
                from src.crawling.block_detector import BlockType
                try:
                    last_block_type = BlockType(profile.last_block_type)
                except ValueError:
                    pass

        while self._retry_manager.advance_never_abandon_cycle(site_id):
            # strategy_name is for logging only — actual strategy selection
            # is handled by DynamicBypassEngine.try_strategies() in Phase A.
            strategy_name, delay = self._retry_manager.get_never_abandon_strategy(site_id)

            logger.warning(
                "never_abandon_cycle site_id=%s cycle=%s base_strategy=%s "
                "block_type=%s delay=%.1fs",
                site_id, state.never_abandon_cycle, strategy_name,
                last_block_type.value if last_block_type else "unknown", delay,
            )

            # Wait with exponential backoff
            if delay > 0:
                time.sleep(min(delay, 120.0))  # Cap actual sleep for usability

            # Reset circuit breaker for fresh attempt
            self._circuit_breakers.reset(site_id)

            # Phase A: Try DynamicBypassEngine strategies on failed URLs
            if state.failed_urls and self._bypass_engine is not None:
                bypass_success = False
                recovered_urls: list[str] = []
                output_path = self._output_dir / "all_articles.jsonl"
                try:
                    with JSONLWriter(output_path) as writer:
                        for failed_url in sorted(state.failed_urls):
                            result = self._bypass_engine.try_strategies(
                                url=failed_url,
                                block_type=last_block_type,
                                site_id=site_id,
                                timeout=30.0,
                            )
                            if result.success:
                                bypass_success = True
                                logger.info(
                                    "never_abandon_bypass_SUCCESS site_id=%s url=%s "
                                    "strategy=%s tier=%s latency=%.0fms",
                                    site_id, failed_url[:80],
                                    result.strategy_name, result.strategy_tier,
                                    result.latency_ms,
                                )
                                self._write_bypass_result(
                                    site_id, failed_url, result.html, writer,
                                )
                                recovered_urls.append(failed_url)
                            else:
                                # Update block type cache if detection changed
                                if result.block_detected:
                                    last_block_type = result.block_detected
                                    self._bypass_engine.update_block_cache(
                                        site_id, result.block_detected,
                                    )
                except Exception as e:
                    logger.warning(
                        "never_abandon_write_error site_id=%s error=%s",
                        site_id, str(e)[:100],
                    )

                for url in recovered_urls:
                    state.failed_urls.discard(url)
                    state.successful_urls.add(url)

                if bypass_success and not state.failed_urls:
                    logger.info(
                        "never_abandon_ALL_SUCCESS site_id=%s cycle=%s "
                        "— all URLs recovered via DynamicBypassEngine",
                        site_id, state.never_abandon_cycle,
                    )
                    state.never_abandon_active = False
                    return  # SUCCESS — mission accomplished

                if bypass_success:
                    logger.info(
                        "never_abandon_partial_success site_id=%s "
                        "remaining_failed=%s",
                        site_id, len(state.failed_urls),
                    )

            # Phase B: Fallback — full pipeline re-crawl with TotalWar
            state.current_strategy = StrategyMode.TOTALWAR
            state.current_round = 1
            state.exhausted = False

            if state.failed_urls:
                state.pending_urls = sorted(state.failed_urls)
                state.failed_urls = set()

            try:
                site_cfg = self._get_site_config(site_id)
                if site_cfg is None:
                    logger.error("never_abandon_no_config site_id=%s", site_id)
                    break

                output_path = self._output_dir / "all_articles.jsonl"
                with JSONLWriter(output_path, append=True) as writer:
                    result = self._crawl_site_with_retry(site_id, site_cfg, writer)

                if result.extracted_count > 0:
                    logger.info(
                        "never_abandon_SUCCESS site_id=%s cycle=%s "
                        "strategy=totalwar_fallback articles=%s",
                        site_id, state.never_abandon_cycle,
                        result.extracted_count,
                    )
                    state.never_abandon_active = False
                    return  # SUCCESS — mission accomplished
                else:
                    logger.warning(
                        "never_abandon_cycle_failed site_id=%s cycle=%s",
                        site_id, state.never_abandon_cycle,
                    )

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(
                    "never_abandon_cycle_error site_id=%s cycle=%s error=%s",
                    site_id, state.never_abandon_cycle, str(e)[:200],
                )

        # Safety cap reached — log final status but DO NOT mark as complete.
        # The site will be retried on the next daily run.
        # Also log DynamicBypassEngine stats for post-mortem analysis.
        if self._bypass_engine is not None:
            stats = self._bypass_engine.get_domain_stats(site_id)
            logger.error(
                "never_abandon_safety_cap_reached site_id=%s total_cycles=%s "
                "bypass_stats=%s — site will be retried on next daily run",
                site_id, state.never_abandon_cycle, stats,
            )
        else:
            logger.error(
                "never_abandon_safety_cap_reached site_id=%s total_cycles=%s "
                "— site will be retried on next daily run",
                site_id, state.never_abandon_cycle,
            )

    def _write_bypass_result(
        self,
        site_id: str,
        url: str,
        html: str,
        writer: JSONLWriter,
    ) -> None:
        """Extract article from bypassed HTML and write to JSONL.

        Applies the same validation pipeline as the normal crawl path:
        freshness check (24h lookback) and 3-level dedup (URL/Title/SimHash).

        Args:
            site_id: Site identifier.
            url: Original article URL.
            html: HTML content fetched via bypass strategy.
            writer: JSONL writer instance.
        """
        assert self._extractor is not None

        try:
            site_cfg = self._get_site_config(site_id) or {}
            article = self._extractor.extract(
                url=url,
                source_id=site_id,
                site_config=site_cfg,
                html=html,
            )
            if article is None:
                return

            # Freshness check — same 24h lookback as normal path
            if article.published_at is not None:
                pub_dt = article.published_at
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt < self._lookback_cutoff:
                    logger.debug(
                        "bypass_article_outside_24h url=%s published_at=%s",
                        url[:80], pub_dt.isoformat(),
                    )
                    return

            # 3-level dedup check — same cascade as normal path
            if self._dedup is not None:
                dedup_result = self._dedup.is_duplicate(
                    url=article.url,
                    title=article.title,
                    body=article.body,
                    source_id=site_id,
                    article_id=str(uuid.uuid4()),
                )
                if dedup_result.is_duplicate:
                    logger.debug(
                        "bypass_article_deduped url=%s level=%s reason=%s",
                        url[:80], dedup_result.level, dedup_result.reason[:80],
                    )
                    return

            writer.write_article(article)
            logger.debug(
                "bypass_article_written site_id=%s url=%s title=%s",
                site_id, url[:80], (article.title or "")[:60],
            )
        except Exception as e:
            logger.warning(
                "bypass_extraction_error site_id=%s url=%s error=%s",
                site_id, url[:80], str(e)[:100],
            )

    def _get_site_config(self, site_id: str) -> dict[str, Any] | None:
        """Get site configuration by site_id.

        Args:
            site_id: Site identifier.

        Returns:
            Site config dict, or None if not found.
        """
        try:
            return get_site_config(site_id)
        except KeyError:
            return None

    # -----------------------------------------------------------------------
    # Single Pass (all sites, with Level 2+3 retry per site)
    # -----------------------------------------------------------------------

    def _run_single_pass(
        self,
        target_sites: dict[str, dict[str, Any]],
        restart_number: int,
    ) -> list[CrawlResult]:
        """Run a single pass through all target sites.

        For each site, runs the full per-site pipeline with Level 2
        (Standard/TotalWar) and Level 3 (rounds) retry.

        Args:
            target_sites: Sites to crawl.
            restart_number: Current Level 4 restart number.

        Returns:
            List of CrawlResult, one per site.
        """
        results: list[CrawlResult] = []
        total = len(target_sites)

        # Open consolidated JSONL writer for all sites
        output_path = self._output_dir / "all_articles.jsonl"

        with JSONLWriter(output_path) as writer:
            for idx, (site_id, site_cfg) in enumerate(target_sites.items(), 1):
                logger.info(
                    "crawl_site_start site=%s (%s/%s) restart=%s",
                    site_id, idx, total, restart_number,
                )

                # Skip already-completed sites
                assert self._crawl_state is not None
                if self._crawl_state.is_site_complete(site_id):
                    logger.info("site_already_complete site_id=%s", site_id)
                    results.append(CrawlResult(source_id=site_id))
                    continue

                # Skip disabled sites — D-7 (13): ENABLED_DEFAULT from constants.py (SOT)
                if not site_cfg.get("meta", {}).get("enabled", ENABLED_DEFAULT):
                    logger.info("site_disabled site_id=%s", site_id)
                    results.append(CrawlResult(source_id=site_id))
                    continue

                # Check circuit breaker — Crawling Absolute Principle (크롤링 절대 원칙):
                # NEVER abandon a site. When circuit breaker is OPEN, force
                # immediate HALF_OPEN probe with maximum escalation (TotalWar).
                assert self._circuit_breakers is not None
                if not self._circuit_breakers.is_allowed(site_id):
                    if CRAWL_NEVER_ABANDON:
                        logger.warning(
                            "circuit_breaker_open_force_probe site_id=%s "
                            "state=%s policy=never_abandon",
                            site_id, self._circuit_breakers.get_state(site_id).value,
                        )
                        self._circuit_breakers.force_half_open(site_id)
                        # Fall through to _crawl_site_with_retry which will
                        # escalate to TotalWar mode on failures
                    else:
                        logger.warning(
                            "circuit_breaker_open site_id=%s state=%s",
                            site_id, self._circuit_breakers.get_state(site_id).value,
                        )
                        results.append(CrawlResult(
                            source_id=site_id,
                            errors=["Circuit breaker OPEN -- site skipped"],
                        ))
                        continue

                try:
                    result = self._crawl_site_with_retry(
                        site_id, site_cfg, writer,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(
                        "site_crawl_fatal site_id=%s error=%s error_type=%s",
                        site_id, str(e), type(e).__name__,
                    )
                    result = CrawlResult(
                        source_id=site_id,
                        errors=[f"Fatal: {type(e).__name__}: {e}"],
                    )

                results.append(result)

                logger.info(
                    "crawl_site_complete site=%s articles=%s discovered=%s failed=%s elapsed=%ss",
                    site_id, result.extracted_count, result.discovered_urls,
                    result.failed_count, round(result.elapsed_seconds, 1),
                )

        return results

    # -----------------------------------------------------------------------
    # Per-Site Crawl with Level 2+3 Retry
    # -----------------------------------------------------------------------

    def _crawl_site_with_retry(
        self,
        site_id: str,
        site_cfg: dict[str, Any],
        writer: JSONLWriter,
    ) -> CrawlResult:
        """Crawl a single site with Level 2 and Level 3 retry.

        Level 2: Standard mode first, then TotalWar if >50% failure.
        Level 3: Up to 3 rounds, with increasing delays between rounds.

        Args:
            site_id: Site identifier.
            site_cfg: Site configuration from sources.yaml.
            writer: JSONL writer for article output.

        Returns:
            CrawlResult with all statistics.
        """
        start_time = time.monotonic()
        result = CrawlResult(source_id=site_id)

        # Configure NetworkGuard for this site
        assert self._guard is not None
        crawl_config = site_cfg.get("crawl", {})
        rate_limit = crawl_config.get("rate_limit_seconds", DEFAULT_RATE_LIMIT_SECONDS)
        jitter = crawl_config.get("jitter_seconds", 0)

        self._guard.configure_site(
            site_id,
            rate_limit_seconds=rate_limit,
            jitter_seconds=jitter,
        )

        # Level 3: Crawler rounds
        assert self._retry_manager is not None

        for round_num in range(1, L3_MAX_ROUNDS + 1):
            retry_state = self._retry_manager.get_state(site_id)
            retry_state.current_round = round_num

            logger.info(
                "crawl_round_start site_id=%s round=%s/%s",
                site_id, round_num, L3_MAX_ROUNDS,
            )

            # Phase 1: URL Discovery
            discovered = self._discover_urls(site_id, site_cfg)
            if not discovered:
                logger.warning("no_urls_discovered site_id=%s round=%s", site_id, round_num)
                if round_num == 1:
                    result.elapsed_seconds = time.monotonic() - start_time
                    return result
                break

            # Filter out already-processed URLs
            new_urls = self._filter_processed_urls(site_id, discovered)
            if not new_urls:
                logger.info("all_urls_processed site_id=%s round=%s", site_id, round_num)
                break

            result.discovered_urls = max(result.discovered_urls, len(discovered))

            # Initialize retry state with pending URLs
            self._retry_manager.init_site(
                site_id,
                [u.url for u in new_urls],
            )

            # Level 2: Standard mode
            round_result = self._crawl_urls(
                site_id, site_cfg, new_urls, writer,
                strategy=StrategyMode.STANDARD,
                rate_limit=rate_limit,
            )
            self._merge_result(result, round_result)

            # Level 2: TotalWar escalation if >50% failure
            if self._retry_manager.should_escalate_to_totalwar(site_id):
                self._retry_manager.escalate_to_totalwar(site_id)
                retry_state = self._retry_manager.get_state(site_id)

                if retry_state.pending_urls:
                    # Re-fetch failed URLs with TotalWar settings
                    failed_url_objs = [
                        url_obj for url_obj in new_urls
                        if url_obj.url in set(retry_state.pending_urls)
                    ]

                    totalwar_result = self._crawl_urls(
                        site_id, site_cfg, failed_url_objs, writer,
                        strategy=StrategyMode.TOTALWAR,
                        rate_limit=rate_limit * TOTALWAR_DELAY_MULTIPLIER,
                    )
                    self._merge_result(result, totalwar_result)

            # Check if we need another round
            retry_state = self._retry_manager.get_state(site_id)
            if not retry_state.failed_urls:
                logger.info("all_urls_successful site_id=%s round=%s", site_id, round_num)
                break

            if round_num < L3_MAX_ROUNDS:
                # Delay before next round
                if self._retry_manager.should_start_new_round(site_id):
                    delay = self._retry_manager.start_new_round(site_id)
                    logger.info(
                        "round_delay site_id=%s delay=%ss next_round=%s",
                        site_id, delay, round_num + 1,
                    )
                    time.sleep(min(delay, 30.0))  # Cap actual sleep for usability
                else:
                    break

        # Finalize
        assert self._crawl_state is not None
        if result.extracted_count > 0:
            self._crawl_state.mark_site_complete(site_id)
        self._crawl_state.save()

        result.elapsed_seconds = time.monotonic() - start_time

        # Check exhaustion for Tier 6 escalation
        if self._retry_manager.is_exhausted(site_id):
            self._retry_manager.escalate_tier6(site_id)
            result.tier_used = 6

        return result

    # -----------------------------------------------------------------------
    # URL Discovery
    # -----------------------------------------------------------------------

    def _discover_urls(
        self,
        site_id: str,
        site_cfg: dict[str, Any],
    ) -> list[DiscoveredURL]:
        """Run URL discovery for a site.

        Uses the 3-tier discovery pipeline: RSS -> Sitemap -> DOM.
        Also tries adapter-provided section URLs for DOM discovery.

        Args:
            site_id: Site identifier.
            site_cfg: Site configuration.

        Returns:
            List of discovered URLs.
        """
        assert self._url_discovery is not None

        try:
            discovered = self._url_discovery.discover(site_cfg, site_id)
        except Exception as e:
            logger.error(
                "discovery_error site_id=%s error=%s error_type=%s",
                site_id, str(e), type(e).__name__,
            )
            discovered = []

        logger.info(
            "urls_discovered site_id=%s count=%s",
            site_id, len(discovered),
        )
        return discovered

    def _filter_processed_urls(
        self,
        site_id: str,
        discovered: list[DiscoveredURL],
    ) -> list[DiscoveredURL]:
        """Filter out URLs that have already been processed or deduped.

        Checks against:
        1. CrawlState (URLs processed in previous runs)
        2. DedupEngine (URLs in the dedup database)

        Args:
            site_id: Site identifier.
            discovered: All discovered URLs.

        Returns:
            Filtered list of new URLs to process.
        """
        assert self._crawl_state is not None
        assert self._dedup is not None

        new_urls: list[DiscoveredURL] = []

        for url_obj in discovered:
            # Check CrawlState
            if self._crawl_state.is_url_processed(site_id, url_obj.url):
                continue

            # Check DedupEngine (URL-level only -- fast O(1) check)
            # check_only=True prevents premature URL registration that would
            # cause the extraction phase to reject the article as a duplicate.
            try:
                dedup_result = self._dedup.is_duplicate(
                    url=url_obj.url,
                    title="",
                    body="",
                    source_id=site_id,
                    article_id="",
                    check_only=True,
                )
                if dedup_result.is_duplicate and dedup_result.level == 1:
                    continue
            except Exception:
                pass  # Dedup check failure should not block crawling

            new_urls.append(url_obj)

        filtered_count = len(discovered) - len(new_urls)
        if filtered_count > 0:
            logger.info(
                "urls_filtered site_id=%s total=%s new=%s deduped=%s",
                site_id, len(discovered), len(new_urls), filtered_count,
            )

        return new_urls

    # -----------------------------------------------------------------------
    # Article Extraction
    # -----------------------------------------------------------------------

    def _crawl_urls(
        self,
        site_id: str,
        site_cfg: dict[str, Any],
        urls: list[DiscoveredURL],
        writer: JSONLWriter,
        strategy: int = StrategyMode.STANDARD,
        rate_limit: float = DEFAULT_RATE_LIMIT_SECONDS,
    ) -> CrawlResult:
        """Fetch and extract articles for a list of URLs.

        Args:
            site_id: Site identifier.
            site_cfg: Site configuration.
            urls: URLs to process.
            writer: JSONL writer for output.
            strategy: Current strategy mode (Standard or TotalWar).
            rate_limit: Rate limit for this strategy.

        Returns:
            CrawlResult with extraction statistics.
        """
        assert self._extractor is not None
        assert self._dedup is not None
        assert self._ua_manager is not None
        assert self._retry_manager is not None
        assert self._crawl_state is not None
        assert self._circuit_breakers is not None
        assert self._guard is not None

        result = CrawlResult(source_id=site_id)
        source_name = site_cfg.get("name", site_id)

        strategy_name = "Standard" if strategy == StrategyMode.STANDARD else "TotalWar"
        logger.info(
            "crawl_urls_start site_id=%s urls=%s strategy=%s",
            site_id, len(urls), strategy_name,
        )

        # Apply TotalWar rate limit override
        if strategy == StrategyMode.TOTALWAR:
            self._guard.configure_site(
                site_id,
                rate_limit_seconds=rate_limit,
                jitter_seconds=rate_limit * 0.3,
            )

        for url_obj in urls:
            # Check article cap
            if result.extracted_count >= self._max_articles:
                logger.info(
                    "max_articles_reached site_id=%s limit=%s",
                    site_id, self._max_articles,
                )
                break

            # Check circuit breaker
            if not self._circuit_breakers.is_allowed(site_id):
                logger.warning("circuit_breaker_tripped site_id=%s", site_id)
                result.errors.append("Circuit breaker opened during crawl")
                break

            # Apply UA rotation
            ua_string = self._ua_manager.get_ua(site_id)

            # Attempt extraction
            try:
                article = self._extractor.extract(
                    url=url_obj.url,
                    source_id=site_id,
                    site_config=site_cfg,
                    title_hint=url_obj.title_hint,
                )

                # Override crawl_method from discovery
                # RawArticle is frozen, so we create a new one with the right method.
                # Preserve non-discovery crawl_methods set by ArticleExtractor:
                #   "adaptive" = browser render + adaptive CSS selector extraction
                #   "playwright" = browser render + standard extraction chain
                # Overwriting these with discovered_via would lose paywall bypass tracking.
                effective_method = (
                    article.crawl_method
                    if article.crawl_method in ("adaptive", "playwright")
                    else url_obj.discovered_via
                )
                article = RawArticle(
                    url=article.url,
                    title=article.title,
                    body=article.body,
                    source_id=article.source_id,
                    source_name=article.source_name,
                    language=article.language,
                    published_at=article.published_at,
                    crawled_at=article.crawled_at,
                    author=article.author,
                    category=article.category,
                    content_hash=article.content_hash,
                    crawl_tier=article.crawl_tier,
                    crawl_method=effective_method,
                    is_paywall_truncated=article.is_paywall_truncated,
                )

                # 24-hour lookback filter (absolute rule for daily runs).
                # Drop articles published before the cutoff even if they
                # passed the coarser URL-discovery date filter.
                # NOTE: article.published_at is datetime|None (not str).
                if article.published_at is not None:
                    pub_dt = article.published_at
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    if pub_dt < self._lookback_cutoff:
                        result.skipped_freshness_count += 1
                        # Extraction succeeded — record success so circuit
                        # breakers and retry manager stay accurate.
                        self._retry_manager.mark_url_success(site_id, url_obj.url)
                        self._circuit_breakers.record_success(site_id)
                        logger.debug(
                            "article_outside_24h url=%s published_at=%s cutoff=%s",
                            url_obj.url[:80],
                            pub_dt.isoformat(),
                            self._lookback_cutoff.isoformat(),
                        )
                        continue

                # Content-level dedup check
                article_id = str(uuid.uuid4())
                dedup_result = self._dedup.is_duplicate(
                    url=article.url,
                    title=article.title,
                    body=article.body,
                    source_id=site_id,
                    article_id=article_id,
                )

                if dedup_result.is_duplicate:
                    result.skipped_dedup_count += 1
                    logger.debug(
                        "article_deduped url=%s level=%s reason=%s",
                        url_obj.url[:80], dedup_result.level, dedup_result.reason[:80],
                    )
                else:
                    # Write to JSONL
                    writer.write_article(article)
                    result.articles.append(article)
                    result.extracted_count += 1

                # Mark success
                self._retry_manager.mark_url_success(site_id, url_obj.url)
                self._circuit_breakers.record_success(site_id)

            except ParseError as e:
                result.failed_count += 1
                result.errors.append(f"Parse: {url_obj.url}: {e}")
                self._retry_manager.handle_url_failure(
                    site_id, url_obj.url,
                    error_type="ParseError", error_msg=str(e),
                )
                logger.warning(
                    "extraction_parse_error url=%s site_id=%s error=%s",
                    url_obj.url[:80], site_id, str(e)[:200],
                )

            except NetworkError as e:
                result.failed_count += 1
                result.errors.append(f"Network: {url_obj.url}: {e}")
                self._retry_manager.handle_url_failure(
                    site_id, url_obj.url,
                    error_type="NetworkError", error_msg=str(e),
                )
                self._circuit_breakers.record_failure(site_id, "network_error")
                logger.warning(
                    "extraction_network_error url=%s site_id=%s error=%s",
                    url_obj.url[:80], site_id, str(e)[:200],
                )

            except BlockDetectedError as e:
                result.failed_count += 1
                result.tier_used = max(result.tier_used, 2)
                result.errors.append(f"Blocked: {url_obj.url}: {e}")
                self._retry_manager.handle_url_failure(
                    site_id, url_obj.url,
                    error_type="BlockDetectedError", error_msg=str(e),
                )
                self._circuit_breakers.record_failure(site_id, e.block_type)
                logger.warning(
                    "extraction_blocked url=%s site_id=%s block_type=%s",
                    url_obj.url[:80], site_id, e.block_type,
                )

            except RateLimitError as e:
                result.failed_count += 1
                result.errors.append(f"RateLimit: {url_obj.url}: {e}")
                self._retry_manager.handle_url_failure(
                    site_id, url_obj.url,
                    error_type="RateLimitError", error_msg=str(e),
                )
                logger.warning(
                    "extraction_rate_limited url=%s site_id=%s",
                    url_obj.url[:80], site_id,
                )
                # Respect rate limit -- wait
                if e.retry_after:
                    time.sleep(min(e.retry_after, 60.0))

            except Exception as e:
                result.failed_count += 1
                result.errors.append(f"Unexpected: {url_obj.url}: {type(e).__name__}: {e}")
                self._retry_manager.handle_url_failure(
                    site_id, url_obj.url,
                    error_type=type(e).__name__, error_msg=str(e),
                )
                logger.error(
                    "extraction_unexpected url=%s site_id=%s error=%s error_type=%s",
                    url_obj.url[:80], site_id, str(e)[:200], type(e).__name__,
                )

            # Mark URL as processed (for CrawlState resume)
            self._crawl_state.mark_url_processed(site_id, url_obj.url)

        return result

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _merge_result(target: CrawlResult, source: CrawlResult) -> None:
        """Merge a source CrawlResult into a target.

        Adds statistics from source into target without creating a new object.

        Args:
            target: Result to merge into.
            source: Result to merge from.
        """
        target.articles.extend(source.articles)
        target.extracted_count += source.extracted_count
        target.failed_count += source.failed_count
        target.skipped_dedup_count += source.skipped_dedup_count
        target.skipped_freshness_count += source.skipped_freshness_count
        target.tier_used = max(target.tier_used, source.tier_used)
        target.errors.extend(source.errors)


# ---------------------------------------------------------------------------
# Entry point for main.py
# ---------------------------------------------------------------------------

def run_crawl_pipeline(
    crawl_date: str | None = None,
    sites: list[str] | None = None,
    groups: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Convenience function for running the crawling pipeline.

    Called from ``main.py cmd_crawl()``.

    Args:
        crawl_date: Target date (YYYY-MM-DD), defaults to today.
        sites: Optional list of site IDs to crawl.
        groups: Optional list of group letters to crawl.
        dry_run: If True, validate config without making requests.

    Returns:
        Crawl report dictionary.
    """
    with CrawlingPipeline(
        crawl_date=crawl_date,
        sites_filter=sites,
        groups_filter=groups,
        dry_run=dry_run,
    ) as pipeline:
        return pipeline.run()
