"""YAML configuration loading and validation for sources.yaml and pipeline.yaml.

Provides a singleton-pattern config loader that reads, validates, and caches
configuration from YAML files. Validation rules are derived from Step 5
architecture blueprint Section 5c (sources) and 5d (pipeline).

Usage:
    from src.utils.config_loader import load_sources_config, load_pipeline_config
    sources = load_sources_config()
    pipeline = load_pipeline_config()
"""

from pathlib import Path
from typing import Any

import yaml

from src.config.constants import (
    SOURCES_YAML_PATH,
    PIPELINE_YAML_PATH,
    VALID_REGIONS,
    VALID_LANGUAGES,
    VALID_GROUPS,
    VALID_CRAWL_METHODS,
    VALID_PAYWALL_TYPES,
    VALID_DIFFICULTY_TIERS,
    VALID_BOT_BLOCK_LEVELS,
    VALID_PARQUET_COMPRESSIONS,
    MAX_MEMORY_GB,
    DEFAULT_RATE_LIMIT_SECONDS,
    ENABLED_DEFAULT,
)


# Sensible defaults for optional site configuration fields.
# Applied during load to avoid bloating sources.yaml while ensuring
# all downstream code receives a complete configuration.
#
# D-7: These values mirror adapter class attributes (BaseSiteAdapter subclasses).
# Adapter class attrs are code-level documentation; these config defaults are
# the runtime truth used by pipeline.py (line ~935) and network_guard.py.
# When adapter defaults change, re-run the enrichment script or update here.
# Cross-ref: src/crawling/adapters/base_adapter.py class attributes.
_SOURCE_DEFAULTS: dict[str, dict[str, Any]] = {
    "crawl": {
        "rate_limit_seconds": DEFAULT_RATE_LIMIT_SECONDS,
        "max_requests_per_hour": 720,
        "jitter_seconds": 0,
    },
    "anti_block": {
        "bot_block_level": "LOW",
        "default_escalation_tier": 1,
        "max_escalation_tier": 5,
        "requires_proxy": False,
    },
    "extraction": {
        "paywall_type": "none",
        "title_only": False,
        "rendering_required": False,
        "charset": "utf-8",
    },
    "meta": {
        "difficulty_tier": "Medium",
        "daily_article_estimate": 50,
        "sections_count": 5,
        # D-7 (13): opt-out pattern — import from constants.py (SOT)
        "enabled": ENABLED_DEFAULT,
    },
}


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration validation failed with {len(errors)} error(s):\n" +
                         "\n".join(f"  - {e}" for e in errors))


# Module-level cache for singleton pattern
_sources_cache: dict[str, Any] | None = None
_pipeline_cache: dict[str, Any] | None = None


def _normalize_sources(config: dict[str, Any]) -> dict[str, Any]:
    """Apply default values to each site in sources config.

    Deep-merges _SOURCE_DEFAULTS into each site entry, filling only
    missing keys. Existing values are never overwritten.

    Args:
        config: Raw parsed sources.yaml content.

    Returns:
        Config with defaults applied (mutates in-place and returns).
    """
    sources = config.get("sources", {})
    for site in sources.values():
        if not isinstance(site, dict):
            continue
        for section, defaults in _SOURCE_DEFAULTS.items():
            if section not in site:
                site[section] = {}
            sec = site[section]
            if not isinstance(sec, dict):
                continue
            for key, default_val in defaults.items():
                if key not in sec:
                    sec[key] = default_val
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level of {path}, got {type(data).__name__}")
    return data


def validate_sources_config(config: dict[str, Any]) -> list[str]:
    """Validate sources.yaml configuration against schema rules.

    Validation rules from Step 5, Section 5c:
        - source_id: lowercase alphanumeric + underscore, unique
        - name: non-empty string
        - url: valid URL starting with http:// or https://
        - region: one of VALID_REGIONS
        - language: ISO 639-1 code
        - group: A-J
        - crawl.primary_method: rss/sitemap/api/playwright/dom
        - crawl.rate_limit_seconds: integer >= 1
        - anti_block.ua_tier: 1-4
        - extraction.paywall_type: none/soft-metered/hard
        - meta.difficulty_tier: Easy/Medium/Hard/Extreme
        - meta.enabled: boolean

    Args:
        config: Parsed sources.yaml content.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    sources = config.get("sources", {})

    if not sources:
        errors.append("No 'sources' key found in configuration")
        return errors

    seen_ids: set[str] = set()

    for source_id, site in sources.items():
        prefix = f"sources.{source_id}"

        # source_id validation
        if not isinstance(source_id, str) or not source_id.replace("_", "").replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").isalpha():
            if not all(c.isalnum() or c == "_" for c in source_id):
                errors.append(f"{prefix}: source_id must be lowercase alphanumeric + underscore")
        if source_id in seen_ids:
            errors.append(f"{prefix}: duplicate source_id")
        seen_ids.add(source_id)

        if not isinstance(site, dict):
            errors.append(f"{prefix}: expected dict, got {type(site).__name__}")
            continue

        # Required top-level fields
        if not site.get("name"):
            errors.append(f"{prefix}: 'name' is required and must be non-empty")

        url = site.get("url", "")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            errors.append(f"{prefix}: 'url' must start with http:// or https://")

        region_val = (site.get("region") or "").lower()
        if region_val not in VALID_REGIONS:
            errors.append(f"{prefix}: 'region' must be one of {VALID_REGIONS}, got '{site.get('region')}'")

        if site.get("language") not in VALID_LANGUAGES:
            errors.append(f"{prefix}: 'language' must be one of {VALID_LANGUAGES}, got '{site.get('language')}'")

        if site.get("group") not in VALID_GROUPS:
            errors.append(f"{prefix}: 'group' must be one of {VALID_GROUPS}, got '{site.get('group')}'")

        # Crawl configuration
        crawl = site.get("crawl", {})
        if not isinstance(crawl, dict):
            errors.append(f"{prefix}.crawl: expected dict")
        else:
            if crawl.get("primary_method") not in VALID_CRAWL_METHODS:
                errors.append(f"{prefix}.crawl.primary_method: must be one of {VALID_CRAWL_METHODS}")
            fallbacks = crawl.get("fallback_methods", [])
            if not isinstance(fallbacks, list):
                errors.append(f"{prefix}.crawl.fallback_methods: must be a list")
            else:
                for fb in fallbacks:
                    if fb not in VALID_CRAWL_METHODS:
                        errors.append(f"{prefix}.crawl.fallback_methods: '{fb}' not in {VALID_CRAWL_METHODS}")
            rate = crawl.get("rate_limit_seconds")
            if not isinstance(rate, (int, float)) or rate < 1:
                errors.append(f"{prefix}.crawl.rate_limit_seconds: must be >= 1")

        # Anti-block configuration
        anti_block = site.get("anti_block", {})
        if isinstance(anti_block, dict):
            ua_tier = anti_block.get("ua_tier")
            if ua_tier not in (1, 2, 3, 4):
                errors.append(f"{prefix}.anti_block.ua_tier: must be 1-4, got {ua_tier}")
            bot_level = anti_block.get("bot_block_level")
            if bot_level not in VALID_BOT_BLOCK_LEVELS:
                errors.append(f"{prefix}.anti_block.bot_block_level: must be one of {VALID_BOT_BLOCK_LEVELS}")

        # Extraction configuration
        extraction = site.get("extraction", {})
        if isinstance(extraction, dict):
            paywall = extraction.get("paywall_type")
            if paywall not in VALID_PAYWALL_TYPES:
                errors.append(f"{prefix}.extraction.paywall_type: must be one of {VALID_PAYWALL_TYPES}")

        # Meta configuration
        meta = site.get("meta", {})
        if isinstance(meta, dict):
            tier = meta.get("difficulty_tier")
            if tier not in VALID_DIFFICULTY_TIERS:
                errors.append(f"{prefix}.meta.difficulty_tier: must be one of {VALID_DIFFICULTY_TIERS}")
            estimate = meta.get("daily_article_estimate")
            if not isinstance(estimate, int) or estimate < 0:
                errors.append(f"{prefix}.meta.daily_article_estimate: must be integer >= 0")
            if not isinstance(meta.get("enabled"), bool):
                errors.append(f"{prefix}.meta.enabled: must be boolean")

    return errors


def validate_pipeline_config(config: dict[str, Any]) -> list[str]:
    """Validate pipeline.yaml configuration against schema rules.

    Validation rules from Step 5, Section 5d:
        - stages.*.enabled: boolean
        - stages.*.input_format: jsonl or parquet
        - stages.*.output_format: parquet or parquet+sqlite
        - stages.*.parallelism: integer >= 1
        - stages.*.memory_limit_gb: float > 0
        - stages.*.timeout_seconds: integer >= 60
        - stages.*.dependencies: list of valid stage names (no cycles)
        - global.max_memory_gb: float > 0, <= MAX_MEMORY_GB
        - global.parquet_compression: zstd/snappy/lz4/none

    Args:
        config: Parsed pipeline.yaml content.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    pipeline = config.get("pipeline", {})

    if not pipeline:
        errors.append("No 'pipeline' key found in configuration")
        return errors

    # Global settings
    global_cfg = pipeline.get("global", {})
    max_mem = global_cfg.get("max_memory_gb")
    if not isinstance(max_mem, (int, float)) or max_mem <= 0 or max_mem > MAX_MEMORY_GB:
        errors.append(f"pipeline.global.max_memory_gb: must be > 0 and <= {MAX_MEMORY_GB} (host limit)")

    compression = global_cfg.get("parquet_compression")
    if compression not in VALID_PARQUET_COMPRESSIONS:
        errors.append(f"pipeline.global.parquet_compression: must be one of {VALID_PARQUET_COMPRESSIONS}")

    # Stage validation
    stages = pipeline.get("stages", {})
    valid_stage_names = set(stages.keys())
    valid_input_formats = {"jsonl", "parquet"}
    valid_output_formats = {"parquet", "parquet+sqlite"}

    for stage_name, stage_cfg in stages.items():
        prefix = f"pipeline.stages.{stage_name}"

        if not isinstance(stage_cfg.get("enabled"), bool):
            errors.append(f"{prefix}.enabled: must be boolean")

        input_fmt = stage_cfg.get("input_format")
        if input_fmt not in valid_input_formats:
            errors.append(f"{prefix}.input_format: must be one of {valid_input_formats}")

        output_fmt = stage_cfg.get("output_format")
        if output_fmt not in valid_output_formats:
            errors.append(f"{prefix}.output_format: must be one of {valid_output_formats}")

        parallelism = stage_cfg.get("parallelism")
        if not isinstance(parallelism, int) or parallelism < 1:
            errors.append(f"{prefix}.parallelism: must be integer >= 1")

        mem_limit = stage_cfg.get("memory_limit_gb")
        if not isinstance(mem_limit, (int, float)) or mem_limit <= 0:
            errors.append(f"{prefix}.memory_limit_gb: must be float > 0")

        timeout = stage_cfg.get("timeout_seconds")
        if not isinstance(timeout, int) or timeout < 60:
            errors.append(f"{prefix}.timeout_seconds: must be integer >= 60")

        deps = stage_cfg.get("dependencies", [])
        if not isinstance(deps, list):
            errors.append(f"{prefix}.dependencies: must be a list")
        else:
            for dep in deps:
                if dep not in valid_stage_names:
                    errors.append(f"{prefix}.dependencies: '{dep}' is not a valid stage name")

    return errors


def load_sources_config(
    path: Path | None = None,
    validate: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load and optionally validate sources.yaml.

    Args:
        path: Path to sources.yaml (defaults to SOURCES_YAML_PATH).
        validate: Whether to run validation (raises on failure).
        use_cache: Whether to use the module-level cache.

    Returns:
        Parsed and validated sources configuration.

    Raises:
        ConfigValidationError: If validation is enabled and fails.
    """
    global _sources_cache
    if use_cache and _sources_cache is not None:
        return _sources_cache

    config = _load_yaml(path or SOURCES_YAML_PATH)
    config = _normalize_sources(config)

    if validate:
        errors = validate_sources_config(config)
        if errors:
            raise ConfigValidationError(errors)

    if use_cache:
        _sources_cache = config
    return config


def load_pipeline_config(
    path: Path | None = None,
    validate: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Load and optionally validate pipeline.yaml.

    Args:
        path: Path to pipeline.yaml (defaults to PIPELINE_YAML_PATH).
        validate: Whether to run validation (raises on failure).
        use_cache: Whether to use the module-level cache.

    Returns:
        Parsed and validated pipeline configuration.

    Raises:
        ConfigValidationError: If validation is enabled and fails.
    """
    global _pipeline_cache
    if use_cache and _pipeline_cache is not None:
        return _pipeline_cache

    config = _load_yaml(path or PIPELINE_YAML_PATH)

    if validate:
        errors = validate_pipeline_config(config)
        if errors:
            raise ConfigValidationError(errors)

    if use_cache:
        _pipeline_cache = config
    return config


def get_site_config(source_id: str) -> dict[str, Any]:
    """Get configuration for a specific site by source_id.

    Args:
        source_id: The unique site identifier (e.g., "chosun").

    Returns:
        Site configuration dictionary.

    Raises:
        KeyError: If source_id is not found in sources.yaml.
    """
    sources = load_sources_config()
    sites = sources.get("sources", {})
    if source_id not in sites:
        raise KeyError(f"Site '{source_id}' not found in sources.yaml. "
                       f"Available: {sorted(sites.keys())}")
    return sites[source_id]


def get_stage_config(stage_name: str) -> dict[str, Any]:
    """Get configuration for a specific pipeline stage.

    Args:
        stage_name: The stage identifier (e.g., "stage_1_preprocessing").

    Returns:
        Stage configuration dictionary.

    Raises:
        KeyError: If stage_name is not found in pipeline.yaml.
    """
    pipeline = load_pipeline_config()
    stages = pipeline.get("pipeline", {}).get("stages", {})
    if stage_name not in stages:
        raise KeyError(f"Stage '{stage_name}' not found in pipeline.yaml. "
                       f"Available: {sorted(stages.keys())}")
    return stages[stage_name]


def get_enabled_sites() -> list[str]:
    """Get list of enabled site source_ids.

    Opt-out pattern: sites without ``meta.enabled`` default to enabled.
    Only sites with ``meta.enabled: false`` are excluded.

    D-7: Default matches ``_SOURCE_DEFAULTS["meta"]["enabled"]`` (True)
    and ``pipeline.py _resolve_target_sites()`` (default True).

    Returns:
        List of source_id strings where meta.enabled is not False.
    """
    sources = load_sources_config()
    # D-7 (13): opt-out pattern — ENABLED_DEFAULT from constants.py (SOT)
    return [
        sid for sid, cfg in sources.get("sources", {}).items()
        if cfg.get("meta", {}).get("enabled", ENABLED_DEFAULT)
    ]


def get_sites_by_group(group: str) -> list[str]:
    """Get site source_ids belonging to a specific group.

    Args:
        group: Group letter (A-J).

    Returns:
        List of source_id strings in the specified group.
    """
    sources = load_sources_config()
    return [
        sid for sid, cfg in sources.get("sources", {}).items()
        if cfg.get("group") == group
    ]


def clear_config_cache() -> None:
    """Clear the configuration cache, forcing reload on next access."""
    global _sources_cache, _pipeline_cache
    _sources_cache = None
    _pipeline_cache = None
