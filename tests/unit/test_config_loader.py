"""Tests for src.utils.config_loader -- YAML config loading and validation."""

import sys
from pathlib import Path

import pytest
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import (
    validate_sources_config,
    validate_pipeline_config,
    load_sources_config,
    load_pipeline_config,
    ConfigValidationError,
    clear_config_cache,
    _normalize_sources,
    _SOURCE_DEFAULTS,
    get_enabled_sites,
)


class TestSourcesValidation:
    """Test sources.yaml validation logic."""

    def test_valid_minimal_config(self, sample_sources_yaml):
        """A minimal valid config should produce no errors."""
        with open(sample_sources_yaml) as f:
            config = yaml.safe_load(f)
        errors = validate_sources_config(config)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_sources_key(self):
        """Config without 'sources' key should fail."""
        errors = validate_sources_config({"version": "1.0"})
        assert len(errors) == 1
        assert "No 'sources' key" in errors[0]

    def test_invalid_region(self):
        """Invalid region code should be flagged."""
        config = {
            "sources": {
                "test": {
                    "name": "Test",
                    "url": "https://example.com",
                    "region": "xx",  # Invalid
                    "language": "en",
                    "group": "E",
                    "crawl": {
                        "primary_method": "rss",
                        "fallback_methods": [],
                        "rate_limit_seconds": 5,
                    },
                    "anti_block": {
                        "ua_tier": 2,
                        "bot_block_level": "MEDIUM",
                    },
                    "extraction": {"paywall_type": "none"},
                    "meta": {
                        "difficulty_tier": "Easy",
                        "daily_article_estimate": 100,
                        "enabled": True,
                    },
                }
            }
        }
        errors = validate_sources_config(config)
        assert any("region" in e for e in errors)

    def test_invalid_url(self):
        """URL not starting with http(s):// should be flagged."""
        config = {
            "sources": {
                "test": {
                    "name": "Test",
                    "url": "ftp://example.com",  # Invalid
                    "region": "us",
                    "language": "en",
                    "group": "E",
                    "crawl": {
                        "primary_method": "rss",
                        "fallback_methods": [],
                        "rate_limit_seconds": 5,
                    },
                    "anti_block": {
                        "ua_tier": 2,
                        "bot_block_level": "MEDIUM",
                    },
                    "extraction": {"paywall_type": "none"},
                    "meta": {
                        "difficulty_tier": "Easy",
                        "daily_article_estimate": 100,
                        "enabled": True,
                    },
                }
            }
        }
        errors = validate_sources_config(config)
        assert any("url" in e.lower() for e in errors)


class TestPipelineValidation:
    """Test pipeline.yaml validation logic."""

    def test_valid_minimal_config(self, sample_pipeline_yaml):
        """A minimal valid pipeline config should produce no errors."""
        with open(sample_pipeline_yaml) as f:
            config = yaml.safe_load(f)
        errors = validate_pipeline_config(config)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_missing_pipeline_key(self):
        """Config without 'pipeline' key should fail."""
        errors = validate_pipeline_config({"version": "1.0"})
        assert len(errors) == 1
        assert "No 'pipeline' key" in errors[0]

    def test_memory_exceeds_limit(self):
        """Memory limit > 10 GB should be flagged."""
        config = {
            "pipeline": {
                "global": {
                    "max_memory_gb": 20,  # Exceeds 10 GB limit
                    "parquet_compression": "zstd",
                },
                "stages": {},
            }
        }
        errors = validate_pipeline_config(config)
        assert any("max_memory_gb" in e for e in errors)

    def test_invalid_compression(self):
        """Invalid compression type should be flagged."""
        config = {
            "pipeline": {
                "global": {
                    "max_memory_gb": 10,
                    "parquet_compression": "bzip2",  # Invalid
                },
                "stages": {},
            }
        }
        errors = validate_pipeline_config(config)
        assert any("parquet_compression" in e for e in errors)


class TestConfigLoading:
    """Test config file loading functionality."""

    def test_load_sources_from_file(self, sample_sources_yaml):
        """Loading a valid sources.yaml should succeed."""
        clear_config_cache()
        config = load_sources_config(path=sample_sources_yaml, use_cache=False)
        assert "sources" in config
        assert "test_site" in config["sources"]

    def test_load_pipeline_from_file(self, sample_pipeline_yaml):
        """Loading a valid pipeline.yaml should succeed."""
        clear_config_cache()
        config = load_pipeline_config(path=sample_pipeline_yaml, use_cache=False)
        assert "pipeline" in config
        assert "stages" in config["pipeline"]

    def test_load_nonexistent_file_raises(self, tmp_path):
        """Loading a nonexistent file should raise FileNotFoundError."""
        clear_config_cache()
        with pytest.raises(FileNotFoundError):
            load_sources_config(path=tmp_path / "nonexistent.yaml", use_cache=False)

    def test_validation_error_raised(self, tmp_path):
        """Invalid config should raise ConfigValidationError when validate=True."""
        clear_config_cache()
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("version: 1.0\n")
        with pytest.raises(ConfigValidationError):
            load_sources_config(path=bad_yaml, use_cache=False, validate=True)


class TestNormalizeSources:
    """Test _normalize_sources() default injection."""

    def _skeleton_config(self, **overrides):
        """Create a minimal skeleton site config."""
        site = {
            "name": "Test",
            "url": "https://example.com",
            "region": "us",
            "language": "en",
            "group": "E",
            "crawl": {"primary_method": "rss", "fallback_methods": ["dom"]},
            "anti_block": {"ua_tier": 2},
        }
        site.update(overrides)
        return {"sources": {"test": site}}

    def test_fills_missing_crawl_defaults(self):
        """Missing crawl fields are filled with defaults."""
        config = self._skeleton_config()
        result = _normalize_sources(config)
        crawl = result["sources"]["test"]["crawl"]
        assert crawl["rate_limit_seconds"] == _SOURCE_DEFAULTS["crawl"]["rate_limit_seconds"]
        assert crawl["max_requests_per_hour"] == 720
        assert crawl["jitter_seconds"] == 0

    def test_fills_missing_anti_block_defaults(self):
        """Missing anti_block fields are filled with defaults."""
        config = self._skeleton_config()
        result = _normalize_sources(config)
        ab = result["sources"]["test"]["anti_block"]
        assert ab["bot_block_level"] == "LOW"
        assert ab["default_escalation_tier"] == 1
        assert ab["requires_proxy"] is False

    def test_fills_missing_extraction_section(self):
        """Missing extraction section is created with defaults."""
        config = self._skeleton_config()
        result = _normalize_sources(config)
        ext = result["sources"]["test"]["extraction"]
        assert ext["paywall_type"] == "none"
        assert ext["charset"] == "utf-8"
        assert ext["rendering_required"] is False

    def test_fills_missing_meta_section(self):
        """Missing meta section is created with defaults."""
        config = self._skeleton_config()
        result = _normalize_sources(config)
        meta = result["sources"]["test"]["meta"]
        assert meta["enabled"] is True  # Opt-out: default enabled
        assert meta["difficulty_tier"] == "Medium"
        assert meta["daily_article_estimate"] == 50

    def test_preserves_existing_values(self):
        """Existing values must NOT be overwritten by defaults."""
        config = self._skeleton_config()
        config["sources"]["test"]["crawl"]["rate_limit_seconds"] = 10
        config["sources"]["test"]["anti_block"]["bot_block_level"] = "HIGH"
        result = _normalize_sources(config)
        assert result["sources"]["test"]["crawl"]["rate_limit_seconds"] == 10
        assert result["sources"]["test"]["anti_block"]["bot_block_level"] == "HIGH"

    def test_preserves_existing_meta_enabled_false(self):
        """Explicitly disabled sites stay disabled (opt-out respected)."""
        config = self._skeleton_config(meta={"enabled": False})
        result = _normalize_sources(config)
        assert result["sources"]["test"]["meta"]["enabled"] is False

    def test_skips_non_dict_site(self):
        """Non-dict site entry is skipped without error."""
        config = {"sources": {"bad": "not a dict"}}
        result = _normalize_sources(config)
        assert result["sources"]["bad"] == "not a dict"

    def test_skips_non_dict_section(self):
        """Non-dict section within site is skipped without error."""
        config = self._skeleton_config()
        config["sources"]["test"]["crawl"] = "broken"
        result = _normalize_sources(config)
        # crawl is still broken, but no crash
        assert result["sources"]["test"]["crawl"] == "broken"

    def test_empty_sources(self):
        """Empty sources dict doesn't crash."""
        config = {"sources": {}}
        result = _normalize_sources(config)
        assert result["sources"] == {}

    def test_no_sources_key(self):
        """Config without sources key doesn't crash."""
        config = {"version": "1.0"}
        result = _normalize_sources(config)
        assert "version" in result


class TestOptOutEnabledPattern:
    """Test the opt-out enabled pattern across config_loader."""

    def test_get_enabled_sites_defaults_to_true(self, tmp_path):
        """Sites without meta.enabled should be included (opt-out)."""
        clear_config_cache()
        content = """
sources:
  site_a:
    name: "A"
    url: "https://a.com"
    region: "us"
    language: "en"
    group: "E"
    crawl:
      primary_method: "rss"
      fallback_methods: []
      rate_limit_seconds: 5
    anti_block:
      ua_tier: 2
      bot_block_level: "LOW"
    extraction:
      paywall_type: "none"
    meta:
      difficulty_tier: "Easy"
      daily_article_estimate: 10
      sections_count: 1
  site_b:
    name: "B"
    url: "https://b.com"
    region: "us"
    language: "en"
    group: "E"
    crawl:
      primary_method: "rss"
      fallback_methods: []
      rate_limit_seconds: 5
    anti_block:
      ua_tier: 2
      bot_block_level: "LOW"
    extraction:
      paywall_type: "none"
    meta:
      difficulty_tier: "Easy"
      daily_article_estimate: 10
      sections_count: 1
      enabled: false
"""
        yaml_path = tmp_path / "sources.yaml"
        yaml_path.write_text(content)
        config = load_sources_config(path=yaml_path, use_cache=False)
        sites = config.get("sources", {})
        enabled = [
            sid for sid, cfg in sites.items()
            if cfg.get("meta", {}).get("enabled", True)
        ]
        assert "site_a" in enabled  # No enabled key → default True
        assert "site_b" not in enabled  # Explicitly disabled
