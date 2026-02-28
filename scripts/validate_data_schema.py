#!/usr/bin/env python3
"""Data Schema Validator — P1 deterministic schema completeness check.

Verifies that generated Parquet files match the authoritative PyArrow schemas
defined in ``src/storage/parquet_writer.py`` and individual stage modules.

Two validation modes:
    --check-definitions   Compare column counts in constants.py against
                          authoritative pa.schema objects (design-time check).
    --check-files         Read actual Parquet files on disk and compare their
                          schemas against authoritative definitions (runtime check).

Config validation (--check-config) validates sources.yaml and pipeline.yaml
structure independently.

Usage:
    python3 scripts/validate_data_schema.py --check-definitions --project-dir .
    python3 scripts/validate_data_schema.py --check-files --project-dir .
    python3 scripts/validate_data_schema.py --check-config --project-dir .
    python3 scripts/validate_data_schema.py --check-definitions --check-files --project-dir .

JSON output to stdout.  Exit code 0 always.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Authoritative Schema Definitions (imported from implementation)
# ---------------------------------------------------------------------------
# These are the ACTUAL schemas used by parquet_writer.py and stage modules.
# They are the single source of truth for column structure.

# Column-name lists derived from the authoritative pa.schema objects.
# We list them here as plain dicts so that this script can run even when
# pyarrow is not installed (CI lint environments).

ARTICLES_EXPECTED = [
    "article_id", "url", "title", "body", "source", "category",
    "language", "published_at", "crawled_at", "author", "word_count",
    "content_hash",
]  # 12 columns — src/storage/parquet_writer.py:ARTICLES_PA_SCHEMA

ANALYSIS_EXPECTED = [
    "article_id", "sentiment_label", "sentiment_score",
    "emotion_joy", "emotion_trust", "emotion_fear", "emotion_surprise",
    "emotion_sadness", "emotion_disgust", "emotion_anger",
    "emotion_anticipation", "topic_id", "topic_label", "topic_probability",
    "steeps_category", "importance_score", "keywords",
    "entities_person", "entities_org", "entities_location", "embedding",
]  # 21 columns — src/storage/parquet_writer.py:ANALYSIS_PA_SCHEMA

SIGNALS_EXPECTED = [
    "signal_id", "signal_layer", "signal_label", "detected_at",
    "topic_ids", "article_ids", "burst_score", "changepoint_significance",
    "novelty_score", "singularity_composite", "evidence_summary",
    "confidence",
]  # 12 columns — src/storage/parquet_writer.py:SIGNALS_PA_SCHEMA

TOPICS_EXPECTED = [
    "article_id", "topic_id", "topic_label", "topic_probability",
    "hdbscan_cluster_id", "nmf_topic_id", "lda_topic_id",
]  # 7 columns — src/storage/parquet_writer.py:TOPICS_PA_SCHEMA

TIMESERIES_EXPECTED = [
    "series_id", "topic_id", "metric_type", "date", "value",
    "trend", "seasonal", "residual", "burst_score", "is_changepoint",
    "changepoint_significance", "prophet_forecast", "prophet_lower",
    "prophet_upper", "ma_short", "ma_long", "ma_signal",
]  # 17 columns — src/analysis/stage5_timeseries.py:TIMESERIES_SCHEMA

CROSS_ANALYSIS_EXPECTED = [
    "analysis_type", "source_entity", "target_entity", "relationship",
    "strength", "p_value", "lag_days", "evidence_articles", "metadata",
]  # 9 columns — src/analysis/stage6_cross_analysis.py:_cross_analysis_schema()

NETWORKS_EXPECTED = [
    "entity_a", "entity_b", "co_occurrence_count", "community_id",
    "source_articles",
]  # 5 columns — src/analysis/stage4_aggregation.py (inline schema)

ARTICLE_ANALYSIS_EXPECTED = [
    "article_id", "sentiment_label", "sentiment_score",
    "emotion_joy", "emotion_trust", "emotion_fear", "emotion_surprise",
    "emotion_sadness", "emotion_anger", "emotion_disgust",
    "emotion_anticipation", "steeps_category", "importance_score",
]  # 13 columns — src/analysis/stage3_article_analysis.py (inline schema)

# Registry: parquet filename stem -> (expected columns, relative path)
SCHEMA_REGISTRY = {
    "articles":          (ARTICLES_EXPECTED,          "data/processed/articles.parquet"),
    "article_analysis":  (ARTICLE_ANALYSIS_EXPECTED,  "data/analysis/article_analysis.parquet"),
    "topics":            (TOPICS_EXPECTED,             "data/analysis/topics.parquet"),
    "networks":          (NETWORKS_EXPECTED,           "data/analysis/networks.parquet"),
    "timeseries":        (TIMESERIES_EXPECTED,         "data/analysis/timeseries.parquet"),
    "cross_analysis":    (CROSS_ANALYSIS_EXPECTED,     "data/analysis/cross_analysis.parquet"),
    "signals":           (SIGNALS_EXPECTED,            "data/output/signals.parquet"),
    "analysis":          (ANALYSIS_EXPECTED,           "data/output/analysis.parquet"),
}

# constants.py column count declarations (for --check-definitions)
CONSTANTS_COLUMN_COUNTS = {
    "ARTICLES_SCHEMA_COLUMNS": ("articles", 12),
    "ANALYSIS_SCHEMA_COLUMNS": ("analysis", 21),
    "SIGNALS_SCHEMA_COLUMNS":  ("signals",  12),
}


# ---------------------------------------------------------------------------
# DS1: Definition consistency check
# ---------------------------------------------------------------------------

def _check_definitions(project_dir):
    """Verify constants.py column counts match authoritative schemas."""
    warnings = []

    for const_name, (table, expected_count) in CONSTANTS_COLUMN_COUNTS.items():
        expected_cols, _ = SCHEMA_REGISTRY[table]
        actual_count = len(expected_cols)
        if actual_count != expected_count:
            warnings.append(
                f"DS1: {const_name}={expected_count} in constants.py but "
                f"authoritative schema for '{table}' has {actual_count} columns"
            )

    # Also try importing the actual constants to cross-check
    try:
        sys.path.insert(0, os.path.join(project_dir))
        from src.config.constants import (
            ARTICLES_SCHEMA_COLUMNS,
            ANALYSIS_SCHEMA_COLUMNS,
            SIGNALS_SCHEMA_COLUMNS,
        )
        runtime_counts = {
            "articles": ARTICLES_SCHEMA_COLUMNS,
            "analysis": ANALYSIS_SCHEMA_COLUMNS,
            "signals": SIGNALS_SCHEMA_COLUMNS,
        }
        for table, runtime_count in runtime_counts.items():
            expected_cols, _ = SCHEMA_REGISTRY[table]
            if runtime_count != len(expected_cols):
                warnings.append(
                    f"DS1b: Runtime {table} column count={runtime_count} but "
                    f"authoritative schema has {len(expected_cols)} columns"
                )
    except ImportError:
        warnings.append("DS1c: Could not import src.config.constants (pyarrow missing?)")

    return warnings


# ---------------------------------------------------------------------------
# DS2-DS3: Parquet file schema check
# ---------------------------------------------------------------------------

def _check_files(project_dir):
    """Read actual Parquet files and compare schemas against authoritative definitions."""
    warnings = []

    try:
        import pyarrow.parquet as pq
    except ImportError:
        warnings.append("DS2: pyarrow not installed — cannot check Parquet files")
        return warnings

    for table_name, (expected_cols, rel_path) in SCHEMA_REGISTRY.items():
        full_path = os.path.join(project_dir, rel_path)
        if not os.path.exists(full_path):
            # Not an error — file may not exist yet (pre-run)
            continue

        try:
            schema = pq.read_schema(full_path)
            actual_cols = schema.names

            # DS2: Check for missing columns
            missing = [c for c in expected_cols if c not in actual_cols]
            if missing:
                warnings.append(
                    f"DS2: {table_name} ({rel_path}) missing expected columns: {missing}"
                )

            # DS3: Check for unexpected extra columns
            extra = [c for c in actual_cols if c not in expected_cols]
            if extra:
                warnings.append(
                    f"DS3: {table_name} ({rel_path}) has unexpected columns: {extra}"
                )

        except Exception as e:
            warnings.append(f"DS2b: {table_name} ({rel_path}) read error: {e}")

    return warnings


# ---------------------------------------------------------------------------
# DS4: Config file structure check
# ---------------------------------------------------------------------------

SITE_CONFIG_REQUIRED_FIELDS = [
    "name", "url", "language", "group",
]


def _check_config_files(project_dir):
    """Validate sources.yaml and pipeline.yaml structure."""
    warnings = []

    # sources.yaml
    sources_path = os.path.join(project_dir, "data", "config", "sources.yaml")
    if not os.path.exists(sources_path):
        warnings.append("DS_CFG1: data/config/sources.yaml not found")
    else:
        try:
            import yaml
            with open(sources_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                # Top-level key is "sources" (dict of site_id -> config)
                sites = data.get("sources", data.get("sites", {}))
                if not sites:
                    warnings.append("DS_CFG2: sources.yaml has no 'sites' section")
                elif isinstance(sites, dict):
                    # sources.yaml uses dict format: {site_id: {config...}}
                    for site_id, site in list(sites.items())[:5]:
                        if isinstance(site, dict):
                            for fld in SITE_CONFIG_REQUIRED_FIELDS:
                                if fld not in site:
                                    warnings.append(
                                        f"DS_CFG3: sources.{site_id} missing field: {fld}"
                                    )
                    if len(sites) < 44:
                        warnings.append(
                            f"DS_CFG4: sources.yaml has {len(sites)} sites (expected 44)"
                        )
                elif isinstance(sites, list):
                    for i, site in enumerate(sites[:5]):
                        if isinstance(site, dict):
                            for fld in SITE_CONFIG_REQUIRED_FIELDS:
                                if fld not in site:
                                    warnings.append(
                                        f"DS_CFG3: sites[{i}] missing field: {fld}"
                                    )
                    if len(sites) < 44:
                        warnings.append(
                            f"DS_CFG4: sources.yaml has {len(sites)} sites (expected 44)"
                        )
        except ImportError:
            warnings.append("DS_CFG1b: PyYAML not installed")
        except Exception as e:
            warnings.append(f"DS_CFG1c: sources.yaml parse error: {e}")

    # pipeline.yaml
    pipeline_path = os.path.join(project_dir, "data", "config", "pipeline.yaml")
    if not os.path.exists(pipeline_path):
        warnings.append("DS_CFG5: data/config/pipeline.yaml not found")
    else:
        try:
            import yaml
            with open(pipeline_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                pipeline = data.get("pipeline", data)
                stages = pipeline.get("stages", {})
                if not stages:
                    warnings.append("DS_CFG6: pipeline.yaml has no 'stages' section")
                elif isinstance(stages, dict) and len(stages) < 8:
                    warnings.append(
                        f"DS_CFG7: pipeline.yaml has {len(stages)} stages (expected 8)"
                    )
        except ImportError:
            warnings.append("DS_CFG5b: PyYAML not installed")
        except Exception as e:
            warnings.append(f"DS_CFG5c: pipeline.yaml parse error: {e}")

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate_schema(project_dir, check_definitions=False, check_files=False,
                    check_config=False):
    """Run selected validation checks."""
    result = {
        "valid": True,
        "tables_registered": len(SCHEMA_REGISTRY),
        "checks_run": [],
        "warnings": [],
    }

    if check_definitions:
        result["checks_run"].append("definitions")
        result["warnings"].extend(_check_definitions(project_dir))

    if check_files:
        result["checks_run"].append("files")
        result["warnings"].extend(_check_files(project_dir))

    if check_config:
        result["checks_run"].append("config")
        result["warnings"].extend(_check_config_files(project_dir))

    if not result["checks_run"]:
        result["warnings"].append("No checks requested (use --check-definitions, "
                                  "--check-files, or --check-config)")

    if result["warnings"]:
        result["valid"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description="Data Schema Validator — P1")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--check-definitions", action="store_true",
                        help="Verify constants.py column counts match authoritative schemas")
    parser.add_argument("--check-files", action="store_true",
                        help="Read Parquet files and compare against authoritative schemas")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate sources.yaml and pipeline.yaml structure")
    # Legacy compatibility: --step and --check-config-only are silently accepted
    parser.add_argument("--step", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Default: run definitions check if nothing specified
    defs = args.check_definitions
    files = args.check_files
    cfg = args.check_config
    if not (defs or files or cfg):
        defs = True

    result = validate_schema(args.project_dir, defs, files, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
