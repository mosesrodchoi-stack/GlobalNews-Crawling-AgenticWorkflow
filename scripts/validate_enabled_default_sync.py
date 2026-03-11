#!/usr/bin/env python3
"""ENABLED_DEFAULT Sync Validator — P1 hallucination prevention.

Cross-validates the meta.enabled default value across all 7 locations in the
codebase to ensure the opt-out pattern (D-7 Instance 13) stays consistent.

Checks:
    ED1: constants.py ENABLED_DEFAULT assignment value (AST extract)
    ED2: config_loader.py _SOURCE_DEFAULTS["meta"]["enabled"] value (AST extract)
    ED3: config_loader.py get_enabled_sites() .get("enabled", <default>) (AST extract)
    ED4: pipeline.py ALL .get("enabled", <default>) calls (AST extract, min 3)
    ED5: main.py status .get("enabled", <default>) (AST extract)
    ED6: preflight_check.py .get("enabled", <default>) (AST extract)
    ED7: crawler.py .get("enabled", <default>) (AST extract)
    ED-CROSS: All 7 extracted values are identical

Usage:
    python3 scripts/validate_enabled_default_sync.py --project-dir .

JSON output to stdout. Exit code 0 if valid, 1 if invalid.

This is a P1 deterministic script: no LLM inference, pure AST extraction.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# AST extraction helpers
# ---------------------------------------------------------------------------

def _extract_assignment_value(tree: ast.Module, var_name: str) -> Any | None:
    """Extract the literal value of a top-level assignment like `VAR = True`.

    Returns the Python literal value, or None if not found.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        return None
    return None


def _extract_dict_nested_value(
    tree: ast.Module, dict_name: str, keys: list[str]
) -> Any | None:
    """Extract a nested literal value from a dict assignment.

    E.g., _SOURCE_DEFAULTS["meta"]["enabled"] → traverse dict literal.
    Handles both plain assignments and annotated assignments (with type hints).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_name:
                    return _resolve_dict_keys(node.value, keys)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == dict_name and node.value:
                return _resolve_dict_keys(node.value, keys)
    return None


def _resolve_dict_keys(node: ast.expr, keys: list[str]) -> Any | None:
    """Recursively resolve dict literal keys."""
    if not keys:
        # Leaf node — try to evaluate
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            # Could be a Name reference (e.g., ENABLED_DEFAULT)
            if isinstance(node, ast.Name):
                return f"__NAME__{node.id}"
            return None

    if not isinstance(node, ast.Dict):
        return None

    target_key = keys[0]
    remaining = keys[1:]

    for k, v in zip(node.keys, node.values):
        if k is None:
            continue
        try:
            key_val = ast.literal_eval(k)
        except (ValueError, TypeError):
            continue
        if str(key_val) == target_key:
            return _resolve_dict_keys(v, remaining)

    return None


def _extract_get_default_in_function(
    tree: ast.Module,
    func_name: str,
    get_key: str,
) -> list[Any]:
    """Extract all .get(key, <default>) second-argument values inside a function.

    Returns list of extracted values (could be multiple .get() calls).
    """
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                _collect_get_defaults(node, get_key, results)
    return results


def _extract_get_default_in_module(
    tree: ast.Module,
    get_key: str,
) -> list[Any]:
    """Extract all .get(key, <default>) second-argument values at module level.

    Searches entire module (all functions).
    """
    results = []
    _collect_get_defaults(tree, get_key, results)
    return results


def _collect_get_defaults(
    node: ast.AST, get_key: str, results: list[Any]
) -> None:
    """Walk AST collecting .get("key", default) second arguments."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        # Match pattern: expr.get("key", default)
        if not isinstance(child.func, ast.Attribute):
            continue
        if child.func.attr != "get":
            continue
        if len(child.args) < 2:
            continue
        # First arg must be the target key
        try:
            first_arg = ast.literal_eval(child.args[0])
        except (ValueError, TypeError):
            continue
        if str(first_arg) != get_key:
            continue
        # Second arg is the default value
        second_arg = child.args[1]
        try:
            val = ast.literal_eval(second_arg)
            results.append(val)
        except (ValueError, TypeError):
            # Could be a Name reference (e.g., ENABLED_DEFAULT)
            if isinstance(second_arg, ast.Name):
                results.append(f"__NAME__{second_arg.id}")
            else:
                results.append(None)


def _resolve_name_refs(
    values: dict[str, Any], sot_value: Any
) -> dict[str, Any]:
    """Resolve __NAME__ENABLED_DEFAULT references to the SOT value.

    When a location imports ENABLED_DEFAULT and uses it as .get("enabled", ENABLED_DEFAULT),
    the AST extracts "__NAME__ENABLED_DEFAULT". This is correct — it means the code
    references the SOT constant. We resolve it to the actual SOT value for comparison.
    """
    resolved = {}
    for key, val in values.items():
        if isinstance(val, str) and val == "__NAME__ENABLED_DEFAULT":
            resolved[key] = sot_value  # References SOT — correct by construction
        elif isinstance(val, list):
            # For functions with multiple .get() calls, resolve each
            resolved[key] = [
                sot_value if isinstance(v, str) and v == "__NAME__ENABLED_DEFAULT" else v
                for v in val
            ]
        else:
            resolved[key] = val
    return resolved


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_ed1(project_dir: Path) -> dict[str, Any]:
    """ED1: constants.py ENABLED_DEFAULT assignment value."""
    path = project_dir / "src" / "config" / "constants.py"
    if not path.exists():
        return {"check": "ED1", "valid": False, "error": f"{path} not found"}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value = _extract_assignment_value(tree, "ENABLED_DEFAULT")

    if value is None:
        return {
            "check": "ED1",
            "valid": False,
            "error": "ENABLED_DEFAULT assignment not found in constants.py",
        }

    return {"check": "ED1", "valid": True, "value": value, "source": "constants.py"}


def check_ed2(project_dir: Path) -> dict[str, Any]:
    """ED2: config_loader.py _SOURCE_DEFAULTS['meta']['enabled'] value."""
    path = project_dir / "src" / "utils" / "config_loader.py"
    if not path.exists():
        return {"check": "ED2", "valid": False, "error": f"{path} not found"}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    value = _extract_dict_nested_value(tree, "_SOURCE_DEFAULTS", ["meta", "enabled"])

    if value is None:
        return {
            "check": "ED2",
            "valid": False,
            "error": "_SOURCE_DEFAULTS['meta']['enabled'] not found in config_loader.py",
        }

    return {
        "check": "ED2",
        "valid": True,
        "value": value,
        "source": "config_loader.py _SOURCE_DEFAULTS",
    }


def check_ed3(project_dir: Path) -> dict[str, Any]:
    """ED3: config_loader.py get_enabled_sites() .get('enabled', default)."""
    path = project_dir / "src" / "utils" / "config_loader.py"
    if not path.exists():
        return {"check": "ED3", "valid": False, "error": f"{path} not found"}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = _extract_get_default_in_function(tree, "get_enabled_sites", "enabled")

    if not values:
        return {
            "check": "ED3",
            "valid": False,
            "error": ".get('enabled', ...) not found in get_enabled_sites()",
        }

    return {
        "check": "ED3",
        "valid": True,
        "value": values[0],
        "all_values": values,
        "source": "config_loader.py get_enabled_sites()",
    }


def check_ed4(project_dir: Path) -> dict[str, Any]:
    """ED4: pipeline.py ALL .get('enabled', default) calls.

    Searches the entire module (not just one function) to catch all
    occurrences across _resolve_target_sites() and _run_single_pass().
    Requires at least 3 occurrences (2 in _resolve_target_sites + 1 in crawl loop).
    """
    path = project_dir / "src" / "crawling" / "pipeline.py"
    if not path.exists():
        return {"check": "ED4", "valid": False, "error": f"{path} not found"}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = _extract_get_default_in_module(tree, "enabled")

    if not values:
        return {
            "check": "ED4",
            "valid": False,
            "error": ".get('enabled', ...) not found in pipeline.py",
        }

    min_expected = 3
    if len(values) < min_expected:
        return {
            "check": "ED4",
            "valid": False,
            "error": (
                f"pipeline.py has {len(values)} .get('enabled', ...) calls, "
                f"expected at least {min_expected}"
            ),
            "all_values": values,
        }

    return {
        "check": "ED4",
        "valid": True,
        "value": values[0],
        "all_values": values,
        "count": len(values),
        "source": "pipeline.py (all functions)",
    }


def check_ed5(project_dir: Path) -> dict[str, Any]:
    """ED5: main.py status reporting .get('enabled', default)."""
    path = project_dir / "main.py"
    if not path.exists():
        return {"check": "ED5", "valid": False, "error": f"{path} not found"}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = _extract_get_default_in_module(tree, "enabled")

    if not values:
        return {
            "check": "ED5",
            "valid": False,
            "error": ".get('enabled', ...) not found in main.py",
        }

    return {
        "check": "ED5",
        "valid": True,
        "value": values[0],
        "all_values": values,
        "source": "main.py",
    }


def check_ed6(project_dir: Path) -> dict[str, Any]:
    """ED6: preflight_check.py .get('enabled', default).

    This is a standalone script that cannot import from src/.
    It hardcodes the value directly — validated here for drift.
    """
    path = project_dir / "scripts" / "preflight_check.py"
    if not path.exists():
        return {"check": "ED6", "valid": False, "error": f"{path} not found"}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = _extract_get_default_in_module(tree, "enabled")

    if not values:
        return {
            "check": "ED6",
            "valid": False,
            "error": ".get('enabled', ...) not found in preflight_check.py",
        }

    return {
        "check": "ED6",
        "valid": True,
        "value": values[0],
        "all_values": values,
        "source": "preflight_check.py (hardcoded — standalone)",
    }


def check_ed7(project_dir: Path) -> dict[str, Any]:
    """ED7: crawler.py .get('enabled', default)."""
    path = project_dir / "src" / "crawling" / "crawler.py"
    if not path.exists():
        return {"check": "ED7", "valid": False, "error": f"{path} not found"}

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = _extract_get_default_in_module(tree, "enabled")

    if not values:
        return {
            "check": "ED7",
            "valid": False,
            "error": ".get('enabled', ...) not found in crawler.py",
        }

    return {
        "check": "ED7",
        "valid": True,
        "value": values[0],
        "all_values": values,
        "source": "crawler.py",
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def check_ed_cross(
    results: dict[str, dict[str, Any]], sot_value: Any
) -> dict[str, Any]:
    """ED-CROSS: All 7 extracted values are identical to SOT."""
    mismatches = []

    for check_id, result in results.items():
        if not result.get("valid"):
            mismatches.append({
                "check": check_id,
                "error": result.get("error", "extraction failed"),
            })
            continue

        # Get the resolved value for comparison
        val = result.get("resolved_value", result.get("value"))

        # Handle list values (functions with multiple .get() calls)
        if isinstance(val, list):
            for i, v in enumerate(val):
                if v != sot_value:
                    mismatches.append({
                        "check": f"{check_id}[{i}]",
                        "expected": sot_value,
                        "actual": v,
                    })
        elif val != sot_value:
            mismatches.append({
                "check": check_id,
                "expected": sot_value,
                "actual": val,
            })

    return {
        "check": "ED-CROSS",
        "valid": len(mismatches) == 0,
        "sot_value": sot_value,
        "mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(project_dir: Path | str) -> dict[str, Any]:
    """Run all ED checks and return structured results."""
    project_dir = Path(project_dir)
    # ED1 is SOT — must succeed first
    ed1 = check_ed1(project_dir)
    if not ed1["valid"]:
        return {
            "valid": False,
            "error": "SOT extraction failed (ED1)",
            "checks": {"ED1": ed1},
        }

    sot_value = ed1["value"]

    # Run remaining checks
    checks = {
        "ED1": ed1,
        "ED2": check_ed2(project_dir),
        "ED3": check_ed3(project_dir),
        "ED4": check_ed4(project_dir),
        "ED5": check_ed5(project_dir),
        "ED6": check_ed6(project_dir),
        "ED7": check_ed7(project_dir),
    }

    # Resolve Name references to SOT value
    for check_id, result in checks.items():
        if not result.get("valid"):
            continue
        val = result.get("value")
        if isinstance(val, str) and val == "__NAME__ENABLED_DEFAULT":
            result["resolved_value"] = sot_value
            result["references_sot"] = True
        elif isinstance(val, list):
            resolved = []
            refs_sot = False
            for v in val:
                if isinstance(v, str) and v == "__NAME__ENABLED_DEFAULT":
                    resolved.append(sot_value)
                    refs_sot = True
                else:
                    resolved.append(v)
            result["resolved_value"] = resolved
            if refs_sot:
                result["references_sot"] = True

    # Cross-validation
    cross = check_ed_cross(checks, sot_value)

    all_valid = all(r["valid"] for r in checks.values()) and cross["valid"]

    return {
        "valid": all_valid,
        "sot_value": sot_value,
        "checks": checks,
        "cross_validation": cross,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ENABLED_DEFAULT sync across D-7 Instance 13 locations"
    )
    parser.add_argument(
        "--project-dir", required=True, help="Project root directory"
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    result = run_validation(project_dir)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
