"""D-7 Sync Tests — P1 Hallucination Prevention.

Cross-validates all D-7 (intentional duplication) instances to ensure
values stay synchronized across files. Each test catches a specific
desync risk that could cause silent runtime failures.

Tests:
    H-5: pACS regex patterns (sot_manager.py ↔ _context_lib.py)
    H-6: Python version constraints (4 files)
    H-8: GATE_DIRS mapping (validate_retry_budget.py ↔ generate_context_summary.py)
    H-9: Site registry cross-validation (5 hardcoded site lists)
    H-13: ENABLED_DEFAULT sync (6 files — D-7 Instance 13)
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

import pytest

# ---------------------------------------------------------------------------
# Module loading helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
HOOKS_DIR = os.path.join(PROJECT_ROOT, ".claude", "hooks", "scripts")


def _import_from_path(name: str, path: str):
    """Import a Python module from absolute path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_file(path: str) -> str:
    """Read file content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# H-5: pACS Regex Sync (sot_manager.py ↔ _context_lib.py)
# ---------------------------------------------------------------------------

class TestPacsRegexSync:
    """P1: Verify pACS parsing regex patterns are identical in both files.

    D-7 Instance 10: _PACS_WITH_MIN_RE and _PACS_SIMPLE_RE must produce
    identical results in sot_manager.py and _context_lib.py.
    Desync causes pACS scores to be parsed differently, leading to
    SM5c accepting/rejecting different scores than PA7.
    """

    @pytest.fixture
    def sot_regexes(self):
        """Extract regex patterns from sot_manager.py."""
        sot = _import_from_path(
            "sot_manager", os.path.join(SCRIPTS_DIR, "sot_manager.py")
        )
        return sot._PACS_WITH_MIN_RE, sot._PACS_SIMPLE_RE

    @pytest.fixture
    def context_lib_regexes(self):
        """Extract regex patterns from _context_lib.py."""
        ctx = _import_from_path(
            "_context_lib", os.path.join(HOOKS_DIR, "_context_lib.py")
        )
        return ctx._PACS_WITH_MIN_RE, ctx._PACS_SIMPLE_RE

    # --- Pattern string identity ---

    def test_min_regex_pattern_identical(self, sot_regexes, context_lib_regexes):
        """_PACS_WITH_MIN_RE pattern string must be identical in both files."""
        sot_min, _ = sot_regexes
        ctx_min, _ = context_lib_regexes
        assert sot_min.pattern == ctx_min.pattern, (
            f"D-7 DESYNC: _PACS_WITH_MIN_RE patterns differ:\n"
            f"  sot_manager: {sot_min.pattern!r}\n"
            f"  _context_lib: {ctx_min.pattern!r}"
        )

    def test_simple_regex_pattern_identical(self, sot_regexes, context_lib_regexes):
        """_PACS_SIMPLE_RE pattern string must be identical in both files."""
        _, sot_simple = sot_regexes
        _, ctx_simple = context_lib_regexes
        assert sot_simple.pattern == ctx_simple.pattern, (
            f"D-7 DESYNC: _PACS_SIMPLE_RE patterns differ:\n"
            f"  sot_manager: {sot_simple.pattern!r}\n"
            f"  _context_lib: {ctx_simple.pattern!r}"
        )

    def test_min_regex_flags_identical(self, sot_regexes, context_lib_regexes):
        """_PACS_WITH_MIN_RE flags must be identical."""
        sot_min, _ = sot_regexes
        ctx_min, _ = context_lib_regexes
        assert sot_min.flags == ctx_min.flags, (
            f"D-7 DESYNC: _PACS_WITH_MIN_RE flags differ: "
            f"sot={sot_min.flags}, ctx={ctx_min.flags}"
        )

    def test_simple_regex_flags_identical(self, sot_regexes, context_lib_regexes):
        """_PACS_SIMPLE_RE flags must be identical."""
        _, sot_simple = sot_regexes
        _, ctx_simple = context_lib_regexes
        assert sot_simple.flags == ctx_simple.flags, (
            f"D-7 DESYNC: _PACS_SIMPLE_RE flags differ: "
            f"sot={sot_simple.flags}, ctx={ctx_simple.flags}"
        )

    # --- Behavioral equivalence on sample strings ---

    _SAMPLE_PACS_STRINGS = [
        ("pACS = min(75, 80, 70) = 70", 70),
        ("pACS = min(F, C, L) = 85", 85),
        ("Translation pACS = min(Ft, Ct, Nt) = 72", 72),
        ("pACS = 65", 65),
        ("pACS=90", 90),
        ("The pACS = min(60, 55, 70) = 55 is low", 55),
    ]

    @pytest.mark.parametrize("text,expected_score", _SAMPLE_PACS_STRINGS)
    def test_both_regexes_extract_same_score(
        self, sot_regexes, context_lib_regexes, text, expected_score
    ):
        """Both regex sets must extract the same pACS score from sample strings."""
        sot_min, sot_simple = sot_regexes
        ctx_min, ctx_simple = context_lib_regexes

        def _extract(min_re, simple_re, txt):
            m = min_re.search(txt)
            if m:
                return int(m.group(1))
            ms = simple_re.findall(txt)
            if ms:
                return int(ms[-1])
            return None

        sot_result = _extract(sot_min, sot_simple, text)
        ctx_result = _extract(ctx_min, ctx_simple, text)
        assert sot_result == ctx_result == expected_score, (
            f"pACS extraction mismatch on {text!r}: "
            f"sot={sot_result}, ctx={ctx_result}, expected={expected_score}"
        )


# ---------------------------------------------------------------------------
# H-6: Python Version Constraint Sync (4 files)
# ---------------------------------------------------------------------------

class TestPythonVersionSync:
    """P1: Verify Python version constraints are consistent across 4 files.

    D-7 Instance 9: All 4 files must agree on allowed Python versions
    (currently 3.12 and 3.13, excluding 3.14 due to spaCy/pydantic v1).
    Desync causes environment validation to pass in one check but fail
    in another, leading to runtime ImportError.
    """

    def _extract_pyproject_constraint(self) -> tuple[int, int] | None:
        """Extract (min_minor, max_minor_exclusive) from pyproject.toml."""
        path = os.path.join(PROJECT_ROOT, "pyproject.toml")
        if not os.path.isfile(path):
            return None
        content = _read_file(path)
        # requires-python = ">=3.12,<3.14"
        m = re.search(r'requires-python\s*=\s*">=3\.(\d+),<3\.(\d+)"', content)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _extract_main_py_constraint(self) -> int | None:
        """Extract excluded minor version from main.py version check."""
        path = os.path.join(PROJECT_ROOT, "main.py")
        if not os.path.isfile(path):
            return None
        content = _read_file(path)
        # sys.version_info >= (3, 14)  -> excluded_minor = 14
        m = re.search(r"sys\.version_info\s*>=\s*\(3,\s*(\d+)\)", content)
        if m:
            return int(m.group(1))
        return None

    def _extract_setup_init_constraint(self) -> set[int] | None:
        """Extract allowed minor versions from setup_init.py."""
        path = os.path.join(HOOKS_DIR, "setup_init.py")
        if not os.path.isfile(path):
            return None
        content = _read_file(path)
        # minor in (12, 13)
        m = re.search(r"minor\s+in\s+\(([^)]+)\)", content)
        if m:
            return {int(x.strip()) for x in m.group(1).split(",")}
        return None

    def _extract_preflight_constraint(self) -> tuple[int, int] | None:
        """Extract (min_minor, excluded_minor) from preflight_check.py.

        Parses two patterns:
          ok = ver.major == 3 and ver.minor >= 12  → min_minor = 12
          if ver.minor >= 14:  (warning/exclusion)  → excluded_minor = 14
        Returns (min_minor, excluded_minor) or None.
        """
        path = os.path.join(SCRIPTS_DIR, "preflight_check.py")
        if not os.path.isfile(path):
            return None
        content = _read_file(path)
        # Find all "minor >= N" patterns
        matches = re.findall(r"minor\s*>=\s*(\d+)", content)
        if len(matches) >= 2:
            # First = inclusion (min), second = exclusion (max)
            return int(matches[0]), int(matches[1])
        return None

    def test_all_constraints_agree_on_allowed_versions(self):
        """All 4 files must agree on the set of allowed Python minor versions."""
        pyproject = self._extract_pyproject_constraint()
        main_excluded = self._extract_main_py_constraint()
        setup_allowed = self._extract_setup_init_constraint()
        preflight_excluded = self._extract_preflight_constraint()

        # Derive allowed set from each source
        allowed_sets: dict[str, set[int]] = {}

        if pyproject is not None:
            min_minor, max_minor_excl = pyproject
            allowed_sets["pyproject.toml"] = set(range(min_minor, max_minor_excl))

        if main_excluded is not None:
            # main.py excludes >= this version → allowed = {12, 13, ..., excluded-1}
            allowed_sets["main.py"] = set(range(12, main_excluded))

        if setup_allowed is not None:
            allowed_sets["setup_init.py"] = setup_allowed

        if preflight_excluded is not None:
            pf_min, pf_excl = preflight_excluded
            allowed_sets["preflight_check.py"] = set(range(pf_min, pf_excl))

        # Need at least 2 sources to cross-validate
        if len(allowed_sets) < 2:
            pytest.skip(f"Only {len(allowed_sets)} version sources found")

        reference_name = list(allowed_sets.keys())[0]
        reference_set = allowed_sets[reference_name]

        for name, allowed in allowed_sets.items():
            if name == reference_name:
                continue
            assert allowed == reference_set, (
                f"D-7 DESYNC: Python version constraints differ:\n"
                f"  {reference_name}: {sorted(reference_set)}\n"
                f"  {name}: {sorted(allowed)}"
            )

    def test_allowed_versions_include_312_and_313(self):
        """Allowed versions must include 3.12 and 3.13 (current targets)."""
        pyproject = self._extract_pyproject_constraint()
        if pyproject is None:
            pytest.skip("pyproject.toml not found")
        min_minor, max_minor_excl = pyproject
        allowed = set(range(min_minor, max_minor_excl))
        assert 12 in allowed, "Python 3.12 must be allowed"
        assert 13 in allowed, "Python 3.13 must be allowed"

    def test_314_excluded(self):
        """Python 3.14 must be excluded (spaCy/pydantic v1 incompatibility)."""
        pyproject = self._extract_pyproject_constraint()
        if pyproject is None:
            pytest.skip("pyproject.toml not found")
        _, max_minor_excl = pyproject
        assert max_minor_excl <= 14, (
            f"pyproject.toml allows Python 3.14+ — spaCy incompatibility risk"
        )


# ---------------------------------------------------------------------------
# H-8: GATE_DIRS Mapping Sync
# ---------------------------------------------------------------------------

class TestGateDirsSync:
    """P1: Verify GATE_DIRS mapping is identical across files.

    D-7 Instance 7: validate_retry_budget.py GATE_DIRS must match
    generate_context_summary.py's gate_dirs.
    """

    def test_gate_dirs_match(self):
        """GATE_DIRS in validate_retry_budget.py must match generate_context_summary.py."""
        retry_path = os.path.join(HOOKS_DIR, "validate_retry_budget.py")
        summary_path = os.path.join(HOOKS_DIR, "generate_context_summary.py")

        if not os.path.isfile(retry_path) or not os.path.isfile(summary_path):
            pytest.skip("Required files not found")

        retry_content = _read_file(retry_path)
        summary_content = _read_file(summary_path)

        def _extract_gate_dirs(content: str) -> dict[str, str]:
            """Extract gate→dir mappings from source code."""
            mapping = {}
            for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]+)"', content):
                key, val = m.group(1), m.group(2)
                if key in ("verification", "pacs", "review") and "-logs" in val:
                    mapping[key] = val
            return mapping

        retry_dirs = _extract_gate_dirs(retry_content)
        summary_dirs = _extract_gate_dirs(summary_content)

        assert retry_dirs, "Could not extract GATE_DIRS from validate_retry_budget.py"
        assert summary_dirs, "Could not extract gate_dirs from generate_context_summary.py"

        for gate in ("verification", "pacs", "review"):
            assert retry_dirs.get(gate) == summary_dirs.get(gate), (
                f"D-7 DESYNC: GATE_DIRS['{gate}'] differs: "
                f"retry_budget={retry_dirs.get(gate)!r}, "
                f"summary={summary_dirs.get(gate)!r}"
            )


# ---------------------------------------------------------------------------
# H-9: Site Registry Cross-Validation
# ---------------------------------------------------------------------------

class TestSiteRegistrySync:
    """P1: Verify all hardcoded site lists are synchronized.

    Uses validate_site_registry_sync.py to cross-validate 5 independent
    site list sources. This is the most critical D-7 sync check —
    site list desync was the actual bug that triggered this test suite.
    """

    def test_site_registry_sync(self):
        """All 5 site registries must have identical normalized domain sets."""
        validator = _import_from_path(
            "validate_site_registry_sync",
            os.path.join(SCRIPTS_DIR, "validate_site_registry_sync.py"),
        )
        result = validator.validate_sync(PROJECT_ROOT, require_sot=False)

        # RS1 (cross-validate) and RS2 (group counts) must both pass
        assert result["checks"].get("RS1_cross_validate") == "PASS", (
            f"Site registry desync:\n" +
            "\n".join(result.get("errors", []))
        )
        assert result["checks"].get("RS2_group_counts") == "PASS", (
            f"Group count mismatch:\n" +
            "\n".join(result.get("errors", []))
        )

    def test_total_121_sites(self):
        """All lists must have exactly 121 sites."""
        validator = _import_from_path(
            "validate_site_registry_sync",
            os.path.join(SCRIPTS_DIR, "validate_site_registry_sync.py"),
        )
        result = validator.validate_sync(PROJECT_ROOT, require_sot=False)
        assert result["checks"].get("RS4_total_count") == "PASS", (
            f"Total count mismatch:\n" +
            "\n".join(result.get("errors", []))
        )


# ---------------------------------------------------------------------------
# H-13: ENABLED_DEFAULT Sync (D-7 Instance 13)
# ---------------------------------------------------------------------------

class TestEnabledDefaultSync:
    """P1: Verify meta.enabled default value is identical across 6 files.

    D-7 Instance 13: The opt-out pattern (sites enabled by default) uses
    ENABLED_DEFAULT in 6 locations. 4 import from constants.py (SOT),
    1 hardcodes (preflight_check.py — standalone, can't import src).
    Desync causes site filtering to silently include/exclude different sites.
    """

    @pytest.fixture
    def validator(self):
        return _import_from_path(
            "validate_enabled_default_sync",
            os.path.join(SCRIPTS_DIR, "validate_enabled_default_sync.py"),
        )

    def test_all_checks_pass(self, validator):
        """ED1-ED6 + ED-CROSS must all pass."""
        result = validator.run_validation(PROJECT_ROOT)
        assert result["valid"], (
            f"ENABLED_DEFAULT sync failed:\n"
            + json.dumps(result, indent=2)
        )

    def test_sot_value_is_true(self, validator):
        """SOT (constants.py) ENABLED_DEFAULT must be True (opt-out pattern)."""
        result = validator.run_validation(PROJECT_ROOT)
        assert result["sot_value"] is True, (
            f"ENABLED_DEFAULT SOT value is {result['sot_value']!r}, expected True"
        )

    def test_importers_reference_sot(self, validator):
        """ED2-ED5, ED7 must reference ENABLED_DEFAULT by name (import from SOT)."""
        result = validator.run_validation(PROJECT_ROOT)
        for check_id in ("ED2", "ED3", "ED4", "ED5", "ED7"):
            check = result["checks"][check_id]
            assert check.get("references_sot"), (
                f"{check_id} does not reference ENABLED_DEFAULT from SOT — "
                f"uses hardcoded value instead"
            )

    def test_preflight_hardcoded_matches(self, validator):
        """ED6 (preflight_check.py) hardcoded value must match SOT."""
        result = validator.run_validation(PROJECT_ROOT)
        ed6 = result["checks"]["ED6"]
        assert ed6["valid"], f"ED6 extraction failed: {ed6.get('error')}"
        assert ed6["value"] == result["sot_value"], (
            f"preflight_check.py hardcodes {ed6['value']!r} "
            f"but SOT is {result['sot_value']!r}"
        )

    def test_pipeline_has_at_least_3_locations(self, validator):
        """ED4 must find at least 3 .get('enabled', ...) in pipeline.py."""
        result = validator.run_validation(PROJECT_ROOT)
        ed4 = result["checks"]["ED4"]
        assert ed4["valid"], f"ED4 failed: {ed4.get('error')}"
        assert ed4["count"] >= 3, (
            f"pipeline.py has only {ed4['count']} .get('enabled', ...) calls, "
            f"expected at least 3"
        )

    def test_crawler_references_sot(self, validator):
        """ED7 (crawler.py) must reference ENABLED_DEFAULT from SOT."""
        result = validator.run_validation(PROJECT_ROOT)
        ed7 = result["checks"]["ED7"]
        assert ed7["valid"], f"ED7 extraction failed: {ed7.get('error')}"
        assert ed7.get("references_sot"), (
            f"crawler.py uses hardcoded value {ed7['value']!r} "
            f"instead of ENABLED_DEFAULT"
        )

    def test_no_cross_validation_mismatches(self, validator):
        """ED-CROSS must have zero mismatches."""
        result = validator.run_validation(PROJECT_ROOT)
        cross = result["cross_validation"]
        assert cross["valid"], (
            f"ENABLED_DEFAULT cross-validation mismatches:\n"
            + json.dumps(cross["mismatches"], indent=2)
        )
