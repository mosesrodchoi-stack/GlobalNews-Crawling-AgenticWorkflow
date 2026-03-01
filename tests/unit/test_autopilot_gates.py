"""Unit tests for autopilot structural reinforcement — ST7, HQ1/HQ2/HQ3, S6, DL1-DL6.

Tests cover:
- ST7: Decision log check in validate_step_transition.py
- HQ1/HQ2/HQ3: Human quality gates in run_quality_gates.py
- S6/S6b: HUMAN_STEPS_SET validation in _context_lib.py validate_sot_schema()
- D-7: HUMAN_STEPS constant consistency across 4 files
- DL1-DL6: Decision log P1 validation in _context_lib.py validate_decision_log()
"""

import importlib.util
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Module import helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
HOOKS_DIR = os.path.join(PROJECT_ROOT, ".claude", "hooks", "scripts")

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _import_script(name, directory=SCRIPTS_DIR):
    """Import a script module by name from a given directory."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(directory, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sot_mod():
    return _import_script("sot_manager")


@pytest.fixture
def transition_mod():
    return _import_script("validate_step_transition")


@pytest.fixture
def gates_mod():
    return _import_script("run_quality_gates")


@pytest.fixture
def context_lib():
    return _import_script("_context_lib", HOOKS_DIR)


@pytest.fixture
def tmp_project(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    return tmp_path


@pytest.fixture
def tmp_project_with_sot(tmp_project, sot_mod):
    result = sot_mod.cmd_init(str(tmp_project), "Test Workflow", 20)
    assert result["valid"]
    return tmp_project


def _enable_autopilot_at_step(sot_mod, pd, step_num):
    """Enable autopilot and set current_step to step_num."""
    import yaml
    sot_mod.cmd_set_autopilot(pd, "true")
    sot_path = sot_mod._sot_path(pd)
    with open(sot_path, "r") as f:
        data = yaml.safe_load(f)
    data["workflow"]["current_step"] = step_num
    with open(sot_path, "w") as f:
        yaml.dump(data, f)


def _create_decision_log(pd, step_num, size=200, template=True):
    """Create a decision log file.

    If template=True, creates a properly formatted decision log matching
    the autopilot-decision-template.md structure. If False, creates raw text.
    """
    log_dir = os.path.join(pd, "autopilot-logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"step-{step_num}-decision.md")
    if template:
        content = (
            f"# Decision Log — Step {step_num}\n\n"
            f"- **Step**: {step_num}\n"
            f"- **Checkpoint Type**: (human) — Test checkpoint\n"
            f"- **Decision**: Proceed with quality-maximizing defaults\n"
            f"- **Rationale**: Absolute criterion 1 — quality maximization requires full coverage\n"
            f"- **Timestamp**: 2026-02-28 12:00:00\n"
        )
        # Pad to minimum size if needed
        if len(content) < size:
            content += "\n" + "x" * (size - len(content))
        with open(log_path, "w") as f:
            f.write(content)
    else:
        with open(log_path, "w") as f:
            f.write("# Decision Log\n" + "x" * size)


def _create_output(pd, path, size=200):
    """Create a dummy output file."""
    full = os.path.join(pd, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write("# Output\n" + "x" * size)


# ============================================================================
# D-7: HUMAN_STEPS constant consistency
# ============================================================================

class TestHumanStepsConsistency:
    """Verify HUMAN_STEPS is identical across all 4 files (D-7)."""

    def test_d7_human_steps_all_match(self, sot_mod, transition_mod, gates_mod, context_lib):
        expected = frozenset({4, 8, 18})
        assert sot_mod.HUMAN_STEPS == expected, f"sot_manager: {sot_mod.HUMAN_STEPS}"
        assert transition_mod.HUMAN_STEPS == expected, f"validate_step_transition: {transition_mod.HUMAN_STEPS}"
        assert gates_mod.HUMAN_STEPS == expected, f"run_quality_gates: {gates_mod.HUMAN_STEPS}"
        assert context_lib.HUMAN_STEPS_SET == expected, f"_context_lib: {context_lib.HUMAN_STEPS_SET}"

    def test_d7_all_frozenset(self, sot_mod, transition_mod, gates_mod, context_lib):
        assert isinstance(sot_mod.HUMAN_STEPS, frozenset)
        assert isinstance(transition_mod.HUMAN_STEPS, frozenset)
        assert isinstance(gates_mod.HUMAN_STEPS, frozenset)
        assert isinstance(context_lib.HUMAN_STEPS_SET, frozenset)


# ============================================================================
# ST7: Decision log check in validate_step_transition.py
# ============================================================================

class TestST7DecisionLog:
    """Tests for ST7 blocking check in validate_transition."""

    def test_st7_pass_when_log_exists(self, transition_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        _create_decision_log(pd, 8)
        result = transition_mod.validate_transition(pd, 8)
        assert result["checks"]["ST7"] == "PASS"

    def test_st7_fail_when_log_missing(self, transition_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        # No decision log created
        result = transition_mod.validate_transition(pd, 8)
        assert result["checks"]["ST7"] == "FAIL"
        assert any("ST7" in b for b in result["blocking"])

    def test_st7_na_when_manual_mode(self, transition_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        # Autopilot is off (default)
        import yaml
        sot_path = sot_mod._sot_path(pd)
        with open(sot_path, "r") as f:
            data = yaml.safe_load(f)
        data["workflow"]["current_step"] = 8
        with open(sot_path, "w") as f:
            yaml.dump(data, f)
        result = transition_mod.validate_transition(pd, 8)
        assert result["checks"]["ST7"] == "N/A"

    def test_st7_na_for_non_human_step(self, transition_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 5)
        result = transition_mod.validate_transition(pd, 5)
        assert result["checks"]["ST7"] == "N/A"


# ============================================================================
# HQ1/HQ2/HQ3: Human quality gates in run_quality_gates.py
# ============================================================================

class TestHumanQualityGates:
    """Tests for HQ1/HQ2/HQ3 in check_quality_gates."""

    def test_auto_detect_runs_hq_when_autopilot_on(self, gates_mod, sot_mod, tmp_project_with_sot):
        """Auto-detection: autopilot ON → HQ gates run (no --check-autopilot needed)."""
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        # No decision log → HQ1 should FAIL (proving HQ gates DID run)
        result = gates_mod.check_quality_gates(pd, 8)
        assert "HQ1_decision_log" in result["gates"]
        assert result["gates"]["HQ1_decision_log"] == "FAIL"

    def test_hq_all_pass(self, gates_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        # Create decision log (template format)
        _create_decision_log(pd, 8)
        # Add to auto_approved
        sot_mod.cmd_add_auto_approved(pd, 8)
        # Create previous step output and record it
        _create_output(pd, "planning/step-7.md")
        sot_mod.cmd_record_output(pd, 7, "planning/step-7.md")
        result = gates_mod.check_quality_gates(pd, 8)
        assert result["valid"] is True
        assert result["gates"]["HQ1_decision_log"] == "PASS"
        assert result["gates"]["HQ2_auto_approved"] == "PASS"
        assert result["gates"]["HQ3_prev_output"] == "PASS"

    def test_hq1_fail_missing_log(self, gates_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        sot_mod.cmd_add_auto_approved(pd, 8)
        _create_output(pd, "planning/step-7.md")
        sot_mod.cmd_record_output(pd, 7, "planning/step-7.md")
        result = gates_mod.check_quality_gates(pd, 8)
        assert result["gates"]["HQ1_decision_log"] == "FAIL"
        assert result["valid"] is False

    def test_hq1_fail_too_small(self, gates_mod, sot_mod, tmp_project_with_sot):
        """F5: HQ1 threshold is 100 bytes (aligned with MIN_OUTPUT_SIZE)."""
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        _create_decision_log(pd, 8, size=10, template=False)  # < 100 bytes
        result = gates_mod.check_quality_gates(pd, 8)
        assert result["gates"]["HQ1_decision_log"] == "FAIL"

    def test_hq2_fail_not_approved(self, gates_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        _create_decision_log(pd, 8)
        # Don't add to auto_approved
        result = gates_mod.check_quality_gates(pd, 8)
        assert result["gates"]["HQ2_auto_approved"] == "FAIL"

    def test_hq3_fail_no_prev_output(self, gates_mod, sot_mod, tmp_project_with_sot):
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        _create_decision_log(pd, 8)
        sot_mod.cmd_add_auto_approved(pd, 8)
        # No step-7 output recorded
        result = gates_mod.check_quality_gates(pd, 8)
        assert result["gates"]["HQ3_prev_output"] == "FAIL"

    def test_hq_skip_when_autopilot_off(self, gates_mod, sot_mod, tmp_project_with_sot):
        """Autopilot OFF → human steps get SKIP regardless of flag."""
        pd = str(tmp_project_with_sot)
        # Autopilot is off by default
        import yaml
        sot_path = sot_mod._sot_path(pd)
        with open(sot_path, "r") as f:
            data = yaml.safe_load(f)
        data["workflow"]["current_step"] = 8
        with open(sot_path, "w") as f:
            yaml.dump(data, f)
        result = gates_mod.check_quality_gates(pd, 8)
        assert "human_step" in result["gates"]
        assert result["gates"]["human_step"] == "SKIP"

    def test_hq_saves_audit_log(self, gates_mod, sot_mod, tmp_project_with_sot):
        """F4: HQ gate results are auto-saved to autopilot-logs/step-N-hq-gates.json."""
        pd = str(tmp_project_with_sot)
        _enable_autopilot_at_step(sot_mod, pd, 8)
        _create_decision_log(pd, 8)
        sot_mod.cmd_add_auto_approved(pd, 8)
        _create_output(pd, "planning/step-7.md")
        sot_mod.cmd_record_output(pd, 7, "planning/step-7.md")
        gates_mod.check_quality_gates(pd, 8)
        log_path = os.path.join(pd, "autopilot-logs", "step-8-hq-gates.json")
        assert os.path.exists(log_path)
        with open(log_path, "r") as f:
            log_data = json.load(f)
        assert log_data["step"] == 8
        assert log_data["valid"] is True
        assert "HQ1_decision_log" in log_data["gates"]


# ============================================================================
# S6/S6b: validate_sot_schema in _context_lib.py
# ============================================================================

class TestS6HumanStepsValidation:
    """Tests for S6 HUMAN_STEPS_SET check in validate_sot_schema."""

    def test_s6_valid_human_steps(self, context_lib):
        ap_state = {
            "enabled": True,
            "current_step": 20,
            "auto_approved_steps": [4, 8, 18],
        }
        warnings = context_lib.validate_sot_schema(ap_state)
        s6_warnings = [w for w in warnings if "non-human" in w]
        assert len(s6_warnings) == 0

    def test_s6_non_human_step_warned(self, context_lib):
        ap_state = {
            "enabled": True,
            "current_step": 20,
            "auto_approved_steps": [5],
        }
        warnings = context_lib.validate_sot_schema(ap_state)
        assert any("non-human step" in w for w in warnings)

    def test_s6_mixed_valid_invalid(self, context_lib):
        ap_state = {
            "enabled": True,
            "current_step": 20,
            "auto_approved_steps": [4, 5, 8],
        }
        warnings = context_lib.validate_sot_schema(ap_state)
        non_human = [w for w in warnings if "non-human" in w]
        assert len(non_human) == 1
        assert "5" in non_human[0]

    def test_s6b_enabled_bool_valid(self, context_lib):
        ap_state = {"enabled": True, "auto_approved_steps": []}
        warnings = context_lib.validate_sot_schema(ap_state)
        s6b_warnings = [w for w in warnings if "autopilot.enabled" in w]
        assert len(s6b_warnings) == 0

    def test_s6b_enabled_non_bool_warned(self, context_lib):
        ap_state = {"enabled": "true", "auto_approved_steps": []}
        warnings = context_lib.validate_sot_schema(ap_state)
        assert any("autopilot.enabled" in w for w in warnings)

    def test_s6_empty_auto_approved_valid(self, context_lib):
        ap_state = {"enabled": True, "auto_approved_steps": []}
        warnings = context_lib.validate_sot_schema(ap_state)
        s6_warnings = [w for w in warnings if "auto_approved" in w]
        assert len(s6_warnings) == 0


# ============================================================================
# DL1-DL6: Decision Log P1 Validation in _context_lib.py
# ============================================================================

class TestDecisionLogValidation:
    """Tests for validate_decision_log() P1 checks (DL1-DL6)."""

    def test_dl1_fail_missing_file(self, context_lib, tmp_path):
        """DL1: Non-existent decision log → FAIL."""
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["valid"] is False
        assert result["checks"]["DL1"] == "FAIL"
        assert any("DL1" in w for w in result["warnings"])

    def test_dl1_pass_existing_file(self, context_lib, tmp_path):
        """DL1: Existing decision log → PASS (DL1 only)."""
        _create_decision_log(str(tmp_path), 8)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["checks"]["DL1"] == "PASS"

    def test_dl2_fail_too_small(self, context_lib, tmp_path):
        """DL2: Decision log < 100 bytes → FAIL."""
        _create_decision_log(str(tmp_path), 8, size=10, template=False)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["valid"] is False
        assert result["checks"]["DL2"] == "FAIL"

    def test_dl2_pass_adequate_size(self, context_lib, tmp_path):
        """DL2: Decision log >= 100 bytes → PASS."""
        _create_decision_log(str(tmp_path), 8, size=200)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["checks"]["DL2"] == "PASS"

    def test_dl3_fail_missing_sections(self, context_lib, tmp_path):
        """DL3: Raw text without required sections → FAIL."""
        _create_decision_log(str(tmp_path), 8, size=200, template=False)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["valid"] is False
        assert result["checks"]["DL3"] == "FAIL"
        assert any("DL3" in w and "Missing" in w for w in result["warnings"])

    def test_dl3_pass_all_sections_present(self, context_lib, tmp_path):
        """DL3: Template-formatted log with all sections → PASS."""
        _create_decision_log(str(tmp_path), 8, size=200, template=True)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["checks"]["DL3"] == "PASS"

    def test_dl4_fail_step_mismatch(self, context_lib, tmp_path):
        """DL4: Log says step 4 but we validate step 8 → FAIL."""
        _create_decision_log(str(tmp_path), 4, size=200, template=True)
        # Rename the file to step-8 so DL1 passes, but content says step 4
        log_dir = os.path.join(str(tmp_path), "autopilot-logs")
        os.rename(
            os.path.join(log_dir, "step-4-decision.md"),
            os.path.join(log_dir, "step-8-decision.md"),
        )
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["valid"] is False
        assert result["checks"]["DL4"] == "FAIL"
        assert any("DL4" in w and "mismatch" in w for w in result["warnings"])

    def test_dl4_pass_step_matches(self, context_lib, tmp_path):
        """DL4: Log says step 8 and we validate step 8 → PASS."""
        _create_decision_log(str(tmp_path), 8, size=200, template=True)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["checks"]["DL4"] == "PASS"

    def test_dl5_fail_rationale_too_short(self, context_lib, tmp_path):
        """DL5: Rationale field with < 10 chars → FAIL."""
        log_dir = os.path.join(str(tmp_path), "autopilot-logs")
        os.makedirs(log_dir, exist_ok=True)
        content = (
            "# Decision Log — Step 8\n\n"
            "- **Step**: 8\n"
            "- **Checkpoint Type**: (human) — Test\n"
            "- **Decision**: Proceed with quality-maximizing defaults for this step\n"
            "- **Rationale**: OK\n"  # Only 2 chars — too short
            "- **Timestamp**: 2026-02-28 12:00:00\n"
        )
        content += "\n" + "x" * (200 - len(content))
        with open(os.path.join(log_dir, "step-8-decision.md"), "w") as f:
            f.write(content)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["valid"] is False
        assert result["checks"]["DL5"] == "FAIL"

    def test_dl5_pass_rationale_adequate(self, context_lib, tmp_path):
        """DL5: Rationale with >= 10 chars → PASS."""
        _create_decision_log(str(tmp_path), 8, size=200, template=True)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["checks"]["DL5"] == "PASS"

    def test_dl6_fail_decision_too_short(self, context_lib, tmp_path):
        """DL6: Decision field with < 5 chars → FAIL."""
        log_dir = os.path.join(str(tmp_path), "autopilot-logs")
        os.makedirs(log_dir, exist_ok=True)
        content = (
            "# Decision Log — Step 8\n\n"
            "- **Step**: 8\n"
            "- **Checkpoint Type**: (human) — Test\n"
            "- **Decision**: OK\n"  # Only 2 chars — too short
            "- **Rationale**: Absolute criterion 1 — quality maximization requires full coverage\n"
            "- **Timestamp**: 2026-02-28 12:00:00\n"
        )
        content += "\n" + "x" * (200 - len(content))
        with open(os.path.join(log_dir, "step-8-decision.md"), "w") as f:
            f.write(content)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["valid"] is False
        assert result["checks"]["DL6"] == "FAIL"

    def test_dl6_pass_decision_adequate(self, context_lib, tmp_path):
        """DL6: Decision with >= 5 chars → PASS."""
        _create_decision_log(str(tmp_path), 8, size=200, template=True)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["checks"]["DL6"] == "PASS"

    def test_all_pass_template_log(self, context_lib, tmp_path):
        """All DL1-DL6 pass with a properly formatted template log."""
        _create_decision_log(str(tmp_path), 8, size=200, template=True)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["valid"] is True
        for check_id in ("DL1", "DL2", "DL3", "DL4", "DL5", "DL6"):
            assert result["checks"][check_id] == "PASS", f"{check_id} not PASS"

    def test_early_return_on_dl1_fail(self, context_lib, tmp_path):
        """DL1 failure returns immediately — no DL2-DL6 checks attempted."""
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert "DL1" in result["checks"]
        assert "DL2" not in result["checks"]

    def test_early_return_on_dl2_fail(self, context_lib, tmp_path):
        """DL2 failure returns immediately — no DL3-DL6 checks attempted."""
        _create_decision_log(str(tmp_path), 8, size=10, template=False)
        result = context_lib.validate_decision_log(str(tmp_path), 8)
        assert result["checks"]["DL1"] == "PASS"
        assert result["checks"]["DL2"] == "FAIL"
        assert "DL3" not in result["checks"]
