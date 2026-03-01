#!/usr/bin/env python3
"""Quality Gate Sequencer — P1 deterministic gate ordering.

Enforces L0 → L1 → L1.5 → L2 order. Prevents skipping gates.
The Orchestrator calls this script instead of manually tracking gate results.

Usage:
    python3 scripts/run_quality_gates.py --step 3 --project-dir .
    python3 scripts/run_quality_gates.py --step 3 --project-dir . --check-review
    python3 scripts/run_quality_gates.py --step 3 --project-dir . --skip-review
    python3 scripts/run_quality_gates.py --step 8 --project-dir .
        (human steps: auto-detects autopilot from SOT, runs HQ1/HQ2/HQ3 if enabled)

JSON output to stdout. Exit code 0 always.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Constants — Step configuration
# ---------------------------------------------------------------------------

# Steps that require review (L2) — from workflow.md
REVIEW_STEPS = {1, 3, 5, 7, 16, 19, 20}

# D-7 intentional duplication — must match sot_manager.py:HUMAN_STEPS,
# validate_step_transition.py:HUMAN_STEPS, _context_lib.py:HUMAN_STEPS_SET,
# and prompt/workflow.md "Steps 4, 8, 18"
HUMAN_STEPS = frozenset({4, 8, 18})

# Steps that require translation — from workflow.md
# D-7 cross-reference: must match ORCHESTRATOR-PLAYBOOK.md Step-Type Quick Map "Translation" column
TRANSLATION_STEPS = {1, 3, 5, 7, 16, 19, 20}

# Pre/post processing scripts per step
STEP_SCRIPTS = {
    1: {
        "pre": ["scripts/extract_site_urls.py"],
        "post": ["scripts/generate_sources_yaml_draft.py"],
    },
    3: {
        "pre": ["scripts/merge_recon_and_deps.py"],
        "post": [],
    },
    5: {
        "pre": ["scripts/filter_prd_architecture.py"],
        "post": ["scripts/validate_data_schema.py"],
    },
    6: {
        "pre": ["scripts/split_sites_by_group.py"],
        "post": ["scripts/validate_site_coverage.py"],
    },
    7: {
        "pre": ["scripts/filter_prd_analysis.py"],
        "post": [],
    },
    10: {
        "pre": ["scripts/extract_architecture_crawling.py"],
        "post": [],
    },
    11: {
        "pre": ["scripts/distribute_sites_to_teams.py"],
        "post": ["scripts/verify_adapter_coverage.py"],
    },
    13: {
        "pre": ["scripts/extract_pipeline_design_s1_s4.py"],
        "post": [],
    },
    14: {
        "pre": ["scripts/extract_pipeline_design_s5_s8.py"],
        "post": [],
    },
    16: {
        "pre": [],
        "post": ["scripts/calculate_success_metrics.py"],
    },
}


def _run_validator(cmd, project_dir):
    """Run a validation script and return parsed JSON result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=project_dir,
        )
        if result.stdout.strip():
            return json.loads(result.stdout.strip())
        return {"valid": False, "error": f"No output from {cmd[0]}", "stderr": result.stderr[:500]}
    except subprocess.TimeoutExpired:
        return {"valid": False, "error": f"Timeout running {cmd[0]}"}
    except json.JSONDecodeError:
        return {"valid": False, "error": f"Invalid JSON from {cmd[0]}"}
    except FileNotFoundError:
        return {"valid": False, "error": f"Script not found: {cmd[0]}"}


def _read_autopilot_state(project_dir):
    """Read autopilot state from SOT via sot_manager.py --read (canonical reader).

    Uses subprocess to avoid duplicating SOT parsing logic.
    Returns (state_dict | None, error_str | None).
    - (dict, None): autopilot enabled, dict has keys enabled/auto_approved_steps/current_step/outputs
    - (None, None): autopilot not enabled or no SOT — not an error
    - (None, error_str): SOT read failed — caller must treat as blocking
    """
    try:
        sot_manager = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "sot_manager.py")
        result = subprocess.run(
            [sys.executable, sot_manager, "--read", "--project-dir", project_dir],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if data.get("valid"):
                wf = data.get("workflow", {})
                ap = wf.get("autopilot")
                if isinstance(ap, dict) and ap.get("enabled"):
                    return {
                        "enabled": True,
                        "auto_approved_steps": ap.get("auto_approved_steps", []),
                        "current_step": wf.get("current_step", 0),
                        "outputs": wf.get("outputs", {}),
                    }, None
                return None, None  # Autopilot not enabled — not an error
            return None, f"SOT read returned valid=false: {data.get('error', 'unknown')}"
        if result.returncode != 0:
            return None, f"sot_manager.py exit code {result.returncode}: {result.stderr[:200]}"
        return None, None  # Empty stdout, no SOT
    except subprocess.TimeoutExpired:
        return None, "sot_manager.py --read timed out (10s)"
    except Exception as e:
        return None, f"SOT read failed: {e}"


def _check_human_quality_gates(project_dir, step_num, ap_state):
    """Run HQ1/HQ2/HQ3 human-step quality gates for autopilot mode.

    Returns dict with gate results.
    """
    gates = {}
    blocking = []

    # HQ1: Decision log exists + minimum size (100 bytes — aligned with MIN_OUTPUT_SIZE)
    log_path = os.path.join(project_dir, "autopilot-logs", f"step-{step_num}-decision.md")
    if not os.path.exists(log_path):
        gates["HQ1_decision_log"] = "FAIL"
        blocking.append(f"HQ1: Decision log missing: autopilot-logs/step-{step_num}-decision.md")
    else:
        size = os.path.getsize(log_path)
        if size < 100:
            gates["HQ1_decision_log"] = "FAIL"
            blocking.append(f"HQ1: Decision log too small: {size} bytes (min 100)")
        else:
            gates["HQ1_decision_log"] = "PASS"

    # HQ2: Step recorded in auto_approved_steps
    aas = ap_state.get("auto_approved_steps", []) if ap_state else []
    if step_num in aas:
        gates["HQ2_auto_approved"] = "PASS"
    else:
        gates["HQ2_auto_approved"] = "FAIL"
        blocking.append(f"HQ2: Step {step_num} not in auto_approved_steps: {aas}")

    # HQ3: Previous step output L0 check (input validity)
    prev_step = step_num - 1
    if prev_step >= 1 and prev_step not in HUMAN_STEPS:
        prev_key = f"step-{prev_step}"
        # Use outputs from ap_state (already fetched via canonical reader)
        outputs = ap_state.get("outputs", {}) if ap_state else {}

        if prev_key not in outputs:
            gates["HQ3_prev_output"] = "FAIL"
            blocking.append(f"HQ3: No output recorded for previous {prev_key}")
        else:
            out_path = outputs[prev_key]
            full = os.path.join(project_dir, out_path) if not os.path.isabs(out_path) else out_path
            if not os.path.exists(full):
                gates["HQ3_prev_output"] = "FAIL"
                blocking.append(f"HQ3: Previous step output file not found: {out_path}")
            elif os.path.getsize(full) < 100:
                gates["HQ3_prev_output"] = "FAIL"
                blocking.append(f"HQ3: Previous step output too small: {os.path.getsize(full)} bytes")
            else:
                gates["HQ3_prev_output"] = "PASS"
    else:
        gates["HQ3_prev_output"] = "N/A"

    return gates, blocking


def _save_hq_log(project_dir, step_num, result):
    """Save HQ gate results to autopilot-logs/step-N-hq-gates.json for audit trail.

    Best-effort write — failure is logged as warning but does not block.
    """
    import datetime
    log_dir = os.path.join(project_dir, "autopilot-logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"step-{step_num}-hq-gates.json")
        log_data = {
            "step": step_num,
            "timestamp": datetime.datetime.now().isoformat(),
            "valid": result.get("valid"),
            "gates": result.get("gates", {}),
            "blocking": result.get("blocking", []),
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    except Exception:
        # Best-effort — audit trail failure should not block gate execution
        result.setdefault("warnings", []).append(
            f"Failed to save HQ log to autopilot-logs/step-{step_num}-hq-gates.json"
        )


def check_quality_gates(project_dir, step_num, check_review=False, check_autopilot=False):
    """Run quality gates in order: L0 → L1 → L1.5 → L2.

    For human steps (4, 8, 18): auto-detects autopilot state from SOT.
    If autopilot enabled → runs HQ1/HQ2/HQ3 gates.
    If autopilot disabled → SKIP (manual mode).
    The check_autopilot parameter is kept for backward compatibility but ignored
    — SOT state is the sole determinant (P1: no LLM memory dependency).
    """
    result = {
        "valid": True,
        "step": step_num,
        "gates": {},
        "blocking": [],
        "warnings": [],
        "scripts_status": {"pre": {}, "post": {}},
    }

    # Human steps: auto-detect autopilot state from SOT, run HQ gates if enabled
    if step_num in HUMAN_STEPS:
        ap_state, ap_error = _read_autopilot_state(project_dir)
        if ap_error:
            # Fail-safe: SOT read failure is blocking, not silent skip
            result["gates"]["SOT_read"] = "FAIL"
            result["blocking"].append(f"SOT: Cannot determine autopilot state: {ap_error}")
            result["valid"] = False
            _save_hq_log(project_dir, step_num, result)
            return result
        if ap_state and ap_state.get("enabled"):
            hq_gates, hq_blocking = _check_human_quality_gates(project_dir, step_num, ap_state)
            result["gates"].update(hq_gates)
            result["blocking"].extend(hq_blocking)
            if hq_blocking:
                result["valid"] = False
            _save_hq_log(project_dir, step_num, result)
            return result
        result["gates"]["human_step"] = "SKIP"
        return result

    hooks_dir = os.path.join(project_dir, ".claude", "hooks", "scripts")

    # --- Pre-processing scripts check ---
    step_scripts = STEP_SCRIPTS.get(step_num, {"pre": [], "post": []})
    for script in step_scripts.get("pre", []):
        script_path = os.path.join(project_dir, script)
        result["scripts_status"]["pre"][script] = "EXISTS" if os.path.exists(script_path) else "MISSING"

    # --- L0: Anti-Skip Guard ---
    l0_cmd = [
        sys.executable,
        os.path.join(hooks_dir, "validate_pacs.py"),
        "--step", str(step_num),
        "--check-l0",
        "--project-dir", project_dir,
    ]
    l0_result = _run_validator(l0_cmd, project_dir)
    l0_valid = l0_result.get("l0_valid", l0_result.get("valid", False))
    result["gates"]["L0_anti_skip"] = "PASS" if l0_valid else "FAIL"
    if not l0_valid:
        l0_warnings = l0_result.get("l0_warnings", l0_result.get("warnings", []))
        result["blocking"].append(f"L0: Anti-Skip Guard failed: {l0_warnings}")
        result["valid"] = False
        return result  # Cannot proceed without L0

    # --- L1: Verification Gate ---
    l1_cmd = [
        sys.executable,
        os.path.join(hooks_dir, "validate_verification.py"),
        "--step", str(step_num),
        "--project-dir", project_dir,
    ]
    l1_result = _run_validator(l1_cmd, project_dir)
    l1_valid = l1_result.get("valid", False)
    result["gates"]["L1_verification"] = "PASS" if l1_valid else "FAIL"
    if not l1_valid:
        result["blocking"].append(f"L1: Verification Gate failed: {l1_result.get('warnings', [])}")
        result["valid"] = False
        return result  # Cannot proceed to L1.5 without L1

    # --- L1.5: pACS Self-Rating ---
    l15_cmd = [
        sys.executable,
        os.path.join(hooks_dir, "validate_pacs.py"),
        "--step", str(step_num),
        "--project-dir", project_dir,
    ]
    l15_result = _run_validator(l15_cmd, project_dir)
    l15_valid = l15_result.get("valid", False)
    result["gates"]["L1_5_pacs"] = "PASS" if l15_valid else "FAIL"
    if not l15_valid:
        result["blocking"].append(f"L1.5: pACS validation failed: {l15_result.get('warnings', [])}")
        result["valid"] = False
        return result  # Cannot proceed to L2 without L1.5

    # --- L2: Adversarial Review (optional) ---
    if check_review and step_num in REVIEW_STEPS:
        l2_cmd = [
            sys.executable,
            os.path.join(hooks_dir, "validate_review.py"),
            "--step", str(step_num),
            "--project-dir", project_dir,
            "--check-pacs-arithmetic",
        ]
        l2_result = _run_validator(l2_cmd, project_dir)
        l2_valid = l2_result.get("valid", False)
        verdict = l2_result.get("verdict", "UNKNOWN")
        result["gates"]["L2_review"] = f"{'PASS' if l2_valid and verdict == 'PASS' else 'FAIL'} (verdict={verdict})"
        if not l2_valid or verdict == "FAIL":
            result["blocking"].append(f"L2: Review validation failed: verdict={verdict}")
            result["valid"] = False
    elif step_num in REVIEW_STEPS:
        result["gates"]["L2_review"] = "PENDING"
        result["warnings"].append("L2: Review step — run with --check-review after review completion")
    else:
        result["gates"]["L2_review"] = "N/A"

    # --- Post-processing scripts check ---
    for script in step_scripts.get("post", []):
        script_path = os.path.join(project_dir, script)
        result["scripts_status"]["post"][script] = "EXISTS" if os.path.exists(script_path) else "MISSING"

    # --- Translation check (P1 structural validation, not just existence) ---
    if step_num in TRANSLATION_STEPS:
        ko_path = os.path.join(project_dir, "pacs-logs", f"step-{step_num}-translation-pacs.md")
        if not os.path.exists(ko_path):
            result["warnings"].append(f"Translation pACS log not found for step {step_num}")
            result["gates"]["translation_pacs"] = "MISSING"
        else:
            # Validate via P1 script if available
            trans_validator = os.path.join(hooks_dir, "validate_translation.py")
            if os.path.exists(trans_validator):
                trans_cmd = [
                    sys.executable, trans_validator,
                    "--step", str(step_num),
                    "--project-dir", project_dir,
                    "--check-pacs",
                ]
                trans_result = _run_validator(trans_cmd, project_dir)
                trans_valid = trans_result.get("valid", False)
                result["gates"]["translation_pacs"] = "PASS" if trans_valid else "FAIL"
                if not trans_valid:
                    result["warnings"].append(
                        f"Translation P1 validation failed: {trans_result.get('warnings', [])}"
                    )
            else:
                result["gates"]["translation_pacs"] = "EXISTS"

    return result


def main():
    parser = argparse.ArgumentParser(description="Quality Gate Sequencer — P1")
    parser.add_argument("--step", type=int, required=True, help="Step number")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--check-review", action="store_true", help="Include L2 review check")
    parser.add_argument("--skip-review", action="store_true", help="Skip L2 even for review steps")
    parser.add_argument("--check-autopilot", action="store_true",
                        help="Deprecated — HQ gates now auto-detect from SOT. Kept for backward compat.")
    args = parser.parse_args()

    check_review = args.check_review and not args.skip_review
    result = check_quality_gates(args.project_dir, args.step, check_review)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
