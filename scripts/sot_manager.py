#!/usr/bin/env python3
"""SOT Manager — Deterministic SOT read/write/validate for state.yaml.

P1 Hallucination Prevention: All SOT mutations go through this script.
The Orchestrator (LLM) MUST NOT directly edit state.yaml.

Usage:
    python3 scripts/sot_manager.py --read --project-dir .
    python3 scripts/sot_manager.py --advance-step 3 --project-dir .
    python3 scripts/sot_manager.py --record-output 3 research/output.md --project-dir .
    python3 scripts/sot_manager.py --update-pacs 3 --F 85 --C 78 --L 80 --project-dir .
    python3 scripts/sot_manager.py --update-team '{"name":"team-x","status":"partial","tasks_completed":[],"tasks_pending":["t1"]}' --project-dir .
    python3 scripts/sot_manager.py --init --workflow-name "GlobalNews Auto-Build" --total-steps 20 --project-dir .
    python3 scripts/sot_manager.py --set-autopilot true --project-dir .
    python3 scripts/sot_manager.py --add-auto-approved 8 --project-dir .

All output is JSON to stdout. Exit code 0 always (errors in JSON).
"""

import argparse
import fcntl
import json
import os
import re
import sys
import tempfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# D-7 intentional duplication — must match _context_lib.py:SOT_FILENAMES
SOT_FILENAMES = ("state.yaml", "state.yml", "state.json")
MIN_OUTPUT_SIZE = 100  # bytes — L0 Anti-Skip Guard threshold
# D-7 intentional duplication — must match _context_lib.py:validate_sot_schema() valid_statuses
VALID_STATUSES = {"in_progress", "completed", "failed", "running", "error", "paused"}
VALID_TEAM_STATUSES = {"partial", "all_completed"}
# D-7 intentional duplication — must match run_quality_gates.py:HUMAN_STEPS,
# validate_step_transition.py:HUMAN_STEPS, _context_lib.py:HUMAN_STEPS_SET,
# and prompt/workflow.md "Steps 4, 8, 18"
HUMAN_STEPS = frozenset({4, 8, 18})

# ---------------------------------------------------------------------------
# YAML helpers (PyYAML preferred, regex fallback)
# ---------------------------------------------------------------------------

def _load_yaml(text):
    """Parse YAML text. Returns dict or raises."""
    try:
        import yaml
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("YAML root is not a mapping")
        return data
    except ImportError:
        raise ImportError("PyYAML is required for sot_manager.py")


def _dump_yaml(data):
    """Serialize dict to YAML string."""
    import yaml
    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# SOT path resolution
# ---------------------------------------------------------------------------

def _sot_path(project_dir):
    """Find existing SOT file or return default path."""
    for fn in SOT_FILENAMES:
        p = os.path.join(project_dir, ".claude", fn)
        if os.path.exists(p):
            return p
    # Default: state.yaml
    return os.path.join(project_dir, ".claude", "state.yaml")


def _read_sot(project_dir):
    """Read and parse SOT with shared lock. Returns (data_dict, file_path) or raises.

    For read-only commands (--read). Mutating commands use _read_sot_unlocked()
    inside an exclusive _SOTLock context.
    """
    sot = _sot_path(project_dir)
    with _SOTLock(sot, exclusive=False):
        return _read_sot_unlocked(project_dir)


def _extract_wf(data):
    """Extract workflow dict from SOT data (handles nesting)."""
    wf = data.get("workflow")
    if isinstance(wf, dict):
        return wf
    # Flat schema — data itself is the workflow
    return data


def _write_sot_atomic(sot_path, data):
    """Atomic write: temp file → rename."""
    content = _dump_yaml(data)
    dir_path = os.path.dirname(sot_path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, sot_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Concurrent access safety (fcntl — Unix/macOS)
# ---------------------------------------------------------------------------

class _SOTLock:
    """File-based lock for SOT concurrent access safety.

    Prevents Lost Update when multiple teammates call --update-team
    simultaneously during (team) steps. Uses fcntl.flock() (Unix only).

    Usage:
        with _SOTLock(sot_path, exclusive=True):
            data, sot = _read_sot_unlocked(...)
            ... modify ...
            _write_sot_atomic(sot, data)
    """

    def __init__(self, sot_path, exclusive=False):
        self._lock_path = sot_path + ".lock"
        self._exclusive = exclusive
        self._fd = None

    def __enter__(self):
        self._fd = open(self._lock_path, "w")
        mode = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        fcntl.flock(self._fd, mode)
        return self

    def __exit__(self, *args):
        if self._fd:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()


def _read_sot_unlocked(project_dir):
    """Read and parse SOT WITHOUT locking. Internal use only."""
    sot = _sot_path(project_dir)
    if not os.path.exists(sot):
        raise FileNotFoundError(f"SOT not found: {sot}")
    with open(sot, "r", encoding="utf-8") as f:
        content = f.read()
    data = _load_yaml(content)
    return data, sot


# ---------------------------------------------------------------------------
# Step number validation helper
# ---------------------------------------------------------------------------

def _validate_step_num(wf, step_num, context="operation"):
    """Validate step_num is within [1, total_steps]. Returns error dict or None.

    CR-1: Prevents negative, zero, or out-of-range step numbers from
    corrupting SOT state.
    """
    if not isinstance(step_num, int) or step_num < 1:
        return {
            "valid": False,
            "error": f"SM-R1: step_num={step_num} must be int >= 1 (context: {context})",
        }
    total = wf.get("total_steps")
    if total is not None and isinstance(total, int) and step_num > total:
        return {
            "valid": False,
            "error": f"SM-R1: step_num={step_num} exceeds total_steps={total} (context: {context})",
        }
    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_schema(wf):
    """Validate SOT schema. Returns list of warnings."""
    warnings = []

    # SM1: current_step must be int >= 0
    cs = wf.get("current_step")
    if cs is not None:
        if not isinstance(cs, int):
            warnings.append(f"SM1: current_step is {type(cs).__name__}, expected int")
        elif cs < 0:
            warnings.append(f"SM1: current_step is {cs}, must be >= 0")

    # SM2: outputs must be dict
    outputs = wf.get("outputs")
    if outputs is not None and not isinstance(outputs, dict):
        warnings.append(f"SM2: outputs is {type(outputs).__name__}, expected dict")

    # SM3: status must be valid
    status = wf.get("status", "")
    if status and status not in VALID_STATUSES:
        warnings.append(f"SM3: status '{status}' not in {VALID_STATUSES}")

    # SM-AP1: autopilot.enabled must be bool (if autopilot section exists)
    ap = wf.get("autopilot")
    if ap is not None:
        if not isinstance(ap, dict):
            warnings.append(f"SM-AP1: autopilot is {type(ap).__name__}, expected dict")
        else:
            enabled = ap.get("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                warnings.append(f"SM-AP2: autopilot.enabled is {type(enabled).__name__}, expected bool")

            # SM-AP3: auto_approved_steps must be list of ints in HUMAN_STEPS
            aas = ap.get("auto_approved_steps")
            if aas is not None:
                if not isinstance(aas, list):
                    warnings.append(f"SM-AP3: auto_approved_steps is {type(aas).__name__}, expected list")
                else:
                    for item in aas:
                        if not isinstance(item, int):
                            warnings.append(f"SM-AP3: auto_approved_steps contains non-int: {item}")
                        elif item not in HUMAN_STEPS:
                            warnings.append(f"SM-AP4: auto_approved_steps contains non-human step: {item}")

    return warnings


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_read(project_dir):
    """Read SOT and return as JSON."""
    try:
        data, sot = _read_sot(project_dir)
        wf = _extract_wf(data)
        schema_warnings = _validate_schema(wf)
        return {
            "valid": True,
            "sot_path": sot,
            "workflow": wf,
            "schema_warnings": schema_warnings,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def cmd_init(project_dir, workflow_name, total_steps):
    """Initialize a new SOT file."""
    sot = _sot_path(project_dir)
    if os.path.exists(sot):
        return {"valid": False, "error": f"SOT already exists: {sot}. Delete first to re-initialize."}

    data = {
        "workflow": {
            "name": workflow_name,
            "current_step": 1,
            "status": "in_progress",
            "total_steps": total_steps,
            "parent_genome": {
                "source": "AgenticWorkflow",
                "version": "2026-02-25",
                "inherited_dna": [
                    "absolute-criteria", "sot-pattern", "3-phase-structure",
                    "4-layer-qa", "safety-hooks", "adversarial-review",
                    "decision-log", "context-preservation",
                    "cross-step-traceability",
                ],
            },
            "outputs": {},
            "autopilot": {
                "enabled": False,
                "auto_approved_steps": [],
            },
            "pending_human_action": {"step": None, "options": []},
            "verification": {"last_verified_step": 0, "retries": {}},
            "pacs": {
                "current_step_score": None,
                "dimensions": {"F": None, "C": None, "L": None},
                "weak_dimension": None,
                "pre_mortem_flag": None,
                "history": {},
            },
        }
    }

    os.makedirs(os.path.dirname(sot), exist_ok=True)
    _write_sot_atomic(sot, data)
    return {"valid": True, "sot_path": sot, "action": "initialized", "workflow_name": workflow_name}


def cmd_advance_step(project_dir, step_num):
    """Advance current_step from step_num to step_num+1."""
    try:
        sot = _sot_path(project_dir)
        with _SOTLock(sot, exclusive=True):
            data, sot = _read_sot_unlocked(project_dir)
            wf = _extract_wf(data)

            # CR-1: Step range validation
            range_err = _validate_step_num(wf, step_num, "advance-step")
            if range_err:
                return range_err

            # MD-1: Cannot advance past total_steps
            total = wf.get("total_steps")
            if total is not None and isinstance(total, int) and step_num + 1 > total + 1:
                return {
                    "valid": False,
                    "error": f"SM-R2: advance would set current_step={step_num + 1}, but total_steps={total}. Workflow already complete.",
                }

            # SM3: Pre-condition — current_step must equal step_num
            cs = wf.get("current_step", 0)
            if cs != step_num:
                return {
                    "valid": False,
                    "error": f"SM3: current_step is {cs}, expected {step_num}. Cannot advance.",
                }

            # SM4: Pre-condition — step-N output must exist in outputs
            step_key = f"step-{step_num}"
            outputs = wf.get("outputs", {})
            if step_key not in outputs:
                return {
                    "valid": False,
                    "error": f"SM4: No output recorded for {step_key}. Record output first.",
                }

            # SM4b: Output file must exist on disk
            output_path = outputs[step_key]
            full_path = os.path.join(project_dir, output_path) if not os.path.isabs(output_path) else output_path
            if not os.path.exists(full_path):
                return {
                    "valid": False,
                    "error": f"SM4b: Output file does not exist: {output_path}",
                }

            # Advance
            wf["current_step"] = step_num + 1
            _write_sot_atomic(sot, data)

            return {
                "valid": True,
                "action": "advanced",
                "previous_step": step_num,
                "current_step": step_num + 1,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def cmd_record_output(project_dir, step_num, output_path):
    """Record output path for step-N in SOT."""
    try:
        sot = _sot_path(project_dir)
        with _SOTLock(sot, exclusive=True):
            data, sot = _read_sot_unlocked(project_dir)
            wf = _extract_wf(data)

            # CR-1: Step range validation
            range_err = _validate_step_num(wf, step_num, "record-output")
            if range_err:
                return range_err

            # HI-1: Cannot record output for future steps
            cs = wf.get("current_step", 1)
            if step_num > cs:
                return {
                    "valid": False,
                    "error": f"SM-R3: Cannot record output for step-{step_num} when current_step={cs}. Future step outputs are forbidden.",
                }

            # SM6: File must exist and meet minimum size
            full_path = os.path.join(project_dir, output_path) if not os.path.isabs(output_path) else output_path
            if not os.path.exists(full_path):
                return {
                    "valid": False,
                    "error": f"SM6: Output file does not exist: {output_path}",
                }
            file_size = os.path.getsize(full_path)
            if file_size < MIN_OUTPUT_SIZE:
                return {
                    "valid": False,
                    "error": f"SM6: Output file too small: {file_size} bytes (min {MIN_OUTPUT_SIZE})",
                }

            # SM7: Path format validation
            if not isinstance(output_path, str) or not output_path.strip():
                return {"valid": False, "error": "SM7: Output path must be non-empty string"}

            # Record
            step_key = f"step-{step_num}"
            if "outputs" not in wf:
                wf["outputs"] = {}
            wf["outputs"][step_key] = output_path
            _write_sot_atomic(sot, data)

            return {
                "valid": True,
                "action": "output_recorded",
                "step": step_num,
                "output_path": output_path,
                "file_size": file_size,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def cmd_update_pacs(project_dir, step_num, f_score, c_score, l_score):
    """Update pACS scores for step-N in SOT."""
    try:
        sot = _sot_path(project_dir)
        with _SOTLock(sot, exclusive=True):
            data, sot = _read_sot_unlocked(project_dir)
            wf = _extract_wf(data)

            # CR-1: Step range validation
            range_err = _validate_step_num(wf, step_num, "update-pacs")
            if range_err:
                return range_err

            # SM10: Validate dimension ranges
            for name, val in [("F", f_score), ("C", c_score), ("L", l_score)]:
                if not isinstance(val, (int, float)) or not (0 <= val <= 100):
                    return {
                        "valid": False,
                        "error": f"SM10: {name}={val} must be int/float in [0,100]",
                    }

            # SM11: Compute min
            pacs_score = min(f_score, c_score, l_score)
            weak = "F" if f_score == pacs_score else ("C" if c_score == pacs_score else "L")

            # Determine zone
            if pacs_score < 50:
                zone = "RED"
            elif pacs_score < 70:
                zone = "YELLOW"
            else:
                zone = "GREEN"

            # Update
            if "pacs" not in wf:
                wf["pacs"] = {}
            wf["pacs"]["current_step_score"] = pacs_score
            wf["pacs"]["dimensions"] = {"F": f_score, "C": c_score, "L": l_score}
            wf["pacs"]["weak_dimension"] = weak

            if "history" not in wf["pacs"]:
                wf["pacs"]["history"] = {}
            wf["pacs"]["history"][f"step-{step_num}"] = {
                "score": pacs_score,
                "weak": weak,
            }

            _write_sot_atomic(sot, data)

            return {
                "valid": True,
                "action": "pacs_updated",
                "step": step_num,
                "pacs_score": pacs_score,
                "dimensions": {"F": f_score, "C": c_score, "L": l_score},
                "weak_dimension": weak,
                "zone": zone,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def cmd_update_team(project_dir, team_json):
    """Update active_team state in SOT."""
    try:
        team_data = json.loads(team_json)
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}

    try:
        sot = _sot_path(project_dir)
        with _SOTLock(sot, exclusive=True):
            data, sot = _read_sot_unlocked(project_dir)
            wf = _extract_wf(data)

            # SM8: Validate required fields
            name = team_data.get("name")
            if not name or not isinstance(name, str):
                return {"valid": False, "error": "SM8: active_team.name must be non-empty string"}

            status = team_data.get("status", "partial")
            if status not in VALID_TEAM_STATUSES:
                return {"valid": False, "error": f"SM8: status '{status}' not in {VALID_TEAM_STATUSES}"}

            tc = team_data.get("tasks_completed", [])
            tp = team_data.get("tasks_pending", [])
            cs = team_data.get("completed_summaries", {})

            # SM9: tasks_completed must be subset of all tasks
            if not isinstance(tc, list) or not isinstance(tp, list):
                return {"valid": False, "error": "SM9: tasks_completed and tasks_pending must be lists"}

            # CR-3: Task overlap check — no task can be both completed AND pending
            overlap = set(tc) & set(tp)
            if overlap:
                return {
                    "valid": False,
                    "error": f"SM-R4: Tasks appear in both completed and pending: {sorted(overlap)}. Logical contradiction.",
                }

            # SM9b: completed_summaries keys must be subset of tasks_completed
            if isinstance(cs, dict):
                for k in cs:
                    if k not in tc:
                        return {
                            "valid": False,
                            "error": f"SM9b: completed_summaries key '{k}' not in tasks_completed",
                        }

            # Update
            wf["active_team"] = {
                "name": name,
                "status": status,
                "tasks_completed": tc,
                "tasks_pending": tp,
                "completed_summaries": cs,
            }

            # If all_completed, move to completed_teams
            if status == "all_completed":
                if "completed_teams" not in wf:
                    wf["completed_teams"] = []
                wf["completed_teams"].append(wf["active_team"].copy())

            _write_sot_atomic(sot, data)

            return {
                "valid": True,
                "action": "team_updated",
                "team_name": name,
                "status": status,
                "tasks_completed_count": len(tc),
                "tasks_pending_count": len(tp),
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def cmd_set_autopilot(project_dir, enabled_str):
    """Set autopilot.enabled to true or false."""
    # SM-AP1: Validate input value
    if enabled_str not in ("true", "false"):
        return {"valid": False, "error": f"SM-AP1: enabled must be 'true' or 'false', got '{enabled_str}'"}
    enabled = enabled_str == "true"
    try:
        sot = _sot_path(project_dir)
        with _SOTLock(sot, exclusive=True):
            data, sot = _read_sot_unlocked(project_dir)
            wf = _extract_wf(data)

            # Ensure autopilot section exists
            if "autopilot" not in wf or not isinstance(wf.get("autopilot"), dict):
                wf["autopilot"] = {"enabled": False, "auto_approved_steps": []}

            old_val = wf["autopilot"].get("enabled", False)
            wf["autopilot"]["enabled"] = enabled
            _write_sot_atomic(sot, data)

            return {
                "valid": True,
                "action": "autopilot_set",
                "previous": old_val,
                "enabled": enabled,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def cmd_add_auto_approved(project_dir, step_num):
    """Record a human step as auto-approved in autopilot.auto_approved_steps."""
    # SM-AA1: Must be a human step
    if step_num not in HUMAN_STEPS:
        return {
            "valid": False,
            "error": f"SM-AA1: step {step_num} is not a human step. HUMAN_STEPS = {sorted(HUMAN_STEPS)}",
        }
    try:
        sot = _sot_path(project_dir)
        with _SOTLock(sot, exclusive=True):
            data, sot = _read_sot_unlocked(project_dir)
            wf = _extract_wf(data)

            # SM-AA2: autopilot must be enabled
            ap = wf.get("autopilot")
            if not isinstance(ap, dict) or not ap.get("enabled"):
                return {
                    "valid": False,
                    "error": "SM-AA2: autopilot is not enabled. Enable with --set-autopilot true first.",
                }

            # SM-AA4: Cannot approve future steps
            cs = wf.get("current_step", 0)
            if step_num > cs:
                return {
                    "valid": False,
                    "error": f"SM-AA4: cannot approve future step {step_num} (current_step={cs})",
                }

            # SM-AA3: Idempotent — already present is success
            aas = ap.get("auto_approved_steps", [])
            if not isinstance(aas, list):
                aas = []
            if step_num in aas:
                return {
                    "valid": True,
                    "action": "already_approved",
                    "step": step_num,
                    "auto_approved_steps": sorted(aas),
                }

            aas.append(step_num)
            aas.sort()
            ap["auto_approved_steps"] = aas
            _write_sot_atomic(sot, data)

            return {
                "valid": True,
                "action": "auto_approved",
                "step": step_num,
                "auto_approved_steps": aas,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def cmd_set_status(project_dir, new_status):
    """Set workflow status (e.g., in_progress → completed)."""
    if new_status not in VALID_STATUSES:
        return {"valid": False, "error": f"Invalid status '{new_status}'. Must be one of: {sorted(VALID_STATUSES)}"}
    try:
        sot = _sot_path(project_dir)
        with _SOTLock(sot, exclusive=True):
            data, sot = _read_sot_unlocked(project_dir)
            wf = _extract_wf(data)
            old_status = wf.get("status", "unknown")
            wf["status"] = new_status
            _write_sot_atomic(sot, data)
            return {
                "valid": True,
                "action": "status_updated",
                "previous_status": old_status,
                "new_status": new_status,
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SOT Manager — P1 deterministic SOT operations")
    parser.add_argument("--project-dir", required=True, help="Project root directory")
    parser.add_argument("--read", action="store_true", help="Read and validate SOT")
    parser.add_argument("--init", action="store_true", help="Initialize new SOT")
    parser.add_argument("--workflow-name", default="", help="Workflow name (for --init)")
    parser.add_argument("--total-steps", type=int, default=20, help="Total steps (for --init)")
    parser.add_argument("--advance-step", type=int, help="Advance from step N to N+1")
    parser.add_argument("--record-output", nargs=2, metavar=("STEP", "PATH"), help="Record output for step")
    parser.add_argument("--update-pacs", type=int, metavar="STEP", help="Update pACS for step")
    parser.add_argument("--F", type=float, help="F dimension score")
    parser.add_argument("--C", type=float, help="C dimension score")
    parser.add_argument("--L", type=float, help="L dimension score")
    parser.add_argument("--update-team", metavar="JSON", help="Update active_team (JSON string)")
    parser.add_argument("--set-status", metavar="STATUS", help="Set workflow status (in_progress, completed, failed, etc.)")
    parser.add_argument("--set-autopilot", metavar="BOOL", help="Set autopilot enabled (true/false)")
    parser.add_argument("--add-auto-approved", type=int, metavar="STEP", help="Record a human step as auto-approved")

    args = parser.parse_args()

    if args.read:
        result = cmd_read(args.project_dir)
    elif args.init:
        result = cmd_init(args.project_dir, args.workflow_name, args.total_steps)
    elif args.advance_step is not None:
        result = cmd_advance_step(args.project_dir, args.advance_step)
    elif args.record_output:
        step = int(args.record_output[0])
        path = args.record_output[1]
        result = cmd_record_output(args.project_dir, step, path)
    elif args.update_pacs is not None:
        if args.F is None or args.C is None or args.L is None:
            result = {"valid": False, "error": "pACS update requires --F, --C, --L"}
        else:
            result = cmd_update_pacs(args.project_dir, args.update_pacs, args.F, args.C, args.L)
    elif args.update_team:
        result = cmd_update_team(args.project_dir, args.update_team)
    elif args.set_status:
        result = cmd_set_status(args.project_dir, args.set_status)
    elif args.set_autopilot:
        result = cmd_set_autopilot(args.project_dir, args.set_autopilot)
    elif args.add_auto_approved is not None:
        result = cmd_add_auto_approved(args.project_dir, args.add_auto_approved)
    else:
        result = {"valid": False, "error": "No command specified. Use --read, --init, --advance-step, --record-output, --update-pacs, --update-team, --set-status, --set-autopilot, or --add-auto-approved"}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
