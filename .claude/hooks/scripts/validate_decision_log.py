#!/usr/bin/env python3
"""
Decision Log P1 Validation — validate_decision_log.py

Standalone script called by Orchestrator after (human) step auto-approval in autopilot mode.
NOT a Hook — manually invoked during workflow execution.

Usage:
    python3 .claude/hooks/scripts/validate_decision_log.py --step 8 --project-dir .

Output: JSON to stdout
    {"valid": true, "warnings": [], "checks": {"DL1": "PASS", ...}}

Exit codes:
    0 — validation completed (check "valid" field for result)
    1 — argument error or fatal failure

P1 Compliance: All validation is deterministic (delegates to _context_lib).
SOT Compliance: Read-only — no file writes.
"""

import argparse
import json
import os
import sys

# Add script directory to path for shared library import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _context_lib import validate_decision_log


def main():
    parser = argparse.ArgumentParser(
        description="P1 Validation for Autopilot Decision Log outputs (DL1-DL6)"
    )
    parser.add_argument(
        "--step", type=int, required=True,
        help="Step number to validate"
    )
    parser.add_argument(
        "--project-dir", type=str, default=".",
        help="Project root directory (default: current directory)"
    )
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    result = validate_decision_log(project_dir, args.step)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
