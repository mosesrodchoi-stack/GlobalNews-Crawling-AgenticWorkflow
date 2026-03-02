#!/usr/bin/env python3
"""
PreToolUse Hook — Destructive Command Blocker

Blocks dangerous commands BEFORE execution via exit code 2.
Claude receives stderr feedback and self-corrects.

Triggered by: PreToolUse with matcher "Bash"
Location: .claude/settings.json (Project)
Path: Direct execution (standalone, NOT through context_guard.py)

P1 Hallucination Prevention: Destructive command detection is deterministic
(regex-based). No AI judgment needed — 100% accurate for defined patterns.

Blocked patterns (from Claude Code safety guidelines):
  - git push --force (NOT --force-with-lease or --force-if-includes)
  - git push -f (short flag, including combined forms like -fu)
  - git reset --hard
  - git checkout . (discards ALL unstaged changes)
  - git restore . (discards ALL changes)
  - git clean -f (removes untracked files)
  - git branch -D or --delete --force (force-deletes branch)
  - rm -rf / or rm -rf ~ (catastrophic file deletion)
  - cat .env / printenv / env | / echo $SECRET_VAR (secret exposure)
  - DROP TABLE / DROP DATABASE / TRUNCATE / DELETE without WHERE (destructive SQL)

Known limitations:
  - Commands in string literals may cause false positives
    (e.g., echo "git push --force" would be blocked).
    Acceptable: false positive > false negative for safety hooks.
  - Patterns check the raw command string, not parsed shell AST.
  - SQL in string literals may cause false positives for SQL patterns.

Safety-first: Any unexpected internal error → exit(0) (never block Claude).

ADR-031 in DECISION-LOG.md
"""

import json
import re
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Destructive Git patterns
# Each: (compiled_regex, stderr message for Claude self-correction)
#
# Regex notes:
#   - \s before -- flags (not \b) because \b fails between space and dash
#   - (?![-\w]) after --force to exclude --force-with-lease, --force-if-includes
#   - \s-[a-zA-Z]*f for combined short flags (-f, -uf, -fu all match)
# ---------------------------------------------------------------------------
GIT_PATTERNS = [
    # git push --force (NOT --force-with-lease or --force-if-includes)
    (
        re.compile(r"\bgit\s+push\b.*\s--force(?![-\w])"),
        "git push --force is blocked. "
        "Use --force-with-lease for safer force pushing.",
    ),
    # git push -f (short flag, including combined forms like -fu, -uf)
    (
        re.compile(r"\bgit\s+push\b.*\s-[a-zA-Z]*f"),
        "git push -f is blocked. "
        "Use --force-with-lease for safer force pushing.",
    ),
    # git reset --hard
    (
        re.compile(r"\bgit\s+reset\b.*\s--hard(?![-\w])"),
        "git reset --hard is blocked. "
        "Discards uncommitted changes irreversibly. "
        "Use git stash or git reset --soft instead.",
    ),
    # git checkout . (discard ALL unstaged changes)
    (
        re.compile(r"\bgit\s+checkout\b\s+(?:--\s+)?\.(?:\s|$)"),
        "git checkout . is blocked. "
        "Discards all unstaged changes. "
        "Use git stash to preserve changes first.",
    ),
    # git restore . (discard ALL changes, with or without --staged)
    (
        re.compile(r"\bgit\s+restore\b(?:\s+--[\w-]+)*\s+\.(?:\s|$)"),
        "git restore . is blocked. "
        "Discards all changes. "
        "Use git stash to preserve changes first.",
    ),
    # git clean -f (remove untracked files, any combined flag with f)
    (
        re.compile(r"\bgit\s+clean\b.*\s-[a-zA-Z]*f"),
        "git clean -f is blocked. "
        "Permanently removes untracked files. "
        "Use git clean -n (dry run) to preview first.",
    ),
    # git branch -D (force delete, unlike safe -d)
    (
        re.compile(r"\bgit\s+branch\b.*\s-D"),
        "git branch -D is blocked. "
        "Force-deletes branch even if not fully merged. "
        "Use git branch -d for safe deletion.",
    ),
    # git branch --delete --force (long form of -D, any order)
    (
        re.compile(r"\bgit\s+branch\b.*\s--delete\b.*\s--force\b"),
        "git branch --delete --force is blocked. "
        "Force-deletes branch even if not fully merged. "
        "Use git branch -d for safe deletion.",
    ),
    (
        re.compile(r"\bgit\s+branch\b.*\s--force\b.*\s--delete\b"),
        "git branch --force --delete is blocked. "
        "Force-deletes branch even if not fully merged. "
        "Use git branch -d for safe deletion.",
    ),
]

# ---------------------------------------------------------------------------
# Secret-exposing command patterns (PreToolUse — input-based detection)
# ---------------------------------------------------------------------------
SECRET_SOURCE_PATTERNS = [
    (
        re.compile(r"\bcat\s+[^\s]*\.env\b"),
        "cat .env is blocked. Secrets may leak to output.",
    ),
    (
        re.compile(r"\bprintenv\b"),
        "printenv is blocked. Environment variables may contain secrets.",
    ),
    (
        re.compile(r"\benv\s*\|"),
        "env piped output is blocked. May expose secrets.",
    ),
    (
        re.compile(
            r"\becho\s+\$[A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)",
            re.I,
        ),
        "Echoing secret environment variable is blocked.",
    ),
]

# ---------------------------------------------------------------------------
# Destructive SQL patterns
# ---------------------------------------------------------------------------
SQL_PATTERNS = [
    (
        re.compile(r"\bDROP\s+TABLE\b", re.I),
        "DROP TABLE is blocked. Irreversible data loss.",
    ),
    (
        re.compile(r"\bDROP\s+DATABASE\b", re.I),
        "DROP DATABASE is blocked. Irreversible.",
    ),
    (
        re.compile(r"\bTRUNCATE\s+", re.I),
        "TRUNCATE is blocked. Deletes all rows without logging.",
    ),
    (
        re.compile(r"\bALTER\s+TABLE\b.*\bDROP\b", re.I),
        "ALTER TABLE DROP is blocked.",
    ),
]


def _check_dangerous_sql(sub_command: str) -> Optional[str]:
    """Check for destructive SQL in sub-command (already split by shell operators).

    DELETE FROM is only blocked when no WHERE clause follows
    within the same SQL statement (semicolon-delimited).

    Known limitation: SQL in string literals may cause false positives.
    Acceptable: false positive > false negative (consistent with GIT_PATTERNS philosophy).
    """
    # Check non-statement-level patterns first (these apply to entire sub-command)
    for pattern, message in SQL_PATTERNS:
        if pattern.search(sub_command):
            return message
    # DELETE FROM: split by SQL statement terminator for precise WHERE detection
    for sql_stmt in re.split(r";", sub_command):
        if re.search(r"\bDELETE\s+FROM\b", sql_stmt, re.I):
            if not re.search(r"\bWHERE\b", sql_stmt, re.I):
                return (
                    "DELETE without WHERE is blocked. "
                    "Add WHERE clause for targeted deletion."
                )
    return None


def _check_dangerous_rm(sub_command: str) -> Optional[str]:
    """Check if an rm sub-command targets root or home with recursive+force.

    Parses flags and targets separately to handle all flag orderings:
    rm -rf /, rm -fr /, rm -r -f /, etc.
    C4 fix: also handles sudo prefix (sudo rm -rf /).
    """
    tokens = sub_command.split()
    if not tokens:
        return None
    # Skip sudo prefix(es) to find the actual rm command
    idx = 0
    while idx < len(tokens) and tokens[idx] == "sudo":
        idx += 1
    if idx >= len(tokens) or tokens[idx] != "rm":
        return None
    tokens = tokens[idx:]  # Re-base to rm as first token

    # Collect all single-dash flag characters and targets
    flags = ""
    targets = []
    for token in tokens[1:]:
        if token.startswith("-") and not token.startswith("--"):
            flags += token[1:]  # strip leading dash
        elif not token.startswith("-"):
            targets.append(token.strip("\"'"))

    has_recursive = "r" in flags or "R" in flags
    has_force = "f" in flags

    if not (has_recursive and has_force):
        return None

    # Catastrophic targets only — specific paths, not general directories
    dangerous = {"/", "/*", "~", "~/", "$HOME", "$HOME/", "$HOME/*"}
    for target in targets:
        if target in dangerous:
            return (
                f"rm -rf targeting {target} is blocked. "
                "Catastrophic, irreversible file deletion."
            )
    return None


def check_command(command: str) -> Optional[str]:
    """Check command against all destructive patterns.

    Returns block message if pattern matches, None otherwise.

    Check order:
      (1) Git patterns — entire command string
      (2) Secret source patterns — entire command string
      (3) Sub-command level — SQL patterns + rm patterns (shell operator split)
    """
    # (1) Git patterns: check entire command string (regex handles flag positions)
    for pattern, message in GIT_PATTERNS:
        if pattern.search(command):
            return message

    # (2) Secret source patterns: check entire command string
    for pattern, message in SECRET_SOURCE_PATTERNS:
        if pattern.search(command):
            return message

    # (3) Sub-command level checks (shell operator split)
    for sub_cmd in re.split(r"\s*(?:&&|\|\||;)\s*", command):
        for segment in sub_cmd.split("|"):
            segment = segment.strip()
            # SQL patterns
            result = _check_dangerous_sql(segment)
            if result:
                return result
            # rm patterns (existing)
            result = _check_dangerous_rm(segment)
            if result:
                return result

    return None


def main():
    """Read PreToolUse JSON from stdin, check for destructive commands."""
    # Read Hook JSON payload from stdin
    # Format: {"tool_name": "Bash", "tool_input": {"command": "..."}}
    try:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            sys.exit(0)

        payload = json.loads(stdin_data)
        command = payload.get("tool_input", {}).get("command", "")

        if not command:
            sys.exit(0)
    except (json.JSONDecodeError, KeyError, TypeError):
        # Malformed input — don't block, exit cleanly
        sys.exit(0)

    # Check against destructive patterns
    block_message = check_command(command)

    if block_message:
        # Exit code 2 = Claude Hook blocking signal
        # stderr content is sent to Claude for self-correction
        print(
            f"DESTRUCTIVE COMMAND BLOCKED: {block_message}\n"
            f"Command was: {command[:200]}",
            file=sys.stderr,
        )
        sys.exit(2)

    # No match — allow command to proceed
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Safety-first: never block Claude on unexpected internal errors
        sys.exit(0)
