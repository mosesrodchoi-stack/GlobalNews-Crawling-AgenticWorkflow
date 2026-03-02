#!/usr/bin/env python3
"""
PostToolUse Hook — Secret Output Leak Detector

Detects secret patterns in Bash command output and warns via stderr.
Does NOT block (PostToolUse cannot block — exit code always 0).

Triggered by: PostToolUse with matcher "Bash"
Location: .claude/settings.json (Project)
Path: Direct execution (standalone, NOT through context_guard.py)

P1 Hallucination Prevention: Secret pattern detection is deterministic
(regex-based). No AI judgment needed — 100% accurate for defined patterns.

Best-effort: tool_response structure is not guaranteed by Claude Code API.
Multiple extraction paths are tried (stdout, stderr, content, output, raw string).

Known limitations:
  - tool_response schema may change across Claude Code versions
  - If no content is extractable, silently exits (no false positives)
  - Patterns may match non-secret strings (e.g., "Bearer test-token-for-demo")
    Acceptable: false positive warning > missed leak for security hooks.

Safety-first: Any unexpected internal error → exit(0) (never fail the hook).
"""

import json
import re
import sys

# ---------------------------------------------------------------------------
# Secret output patterns (PostToolUse — output-based detection)
# Each: (compiled_regex, human-readable label)
# ---------------------------------------------------------------------------
SECRET_OUTPUT_PATTERNS = [
    (
        re.compile(r"(?:sk|pk)[-_](?:live|test|prod)[-_][a-zA-Z0-9]{20,}"),
        "API key",
    ),
    (
        re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}"),
        "GitHub token",
    ),
    (
        re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
        "AWS access key",
    ),
    (
        re.compile(r"Bearer\s+[a-zA-Z0-9._\-]{20,}"),
        "Bearer token",
    ),
    (
        re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
        "Private key",
    ),
    (
        re.compile(r"password\s*[=:]\s*[\"'][^\"']{8,}[\"']", re.I),
        "Hardcoded password",
    ),
]


def check_output(content: str) -> list:
    """Check output content against secret patterns.

    Returns list of matched labels. Empty list if no match.
    P1: Deterministic regex matching.
    """
    matched = []
    for pattern, label in SECRET_OUTPUT_PATTERNS:
        if pattern.search(content):
            matched.append(label)
    return matched


def main():
    """Read PostToolUse JSON from stdin, check for secret leaks in output."""
    try:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            sys.exit(0)

        payload = json.loads(stdin_data)
        tool_response = payload.get("tool_response", {})

        # Best-effort: tool_response structure is not guaranteed
        # Try multiple extraction paths, fallback to json.dumps
        content = ""
        if isinstance(tool_response, dict):
            content = tool_response.get("stdout", "")
            content += tool_response.get("stderr", "")
            content += tool_response.get("content", "")
            content += str(tool_response.get("output", ""))
            # C3 fallback: if all specific paths yielded nothing,
            # serialize the entire dict to catch secrets in unknown fields
            if not content.strip() or content.strip() == "None":
                content = json.dumps(tool_response, default=str)
        elif isinstance(tool_response, str):
            content = tool_response

        if not content:
            sys.exit(0)

        matched = check_output(content)
        if matched:
            labels = ", ".join(matched)
            print(
                f"SECRET DETECTED: {labels} found in command output. "
                f"Consider using environment variables or .env files.",
                file=sys.stderr,
            )

    except (json.JSONDecodeError, KeyError, TypeError):
        pass  # Malformed input — don't warn, exit cleanly
    except Exception:
        pass  # Safety-first: never fail the hook

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Safety-first: never fail the hook
        sys.exit(0)
