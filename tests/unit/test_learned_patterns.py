"""P1 Layer 3 Tests — Learned Patterns (extract_learned_patterns in _context_lib.py)

Tests for Improvement 1: P1 Consumer function that aggregates success_patterns
from knowledge-index.jsonl for SessionStart surfacing.
"""

import importlib.util
import json
import os
import sys

import pytest

# Import _context_lib module
HOOKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".claude", "hooks", "scripts",
)

if HOOKS_DIR not in sys.path:
    sys.path.insert(0, HOOKS_DIR)

from _context_lib import extract_learned_patterns


def _write_ki(tmp_path, entries):
    """Write entries to a knowledge-index.jsonl file."""
    ki_path = tmp_path / "knowledge-index.jsonl"
    with open(ki_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(ki_path)


class TestExtractLearnedPatterns:
    """P1 Consumer: extract_learned_patterns() tests."""

    def test_empty_ki_returns_empty(self, tmp_path):
        ki_path = _write_ki(tmp_path, [])
        result = extract_learned_patterns(ki_path)
        assert result == []

    def test_nonexistent_file_returns_empty(self):
        result = extract_learned_patterns("/nonexistent/path.jsonl")
        assert result == []

    def test_single_session_below_threshold(self, tmp_path):
        """One session with a pattern should not meet min_sessions=3."""
        entries = [
            {
                "session_id": "s1",
                "success_patterns": [{"sequence": "Edit→Bash", "files": ["a.py"]}],
            }
        ]
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path)
        assert result == []

    def test_three_sessions_extracts_pattern(self, tmp_path):
        """Pattern appearing in 3 sessions should be extracted."""
        entries = [
            {
                "session_id": f"s{i}",
                "success_patterns": [{"sequence": "Edit→Bash", "files": ["a.py"]}],
            }
            for i in range(3)
        ]
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path)
        assert len(result) == 1
        seq, count, conf, files = result[0]
        assert count == 3
        assert "edit→bash" in seq  # Normalized to lowercase

    def test_confidence_calculation(self, tmp_path):
        """confidence = min(1.0, session_count / 5)"""
        entries = [
            {
                "session_id": f"s{i}",
                "success_patterns": [{"sequence": "Edit→Bash"}],
            }
            for i in range(4)
        ]
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path)
        assert len(result) == 1
        _, count, conf, _ = result[0]
        assert count == 4
        assert abs(conf - 0.8) < 0.01  # 4/5 = 0.8

    def test_confidence_capped_at_1(self, tmp_path):
        """confidence should be capped at 1.0 for count >= 5."""
        entries = [
            {
                "session_id": f"s{i}",
                "success_patterns": [{"sequence": "Edit→Bash"}],
            }
            for i in range(7)
        ]
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path)
        _, count, conf, _ = result[0]
        assert count == 7
        assert conf == 1.0

    def test_max_results_limit(self, tmp_path):
        """Should return at most max_results patterns."""
        entries = []
        for i in range(10):
            pats = [{"sequence": f"Pattern-{j}"} for j in range(4)]
            entries.append({"session_id": f"s{i}", "success_patterns": pats})
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path, min_sessions=3, max_results=2)
        assert len(result) <= 2

    def test_session_dedup(self, tmp_path):
        """Same pattern in same session counted only once."""
        entries = [
            {
                "session_id": "s1",
                "success_patterns": [
                    {"sequence": "Edit→Bash"},
                    {"sequence": "Edit→Bash"},  # Duplicate in same session
                    {"sequence": "Edit→Bash"},
                ],
            },
            {
                "session_id": "s2",
                "success_patterns": [{"sequence": "Edit→Bash"}],
            },
            {
                "session_id": "s3",
                "success_patterns": [{"sequence": "Edit→Bash"}],
            },
        ]
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path)
        assert len(result) == 1
        _, count, _, _ = result[0]
        assert count == 3  # 3 sessions, not 5 occurrences

    def test_malformed_json_skipped(self, tmp_path):
        """Malformed JSON lines should be skipped gracefully."""
        ki_path = tmp_path / "knowledge-index.jsonl"
        with open(ki_path, "w") as f:
            f.write("not valid json\n")
            for i in range(3):
                entry = {
                    "session_id": f"s{i}",
                    "success_patterns": [{"sequence": "Edit→Bash"}],
                }
                f.write(json.dumps(entry) + "\n")
        result = extract_learned_patterns(str(ki_path))
        assert len(result) == 1

    def test_missing_success_patterns_field(self, tmp_path):
        """Entries without success_patterns should be skipped."""
        entries = [
            {"session_id": "s1"},  # No success_patterns
            {
                "session_id": "s2",
                "success_patterns": [{"sequence": "Edit→Bash"}],
            },
        ]
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path, min_sessions=1)
        assert len(result) == 1
        _, count, _, _ = result[0]
        assert count == 1

    def test_sorting_by_count(self, tmp_path):
        """Results should be sorted by session_count descending."""
        entries = []
        # Pattern A: 5 sessions, Pattern B: 3 sessions
        for i in range(5):
            entries.append({
                "session_id": f"sa{i}",
                "success_patterns": [{"sequence": "PatternA"}],
            })
        for i in range(3):
            entries.append({
                "session_id": f"sb{i}",
                "success_patterns": [{"sequence": "PatternB"}],
            })
        ki_path = _write_ki(tmp_path, entries)
        result = extract_learned_patterns(ki_path)
        assert len(result) == 2
        assert result[0][1] > result[1][1]  # First has higher count
