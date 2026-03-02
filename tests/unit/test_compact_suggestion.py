"""P1 Layer 3 Tests — Phase-Aware Compact Suggestion (defect #13)

Tests for Improvement 4: _suggest_compact_if_needed() in generate_context_summary.py.
Validates P1 deterministic conditions for compact suggestion.

C1 fix: Session-scoped marker (session_id|timestamp format).
C2 fix: detect_phase_transitions returns 3-tuples (phase, start_idx, end_idx).
"""

import importlib.util
import os
import sys
from unittest.mock import patch

import pytest

# Import generate_context_summary module
HOOKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".claude", "hooks", "scripts",
)

if HOOKS_DIR not in sys.path:
    sys.path.insert(0, HOOKS_DIR)


def _import_gcs():
    spec = importlib.util.spec_from_file_location(
        "generate_context_summary",
        os.path.join(HOOKS_DIR, "generate_context_summary.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gcs = _import_gcs()

SESSION_ID = "test-session-abc"


def _make_tool_entries(n, tool_name="Edit"):
    """Create n tool_use entries."""
    return [{"type": "tool_use", "tool_name": tool_name} for _ in range(n)]


class TestSuggestCompactIfNeeded:
    """P1 deterministic compact suggestion tests."""

    def test_no_suggestion_when_marker_exists_same_session(self, tmp_path, capsys):
        """Dedup: marker file from SAME session prevents repeated suggestions."""
        marker = tmp_path / ".last_compact_suggestion"
        marker.write_text(f"{SESSION_ID}|2025-01-01T00:00:00")

        entries = _make_tool_entries(60)
        with patch.object(gcs, "estimate_tokens", return_value=(150000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("research", 0, 30), ("implementation", 30, 60)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" not in captured.err

    def test_stale_marker_from_different_session(self, tmp_path, capsys):
        """C1 fix: marker from DIFFERENT session should NOT block suggestion."""
        marker = tmp_path / ".last_compact_suggestion"
        marker.write_text("old-session-xyz|2025-01-01T00:00:00")

        entries = _make_tool_entries(60)
        with patch.object(gcs, "estimate_tokens", return_value=(150000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("research", 0, 30), ("implementation", 30, 60)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" in captured.err

    def test_suggestion_on_phase_transition_50pct(self, tmp_path, capsys):
        """Condition 1: Phase transition + 50%+ tokens -> suggest."""
        entries = _make_tool_entries(40)
        with patch.object(gcs, "estimate_tokens", return_value=(110000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("research", 0, 20), ("implementation", 20, 40)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" in captured.err
        assert "Phase transition" in captured.err
        # Verify marker file was created with session_id
        marker = tmp_path / ".last_compact_suggestion"
        assert marker.exists()
        assert marker.read_text().startswith(SESSION_ID + "|")

    def test_no_suggestion_below_50pct(self, tmp_path, capsys):
        """Condition 1 boundary: tokens < 50% -> no suggestion despite phase transition."""
        entries = _make_tool_entries(40)
        with patch.object(gcs, "estimate_tokens", return_value=(90000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("research", 0, 20), ("implementation", 20, 40)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" not in captured.err

    def test_suggestion_on_65pct_50_tools(self, tmp_path, capsys):
        """Condition 2: 65%+ tokens + 50+ tools -> suggest."""
        entries = _make_tool_entries(55)
        with patch.object(gcs, "estimate_tokens", return_value=(140000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("implementation", 0, 55)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" in captured.err
        assert "Tools 55" in captured.err

    def test_no_suggestion_below_65pct(self, tmp_path, capsys):
        """Condition 2 boundary: tokens < 65% -> no suggestion despite 50+ tools."""
        entries = _make_tool_entries(55)
        with patch.object(gcs, "estimate_tokens", return_value=(120000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("implementation", 0, 55)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" not in captured.err

    def test_no_suggestion_below_50_tools(self, tmp_path, capsys):
        """Condition 2 boundary: < 50 tools -> no suggestion despite high tokens."""
        entries = _make_tool_entries(45)
        with patch.object(gcs, "estimate_tokens", return_value=(140000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("implementation", 0, 45)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" not in captured.err

    def test_marker_file_created_after_suggestion(self, tmp_path):
        """Marker file must be created with session_id after a suggestion."""
        marker_path = tmp_path / ".last_compact_suggestion"
        assert not marker_path.exists()

        entries = _make_tool_entries(55)
        with patch.object(gcs, "estimate_tokens", return_value=(140000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("implementation", 0, 55)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        assert marker_path.exists()
        content = marker_path.read_text()
        parts = content.split("|")
        assert parts[0] == SESSION_ID
        assert len(parts) == 2  # session_id|timestamp

    def test_single_phase_no_suggestion(self, tmp_path, capsys):
        """No phase transition -> Condition 1 cannot trigger."""
        entries = _make_tool_entries(30)
        with patch.object(gcs, "estimate_tokens", return_value=(110000, None)):
            with patch.object(gcs, "EFFECTIVE_CAPACITY", 200000):
                with patch.object(gcs, "detect_phase_transitions", return_value=[
                    ("implementation", 0, 30)
                ]):
                    gcs._suggest_compact_if_needed(
                        entries, "/dummy", str(tmp_path), SESSION_ID
                    )

        captured = capsys.readouterr()
        assert "COMPACT SUGGESTION" not in captured.err
