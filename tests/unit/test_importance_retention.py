"""Unit tests for importance-based retention in _context_lib.py.

Tests cover:
  - _importance_tier(): deterministic tier assignment (Tier 0-3)
  - validate_retention_result(): P1 invariant validation (RI1, RI2, RI4, RI5, RI6)
  - cleanup_knowledge_index(): integration with importance-based selection + fail-safe
"""

import importlib.util
import json
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Module import — _context_lib.py is in .claude/hooks/scripts/
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOKS_DIR = os.path.join(PROJECT_ROOT, ".claude", "hooks", "scripts")


def _import_context_lib():
    """Import _context_lib module from hooks/scripts/."""
    spec = importlib.util.spec_from_file_location(
        "_context_lib", os.path.join(HOOKS_DIR, "_context_lib.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ctx():
    """Module-scoped _context_lib import (heavy module, load once)."""
    return _import_context_lib()


# ==========================================================================
# _importance_tier() tests
# ==========================================================================


class TestImportanceTier:
    """Deterministic tier assignment: pure function, no side effects."""

    # --- Tier 3: high-value cross-session learning assets ---

    def test_tier3_design_decisions(self, ctx):
        entry = {"design_decisions": ["[explicit] chose X over Y"]}
        assert ctx._importance_tier(entry) == 3

    def test_tier3_design_decisions_multiple(self, ctx):
        entry = {"design_decisions": ["[decision] use FIFO", "[rationale] simpler"]}
        assert ctx._importance_tier(entry) == 3

    def test_tier3_error_with_resolution(self, ctx):
        entry = {
            "error_patterns": [
                {"type": "syntax", "resolution": {"tool": "Edit", "file": "a.py"}}
            ]
        }
        assert ctx._importance_tier(entry) == 3

    def test_tier3_error_mixed_some_resolved(self, ctx):
        """If ANY error has resolution, entry is Tier 3."""
        entry = {
            "error_patterns": [
                {"type": "syntax", "resolution": None},
                {"type": "timeout", "resolution": {"tool": "Bash"}},
            ]
        }
        assert ctx._importance_tier(entry) == 3

    def test_tier3_diagnosis_patterns(self, ctx):
        entry = {"diagnosis_patterns": [{"gate": "verification", "step": 5}]}
        assert ctx._importance_tier(entry) == 3

    # --- Tier 2: workflow coordination context ---

    def test_tier2_team_summaries(self, ctx):
        entry = {"team_summaries": {"task-1": {"summary": "done"}}}
        assert ctx._importance_tier(entry) == 2

    def test_tier2_ulw_active(self, ctx):
        entry = {"ulw_active": True}
        assert ctx._importance_tier(entry) == 2

    def test_tier2_pacs_min(self, ctx):
        entry = {"pacs_min": 72}
        assert ctx._importance_tier(entry) == 2

    def test_tier2_pacs_min_zero(self, ctx):
        """pacs_min=0 is a valid score, should still be Tier 2."""
        entry = {"pacs_min": 0}
        assert ctx._importance_tier(entry) == 2

    # --- Tier 1: session had code changes ---

    def test_tier1_modified_files(self, ctx):
        entry = {"modified_files": ["src/main.py"]}
        assert ctx._importance_tier(entry) == 1

    # --- Tier 0: trivial ---

    def test_tier0_empty_dict(self, ctx):
        assert ctx._importance_tier({}) == 0

    def test_tier0_read_only_session(self, ctx):
        """Session with read_files but no edits."""
        entry = {"read_files": ["README.md"], "trigger": "stop"}
        assert ctx._importance_tier(entry) == 0

    def test_tier0_non_dict(self, ctx):
        assert ctx._importance_tier("not a dict") == 0

    def test_tier0_none(self, ctx):
        assert ctx._importance_tier(None) == 0

    def test_tier0_integer(self, ctx):
        assert ctx._importance_tier(42) == 0

    def test_tier0_error_without_resolution(self, ctx):
        """Errors without resolution are NOT high-value (no learning)."""
        entry = {"error_patterns": [{"type": "syntax", "resolution": None}]}
        assert ctx._importance_tier(entry) == 0

    def test_tier0_empty_design_decisions(self, ctx):
        """Empty list should not be Tier 3."""
        entry = {"design_decisions": []}
        assert ctx._importance_tier(entry) == 0

    # --- Priority order: Tier 3 > Tier 2 > Tier 1 ---

    def test_priority_tier3_over_tier2(self, ctx):
        """Entry with both Tier 3 and Tier 2 signals → Tier 3."""
        entry = {
            "design_decisions": ["[explicit] chose X"],
            "ulw_active": True,
            "modified_files": ["a.py"],
        }
        assert ctx._importance_tier(entry) == 3

    def test_priority_tier2_over_tier1(self, ctx):
        """Entry with both Tier 2 and Tier 1 signals → Tier 2."""
        entry = {"ulw_active": True, "modified_files": ["a.py"]}
        assert ctx._importance_tier(entry) == 2


# ==========================================================================
# validate_retention_result() tests
# ==========================================================================


class TestValidateRetentionResult:
    """P1 invariant validation: RI1, RI2, RI4, RI5, RI6."""

    def _make_line(self, session_id, timestamp, **extra):
        """Helper: create a JSONL line."""
        entry = {"session_id": session_id, "timestamp": timestamp, **extra}
        return json.dumps(entry, ensure_ascii=False) + "\n"

    # --- RI1: Non-empty ---

    def test_ri1_empty_list(self, ctx):
        w = ctx.validate_retention_result([], 10)
        assert any("RI1" in x for x in w)

    def test_ri1_non_empty(self, ctx):
        w = ctx.validate_retention_result([self._make_line("a", "2026-01-01")], 1)
        assert not any("RI1" in x for x in w)

    # --- RI2: Count within budget ---

    def test_ri2_over_budget(self, ctx):
        lines = [self._make_line(f"s{i}", "2026-01-01") for i in range(201)]
        w = ctx.validate_retention_result(lines, 250)
        assert any("RI2" in x for x in w)

    def test_ri2_at_budget(self, ctx):
        lines = [self._make_line(f"s{i}", "2026-01-01") for i in range(200)]
        w = ctx.validate_retention_result(lines, 250)
        assert not any("RI2" in x for x in w)

    # --- RI4: Date-level chronological order ---

    def test_ri4_correct_order(self, ctx):
        lines = [
            self._make_line("a", "2026-02-16T10:00:00"),
            self._make_line("b", "2026-02-17T10:00:00"),
        ]
        w = ctx.validate_retention_result(lines, 2)
        assert not any("RI4" in x for x in w)

    def test_ri4_same_day_reorder_tolerated(self, ctx):
        """Same-day timestamp reordering should NOT trigger RI4."""
        lines = [
            self._make_line("a", "2026-02-16T12:19:07"),
            self._make_line("b", "2026-02-16T10:37:44"),  # earlier time, same day
        ]
        w = ctx.validate_retention_result(lines, 2)
        assert not any("RI4" in x for x in w)

    def test_ri4_cross_day_reversal_detected(self, ctx):
        """Cross-day reversal MUST trigger RI4."""
        lines = [
            self._make_line("a", "2026-02-17T10:00:00"),
            self._make_line("b", "2026-02-16T10:00:00"),  # previous day
        ]
        w = ctx.validate_retention_result(lines, 2)
        assert any("RI4" in x for x in w)

    def test_ri4_missing_timestamps_skipped(self, ctx):
        """Entries without timestamps should not cause violations."""
        lines = [
            self._make_line("a", "2026-02-16T10:00:00"),
            json.dumps({"session_id": "b"}) + "\n",  # no timestamp
            self._make_line("c", "2026-02-17T10:00:00"),
        ]
        w = ctx.validate_retention_result(lines, 3)
        assert not any("RI4" in x for x in w)

    # --- RI5: No duplicate session_ids ---

    def test_ri5_no_duplicates(self, ctx):
        lines = [
            self._make_line("a", "2026-01-01"),
            self._make_line("b", "2026-01-02"),
        ]
        w = ctx.validate_retention_result(lines, 2)
        assert not any("RI5" in x for x in w)

    def test_ri5_duplicates_detected(self, ctx):
        lines = [
            self._make_line("same", "2026-01-01"),
            self._make_line("same", "2026-01-02"),
        ]
        w = ctx.validate_retention_result(lines, 2)
        assert any("RI5" in x for x in w)

    # --- RI6: Exact budget fill ---

    def test_ri6_exact_fill(self, ctx):
        """When input > MAX, output must be exactly MAX."""
        lines = [self._make_line(f"s{i}", "2026-01-01") for i in range(200)]
        w = ctx.validate_retention_result(lines, 250)
        assert not any("RI6" in x for x in w)

    def test_ri6_under_fill(self, ctx):
        """Fewer than MAX when input > MAX → RI6 fires."""
        lines = [self._make_line(f"s{i}", "2026-01-01") for i in range(190)]
        w = ctx.validate_retention_result(lines, 250)
        assert any("RI6" in x for x in w)

    def test_ri6_not_applicable_below_max(self, ctx):
        """RI6 does NOT fire when input <= MAX."""
        lines = [self._make_line(f"s{i}", "2026-01-01") for i in range(50)]
        w = ctx.validate_retention_result(lines, 50)
        assert not any("RI6" in x for x in w)

    # --- All pass ---

    def test_all_pass(self, ctx):
        lines = [
            self._make_line(
                f"s{i}",
                f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T10:00:00",
            )
            for i in range(200)
        ]
        w = ctx.validate_retention_result(lines, 250)
        assert w == [], f"Expected no warnings, got: {w}"


# ==========================================================================
# cleanup_knowledge_index() integration tests
# ==========================================================================


class TestCleanupKnowledgeIndex:
    """Integration: importance-based selection + re-sort + fail-safe."""

    def _write_ki(self, tmp_path, entries):
        """Helper: write entries to a temp knowledge-index.jsonl."""
        snapshot_dir = str(tmp_path / "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        ki_path = os.path.join(snapshot_dir, "knowledge-index.jsonl")
        with open(ki_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return snapshot_dir, ki_path

    def _read_ki(self, ki_path):
        """Helper: read entries from knowledge-index.jsonl."""
        entries = []
        with open(ki_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries

    def test_below_max_is_noop(self, ctx, tmp_path):
        """When entries <= MAX, file should be unchanged."""
        entries = [
            {"session_id": f"s{i}", "timestamp": f"2026-01-{i+1:02d}T10:00:00"}
            for i in range(50)
        ]
        snapshot_dir, ki_path = self._write_ki(tmp_path, entries)
        ctx.cleanup_knowledge_index(snapshot_dir)
        result = self._read_ki(ki_path)
        assert len(result) == 50

    def test_tier3_survives_over_tier0(self, ctx, tmp_path):
        """Tier 3 entries survive while Tier 0 entries are dropped."""
        entries = []
        for i in range(210):
            e = {
                "session_id": f"s{i:04d}",
                "timestamp": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T10:00:00",
            }
            if i % 7 == 0:
                e["design_decisions"] = [f"[explicit] decision {i}"]  # Tier 3
            entries.append(e)

        snapshot_dir, ki_path = self._write_ki(tmp_path, entries)
        ctx.cleanup_knowledge_index(snapshot_dir)
        result = self._read_ki(ki_path)

        assert len(result) == 200

        # All Tier 3 entries should survive
        tier3_count = sum(1 for e in result if ctx._importance_tier(e) == 3)
        tier3_input = sum(1 for e in entries if ctx._importance_tier(e) == 3)
        assert tier3_count == tier3_input, (
            f"Tier 3: {tier3_count}/{tier3_input} survived"
        )

    def test_chronological_order_preserved(self, ctx, tmp_path):
        """After rotation, entries must be in chronological order."""
        entries = []
        for i in range(210):
            e = {
                "session_id": f"s{i:04d}",
                "timestamp": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T10:00:00",
            }
            if i % 5 == 0:
                e["design_decisions"] = ["[explicit] test"]
            entries.append(e)

        snapshot_dir, ki_path = self._write_ki(tmp_path, entries)
        ctx.cleanup_knowledge_index(snapshot_dir)
        result = self._read_ki(ki_path)

        # Verify date-level order
        dates = [e.get("timestamp", "")[:10] for e in result if e.get("timestamp")]
        for i in range(1, len(dates)):
            assert dates[i] >= dates[i - 1], (
                f"Date order violated at {i}: {dates[i-1]} > {dates[i]}"
            )

    def test_nonexistent_file_is_noop(self, ctx, tmp_path):
        """cleanup on nonexistent directory should not crash."""
        ctx.cleanup_knowledge_index(str(tmp_path / "nonexistent"))
        # No exception = pass

    def test_malformed_json_gets_tier0(self, ctx, tmp_path):
        """Malformed JSON lines should be treated as Tier 0."""
        snapshot_dir = str(tmp_path / "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        ki_path = os.path.join(snapshot_dir, "knowledge-index.jsonl")

        with open(ki_path, "w") as f:
            # 195 valid Tier 3 entries
            for i in range(195):
                e = {
                    "session_id": f"s{i:04d}",
                    "timestamp": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T10:00:00",
                    "design_decisions": ["[explicit] test"],
                }
                f.write(json.dumps(e) + "\n")
            # 10 malformed lines
            for i in range(10):
                f.write(f"{{invalid json line {i}\n")

        ctx.cleanup_knowledge_index(snapshot_dir)
        result_lines = []
        with open(ki_path, "r") as f:
            result_lines = [l for l in f if l.strip()]

        assert len(result_lines) == 200
        # All 195 Tier 3 survive, 5 of 10 malformed (Tier 0) survive
        valid_count = 0
        for line in result_lines:
            try:
                json.loads(line)
                valid_count += 1
            except json.JSONDecodeError:
                pass
        assert valid_count == 195


# ==========================================================================
# extract_recurring_error_types() tests
# ==========================================================================


class TestExtractRecurringErrorTypes:
    """P1 Consumer: deterministic error type aggregation from knowledge-index."""

    def test_no_file(self, ctx, tmp_path):
        """Returns empty list when KI file doesn't exist."""
        result = ctx.extract_recurring_error_types(str(tmp_path / "nonexistent.jsonl"))
        assert result == []

    def test_empty_file(self, ctx, tmp_path):
        """Returns empty list for empty KI file."""
        ki = tmp_path / "ki.jsonl"
        ki.write_text("")
        assert ctx.extract_recurring_error_types(str(ki)) == []

    def test_no_errors(self, ctx, tmp_path):
        """Returns empty list when no sessions have error_patterns."""
        ki = tmp_path / "ki.jsonl"
        entries = [{"session_id": f"s{i}", "tags": ["python"]} for i in range(5)]
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        assert ctx.extract_recurring_error_types(str(ki)) == []

    def test_below_threshold(self, ctx, tmp_path):
        """Returns empty list when error types appear in fewer than min_count sessions."""
        ki = tmp_path / "ki.jsonl"
        entries = [
            {"session_id": "s1", "error_patterns": [{"type": "timeout", "tool": "Bash"}]},
            {"session_id": "s2", "error_patterns": [{"type": "timeout", "tool": "Bash"}]},
        ]
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        assert ctx.extract_recurring_error_types(str(ki), min_count=3) == []

    def test_recurring_detected(self, ctx, tmp_path):
        """Detects error types recurring across 3+ sessions."""
        ki = tmp_path / "ki.jsonl"
        entries = [
            {"session_id": f"s{i}", "error_patterns": [{"type": "timeout", "tool": "Bash"}]}
            for i in range(5)
        ]
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = ctx.extract_recurring_error_types(str(ki))
        assert result == [("timeout", 5)]

    def test_counts_sessions_not_occurrences(self, ctx, tmp_path):
        """Counts each error type once per session, not per occurrence."""
        ki = tmp_path / "ki.jsonl"
        # Session with 3 timeout errors — should count as 1 session
        entries = [
            {"session_id": "s1", "error_patterns": [
                {"type": "timeout", "tool": "Bash"},
                {"type": "timeout", "tool": "Edit"},
                {"type": "timeout", "tool": "Bash"},
            ]},
            {"session_id": "s2", "error_patterns": [{"type": "timeout", "tool": "Bash"}]},
        ]
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        # Only 2 sessions, not 4 occurrences
        assert ctx.extract_recurring_error_types(str(ki), min_count=2) == [("timeout", 2)]
        assert ctx.extract_recurring_error_types(str(ki), min_count=3) == []

    def test_sorted_by_count_descending(self, ctx, tmp_path):
        """Results sorted by session count, highest first."""
        ki = tmp_path / "ki.jsonl"
        entries = []
        # timeout in 5 sessions, syntax in 3 sessions
        for i in range(5):
            entries.append({"session_id": f"t{i}", "error_patterns": [{"type": "timeout"}]})
        for i in range(3):
            entries.append({"session_id": f"x{i}", "error_patterns": [{"type": "syntax"}]})
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = ctx.extract_recurring_error_types(str(ki))
        assert result == [("timeout", 5), ("syntax", 3)]

    def test_max_results(self, ctx, tmp_path):
        """Respects max_results limit."""
        ki = tmp_path / "ki.jsonl"
        entries = []
        for etype in ["timeout", "syntax", "permission", "connection"]:
            for i in range(3):
                entries.append({"session_id": f"{etype}{i}", "error_patterns": [{"type": etype}]})
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = ctx.extract_recurring_error_types(str(ki), max_results=2)
        assert len(result) == 2

    def test_unknown_filtered(self, ctx, tmp_path):
        """Error type 'unknown' is excluded from results."""
        ki = tmp_path / "ki.jsonl"
        entries = [
            {"session_id": f"s{i}", "error_patterns": [{"type": "unknown"}]}
            for i in range(5)
        ]
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        assert ctx.extract_recurring_error_types(str(ki)) == []

    def test_malformed_entries_skipped(self, ctx, tmp_path):
        """Malformed JSON lines are gracefully skipped."""
        ki = tmp_path / "ki.jsonl"
        lines = [
            json.dumps({"session_id": "s1", "error_patterns": [{"type": "timeout"}]}),
            "{invalid json",
            json.dumps({"session_id": "s2", "error_patterns": [{"type": "timeout"}]}),
            json.dumps({"session_id": "s3", "error_patterns": [{"type": "timeout"}]}),
        ]
        ki.write_text("\n".join(lines) + "\n")
        result = ctx.extract_recurring_error_types(str(ki))
        assert result == [("timeout", 3)]

    def test_non_dict_error_patterns_skipped(self, ctx, tmp_path):
        """Non-dict items in error_patterns are gracefully skipped."""
        ki = tmp_path / "ki.jsonl"
        entries = [
            {"session_id": f"s{i}", "error_patterns": [
                {"type": "timeout"},
                "not a dict",
                42,
            ]}
            for i in range(3)
        ]
        ki.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        result = ctx.extract_recurring_error_types(str(ki))
        assert result == [("timeout", 3)]
