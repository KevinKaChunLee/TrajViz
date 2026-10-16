"""A trajectory survives serialization: to_dict -> JSON -> from_dict / load.

Other projects cache and transmit analyses. `to_dict(include_raw=True)` is
self-contained, `Trajectory.from_dict()` rebuilds it, `load()` recognizes an
exported `trajectory.json`, and `export_json()` writes one -- so an export
directory reloads to identical analysis: every view of the reloaded run must
equal the original's.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import agent_trajviz as tv

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUNS = [FIXTURES / "native_claude_code" / "django__django-11477.jsonl",
        FIXTURES / "native_claude_code" / "astropy__astropy-13033.jsonl",
        FIXTURES / "codearts_minimal.json", FIXTURES / "cursor_minimal.json", FIXTURES / "icode_minimal.json"]


def _views(run: tv.Trajectory) -> dict:
    return {vid: json.loads(v.to_json())["data"] for vid, v in run.views().items()}


def _legacy_public_claude_export():
    """A schema-v1 normalized document written before native Claude support."""
    call = {"type": "tool_call", "tool_id": "spawn-child", "tool_name": "Task",
            "status": "success", "input": {"prompt": "Inspect the parser"},
            "output": "Parser inspected", "metadata": {}}
    run = tv.Trajectory.from_steps([
        {"role": "assistant", "agent": "", "model_id": "claude-test",
         "tokens": {"total": 10, "input": 2, "output": 3, "cache_read": 5, "cache_write": 0},
         "parts": [call], "tool_calls": [call], "duration": 2.0,
         "time_created_ms": 1767225600000, "time_completed_ms": 1767225602000},
        {"role": "assistant", "agent": "child", "model_id": "claude-test",
         "tokens": {"total": 7, "input": 2, "output": 5, "cache_read": 0, "cache_write": 0},
         "parts": [{"type": "text", "text": "Found the parser"}], "finish": "end_turn",
         "duration": 1.0, "time_created_ms": 1767225601000, "time_completed_ms": 1767225602000},
    ], format="ccsession")
    document = run.to_dict()
    # The old converter identified children through a non-empty agent field,
    # without the explicit hierarchy keys emitted by the native converter.
    for step in document["steps"]:
        for key in ("is_sub_agent", "parent_agent_id", "parent_session_id", "session_depth",
                    "session_title", "compaction_reason", "source_path"):
            step.pop(key, None)
    document["raw"] = {
        "metadata": {"agent": "claude-code", "model": "claude-test"}, "trajectory": [],
        "timing": {"total_duration": 2.0, "started_at": "2026-01-01T00:00:00Z",
                   "finished_at": "2026-01-01T00:00:02Z"},
        "_cc_format": True, "_cc_parsed_steps": copy.deepcopy(document["steps"]),
        "_cc_sub_agents": [{"agent_id": "child", "spawned_by": "spawn-child"}],
    }
    return document


class RoundTripTests(unittest.TestCase):
    def test_from_dict_reproduces_every_view(self):
        for path in RUNS:
            with self.subTest(run=path.name):
                run = tv.load(path)
                data = json.loads(json.dumps(run.to_dict(include_raw=True)))
                again = tv.Trajectory.from_dict(data)
                self.assertEqual((again.format, again.source, again.sha256, again.warnings),
                                 (run.format, run.source, run.sha256, run.warnings))
                self.assertEqual(_views(again), _views(run))

    def test_an_export_directory_reloads(self):
        run = tv.load(RUNS[0])
        with tempfile.TemporaryDirectory() as tmp:
            tv.export_json(run, tmp)
            again = tv.load(Path(tmp) / "trajectory.json")
        self.assertEqual(_views(again), _views(run))
        self.assertEqual(again.sha256, run.sha256)

    def test_without_raw_the_steps_still_round_trip(self):
        run = tv.load(RUNS[0])
        again = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict())))
        self.assertEqual(json.dumps(again.steps, sort_keys=True), json.dumps(run.steps, sort_keys=True))
        self.assertEqual(again.raw, {})

    def test_a_document_from_another_layout_is_refused(self):
        data = tv.load(RUNS[0]).to_dict()
        for broken in ({**data, "schema_version": 99}, {k: v for k, v in data.items() if k != "steps"}, {"x": 1}):
            with self.subTest(keys=sorted(broken)[:3]), self.assertRaises(ValueError):
                tv.Trajectory.from_dict(broken)

    def test_load_marks_an_exported_trajectory_by_type(self):
        data = tv.load(RUNS[0]).to_dict()
        self.assertEqual(data["type"], "agent-trajviz-trajectory")

    def test_load_reads_an_export_written_with_sorted_keys(self):
        run = tv.load(RUNS[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(run.to_dict(include_raw=True), sort_keys=True), encoding="utf-8")
            again = tv.load(path)
        self.assertEqual(_views(again), _views(run))

    def test_load_checks_a_required_format_and_the_version(self):
        run = tv.load(RUNS[0])
        other = next(f for f in tv.FORMATS if f != run.format)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(run.to_dict()), encoding="utf-8")
            self.assertEqual(tv.load(path, format=run.format).format, run.format)
            with self.assertRaises(tv.LoadError) as caught:
                tv.load(path, format=other)
            self.assertEqual(caught.exception.code, "mismatch")
            path.write_text(json.dumps({**run.to_dict(), "schema_version": 99}), encoding="utf-8")
            with self.assertRaises(tv.LoadError) as caught:
                tv.load(path)
            self.assertEqual(caught.exception.code, "unsupported_schema")

    def test_legacy_normalized_claude_export_migrates_without_mutating_input(self):
        document = _legacy_public_claude_export()
        original = copy.deepcopy(document)
        run = tv.Trajectory.from_dict(document)
        self.assertEqual((run.format, run.format_label), ("claude_code", "Claude Code"))
        self.assertEqual(run.metrics["tokens"]["total"], 17)
        self.assertEqual(run.metrics["tool_call_count"], 1)
        child = next(agent for agent in run.agents if agent["agent_id"] == "child")
        self.assertEqual(child["spawned_by_step"], 0)
        self.assertEqual(child["total_tokens"], 7)
        self.assertEqual(child["step_count"], 1)
        self.assertEqual(run.view("agents").data, run.agents)
        for old, new in (("_cc_format", "_claude_code_format"),
                         ("_cc_parsed_steps", "_claude_code_parsed_steps"),
                         ("_cc_sub_agents", "_claude_code_sub_agents")):
            self.assertNotIn(old, run.raw)
            self.assertEqual(run.raw[new], original["raw"][old])
        # Every view still renders and the new export remains self-contained.
        self.assertEqual(_views(tv.Trajectory.from_dict(run.to_dict(include_raw=True))), _views(run))
        self.assertEqual(document, original)
        self.assertEqual(run.to_dict()["schema_version"], tv.SCHEMA_VERSION)

    def test_load_accepts_legacy_normalized_export_with_explicit_native_format(self):
        document = _legacy_public_claude_export()
        expected = tv.Trajectory.from_dict(document)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            for format_hint in (None, "claude_code"):
                with self.subTest(format=format_hint):
                    run = tv.load(path, format=format_hint)
                    self.assertEqual(run.format_label, "Claude Code")
                    self.assertEqual(_views(run), _views(expected))
                    self.assertEqual(next(a for a in run.agents if a["agent_id"] == "child")["spawned_by_step"], 0)

    def test_public_export_migration_does_not_reintroduce_raw_session_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old-source.json"
            path.write_text(json.dumps({"format": "ccsession-trajectory", "trajectory": [
                {"role": "assistant", "content": [{"type": "text", "text": "Old source"}]},
            ]}), encoding="utf-8")
            with self.assertRaises(tv.LoadError):
                tv.load(path)
            with self.assertRaises(tv.LoadError):
                tv.load(path, format="claude_code")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
