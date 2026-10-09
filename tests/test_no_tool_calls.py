"""A run with no tool calls has no tool success rate: "n/a", never "0%".

A 0% rate claims every call failed; with no calls there is nothing to rate.
``compute_metrics`` reports ``None`` and every surface says "n/a" (the health
verdict already did). None of the real exports measured has zero tool calls,
so this changes no reported number on them.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import agent_trajviz as tv
from agent_trajviz.insight.assistant import build_analysis_brief_from_session
from agent_trajviz.insight.presenters.overview import build_overview_outputs
from agent_trajviz.insight.session import LoadedSession, load_session


def _export_without_tool_calls(directory: str) -> Path:
    entries = [
        {"index": 0, "role": "user", "uuid": "u0", "content": "explain the repo",
         "timestamp": "2026-01-01T00:00:00.000Z"},
        {"index": 1, "role": "assistant", "uuid": "a1", "message_id": "m1",
         "content": [{"type": "text", "text": "It is a parser."}],
         "usage": {"input_tokens": 120, "output_tokens": 30}, "timestamp": "2026-01-01T00:00:05.000Z"},
    ]
    path = Path(directory) / "no_tools.json"
    path.write_text(json.dumps({"format": "ccsession-trajectory", "trajectory": entries}), encoding="utf-8")
    return path


class NoToolCallsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = _export_without_tool_calls(self._tmp.name)

    def test_the_metric_is_none(self):
        metrics = tv.load(self.path).metrics
        self.assertEqual(metrics["tool_call_count"], 0)
        self.assertIsNone(metrics["tool_success_rate"])

    def test_the_kpi_card_and_the_brief_say_na(self):
        session = load_session(str(self.path))
        self.assertIsInstance(session, LoadedSession)
        kpi = build_overview_outputs(session)["kpi_html"]
        self.assertNotIn("None%", kpi)
        self.assertNotIn(">0%<", kpi)
        self.assertIn("n/a", kpi)
        brief = build_analysis_brief_from_session(session)
        self.assertIn("n/a success", brief)
        self.assertNotIn("None", brief.split("tool_calls:")[1].splitlines()[0])

    def test_the_health_verdict_is_not_applicable(self):
        verdict = next(v for v in tv.load(self.path).health if v["metric"] == "Tool Success")
        self.assertEqual(verdict["label"], "N/A")


if __name__ == "__main__":
    unittest.main()
