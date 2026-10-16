"""What the dashboard's controls do is available through the public API.

The format selector is `load(..., format=)`; the context-utilization agent,
window and snapshot controls are options of the `context_pressure` and
`context_usage` views; the Workflow tab's step classification and filters are
the `workflow` view. Each must give exactly what the dashboard computes.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import agent_trajviz as tv
from agent_trajviz.insight.charts import activity
from agent_trajviz.insight.context_usage import (
    PRESSURE_ALL_AGENTS,
    context_pressure_series,
    context_usage_breakdown,
    estimate_tokens,
)
from agent_trajviz.insight.presenters.workflow import FILTER_CHIPS_DEFAULT, filter_workflow_steps
from agent_trajviz.insight.rendering import _card_style
from agent_trajviz.insight.session import load_session

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CC_RUN = FIXTURES / "native_claude_code" / "django__django-11477.jsonl"


def _data(view):
    return json.loads(view.to_json())["data"]


class LoadFormatTests(unittest.TestCase):
    def test_a_matching_format_loads(self):
        self.assertEqual(tv.load(FIXTURES / "codearts_minimal.json", format="codearts").format, "codearts")

    def test_a_different_detected_format_is_a_mismatch(self):
        with self.assertRaises(tv.LoadError) as ctx:
            tv.load(CC_RUN, format="codex")
        self.assertEqual(ctx.exception.code, "mismatch")

    def test_an_unknown_format_name_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            tv.load(CC_RUN, format="no-such-format")


class CompareAnchorTests(unittest.TestCase):
    """The Comparison tab's optional anchor patch is `compare(..., anchor_patch=)`."""

    def test_an_anchor_patch_grounds_the_report(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
            fh.write("diff --git a/django/core/checks/urls.py b/django/core/checks/urls.py\n"
                     "--- a/django/core/checks/urls.py\n+++ b/django/core/checks/urls.py\n@@ -1 +1 @@\n-x\n+y\n")
        plain = tv.compare(CC_RUN, CC_RUN)
        anchored = tv.compare(CC_RUN, CC_RUN, anchor_patch=fh.name)
        self.assertIsNone(plain["anchor_analysis"])
        self.assertIsNotNone(anchored["anchor_analysis"])
        self.assertNotEqual(anchored["anchor_mode"], plain["anchor_mode"])

    def test_a_missing_anchor_patch_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            tv.compare(CC_RUN, CC_RUN, anchor_patch=FIXTURES / "no_such.patch")


class ContextPressureOptionTests(unittest.TestCase):
    def setUp(self):
        self.traj = tv.load(CC_RUN)
        self.session = load_session(str(CC_RUN))

    def test_options_are_catalogued_and_checked(self):
        self.assertEqual(set(tv.VIEW_CATALOG["context_pressure"]["options"]),
                         {"agent", "window_limit", "snapshot_step"})
        with self.assertRaises(TypeError):
            self.traj.view("context_pressure", colour="red")
        with self.assertRaises(TypeError):
            self.traj.view("health", agent="x")

    def test_agent_and_window_match_the_dashboard_controls(self):
        data = _data(self.traj.view("context_pressure"))
        keys = [a["key"] for a in data["agents"]]
        self.assertEqual(keys[0], PRESSURE_ALL_AGENTS)
        for key in keys:
            for window in (None, 64_000):
                with self.subTest(agent=key, window=window):
                    view = self.traj.view("context_pressure", agent=key, window_limit=window, snapshot_step=3)
                    expected = context_pressure_series(self.session.steps, agent_key=key, raw=self.session.raw,
                                                       window_limit=window)
                    self.assertEqual(_data(view)["series"], json.loads(json.dumps(expected)))
                    from agent_trajviz.figures import figure

                    dashboard = activity.build_context_pressure_chart(
                        self.session.steps, series=expected, highlight_step=3)
                    self.assertEqual(figure(view).to_plotly_json(), dashboard.to_plotly_json())


def _tok(inp):
    return {"total": inp, "input": inp, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}


def _step(idx, *, sid, agent, inp, text, summary=False):
    return {"index": idx, "role": "assistant", "agent": agent, "is_sub_agent": True, "session_id": sid,
            "tokens": _tok(inp), "parts": [{"type": "text", "text": text}], "tool_calls": [],
            "tool_call_count": 0, "model_id": "", "session_title": "", "summary": summary,
            "is_compaction_checkpoint": False, "message_type": "", "finish": "", "duration": None}


class ContextUsageOptionTests(unittest.TestCase):
    """A sub-agent window that compacts once: the snapshot before it is selectable."""

    def setUp(self):
        sid = "ses_child"
        steps = [
            _step(0, sid=sid, agent="explore", inp=50_000, text="BEFORE " + "x" * 160),
            _step(1, sid=sid, agent="explore", inp=60_000, text="MORE " + "x" * 40),
            _step(2, sid=sid, agent="compaction", inp=8_000, text="OBJECTIVE " + "x" * 120, summary=True),
            _step(3, sid=sid, agent="explore", inp=9_000, text="AFTER " + "x" * 40),
        ]
        steps[2]["mode"] = "compaction"
        self.sid, self.steps = sid, steps
        self.traj = tv.Trajectory(format="opencode", steps=steps)

    def test_snapshots_and_selection_match_the_breakdown(self):
        keys = [a["key"] for a in _data(self.traj.view("context_pressure"))["agents"]]
        agent = next(k for k in keys if k != PRESSURE_ALL_AGENTS)
        current = _data(self.traj.view("context_usage", agent=agent))
        values = [s["value"] for s in current["snapshots"]]
        self.assertGreater(len(values), 1, "a compaction should offer a pre-compaction snapshot")
        snapshot = int(values[1])
        picked = _data(self.traj.view("context_usage", agent=agent, snapshot_step=snapshot))
        expected = context_usage_breakdown(self.steps, agent_key=agent, snapshot_step=snapshot)
        self.assertEqual({k: picked[k] for k in expected}, json.loads(json.dumps(expected)))
        self.assertEqual(picked["buckets"]["conversation"] > 0, True)
        self.assertGreater(estimate_tokens("BEFORE " + "x" * 160), 0)


class WorkflowViewTests(unittest.TestCase):
    def test_categories_and_filters_match_the_workflow_tab(self):
        session = load_session(str(CC_RUN))
        data = _data(tv.load(CC_RUN).view("workflow"))
        self.assertEqual(len(data["steps"]), len(session.steps))
        labels = {"tool_error": "Tool Error", "system_error": "System Error", "final": "Final",
                  "tool_calls": "Tool Calls", "reasoning": "Reasoning"}
        for row, step in zip(data["steps"], session.steps, strict=True):
            with self.subTest(step=row["step"]):
                expected_label = _card_style(step)[2]
                self.assertEqual(labels.get(row["category"], expected_label), expected_label)
        # The tab needs a role chip; feature chips narrow within the roles.
        for chips, wanted in ((["Assistant"], "Assistant"), (["User"], "User"),
                              (["Assistant", "User", "Tool Calls"], "Tool Calls"),
                              (["Assistant", "User", "Errors"], "Errors"),
                              (["Assistant", "User", "Reasoning"], "Reasoning")):
            with self.subTest(filter=wanted):
                shown = set(filter_workflow_steps(session.steps, chips))
                self.assertEqual({r["position"] for r in data["steps"] if wanted in r["filters"]}, shown)
        self.assertEqual(data["filters"]["roles"] + ["All"], FILTER_CHIPS_DEFAULT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
