"""Every chart and table is available as JSON data, and the charts draw from it.

The library's job is parsing and analysis; presentation is optional. So
each chart and table the dashboard shows is a View -- an id, a title, a kind
and JSON data -- that any application can use, and :func:`export_json`
writes them all. The dashboard's figures are drawn from the same data, so the
JSON and the chart cannot disagree; these tests hold both properties.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import agent_trajviz as tv
from agent_trajviz.views import _jsonable

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures"
CC_RUN = FIXTURES / "native_claude_code" / "django__django-11477.jsonl"
RUNS = [CC_RUN, FIXTURES / "codearts_minimal.json", FIXTURES / "cursor_minimal.json", FIXTURES / "icode_minimal.json"]


class CatalogTests(unittest.TestCase):
    def test_every_catalogued_view_is_produced_in_catalog_order(self):
        views = tv.load(CC_RUN).views()
        self.assertEqual(list(views), list(tv.VIEW_CATALOG))

    def test_catalog_entries_are_complete(self):
        for vid, info in tv.VIEW_CATALOG.items():
            with self.subTest(view=vid):
                self.assertIn(info["kind"], ("chart", "table", "summary"))
                self.assertTrue(info["title"] and info["description"])

    def test_only_selects_and_unknown_ids_are_refused(self):
        run = tv.load(CC_RUN)
        self.assertEqual(list(run.views(only=["health", "tool_usage"])), ["health", "tool_usage"])
        self.assertEqual(run.view("summary").id, "summary")
        with self.assertRaises(KeyError):
            run.views(only=["no_such_view"])


class SerializationTests(unittest.TestCase):
    def test_every_view_of_every_fixture_is_strict_json(self):
        for path in RUNS:
            for vid, view in tv.load(path).views().items():
                with self.subTest(run=path.name, view=vid):
                    doc = json.loads(view.to_json())  # to_json refuses NaN/inf
                    self.assertEqual(doc["id"], vid)
                    self.assertEqual(doc["schema_version"], tv.VIEWS_SCHEMA_VERSION)
                    self.assertEqual(set(doc), {"schema_version", "id", "title", "kind", "description", "data"})

    def test_jsonable_normalizes_or_refuses(self):
        self.assertEqual(_jsonable({"s": {"b", "a"}, "t": (1, 2), "n": math.nan}), {"s": ["a", "b"], "t": [1, 2], "n": None})
        with self.assertRaises(TypeError):
            _jsonable({"x": object()})

    def test_views_agree_with_the_trajectory_properties(self):
        run = tv.load(CC_RUN)
        self.assertEqual(run.view("metrics").data, json.loads(json.dumps(run.metrics)))
        self.assertEqual(run.view("agents").data, json.loads(json.dumps(run.agents)))
        self.assertEqual(run.view("health").data, json.loads(json.dumps(run.health)))


class ChartsDrawFromViewsTests(unittest.TestCase):
    """The figure drawn from a view's JSON equals the dashboard's figure."""

    def test_each_chart_view_redraws_the_dashboard_figure(self):
        from agent_trajviz.insight.charts import activity, swimlanes, usage
        from agent_trajviz.insight.session import load_session

        for path in RUNS:
            run = tv.load(path)
            session = load_session(str(path))
            views = {vid: json.loads(v.to_json())["data"] for vid, v in run.views().items()}
            pairs = {
                "token_usage": (usage.draw_token_usage, usage.build_token_chart(session.steps, format=session.format)),
                "step_durations": (usage.draw_step_durations, usage.build_duration_chart(session.steps)),
                "tool_usage": (usage.draw_tool_usage, usage.build_tool_chart(session.steps)),
                "tool_durations": (usage.draw_tool_durations, usage.build_tool_duration_chart(session.steps)),
                "skill_calls": (usage.draw_skill_calls, usage.build_skill_agent_chart(session.steps)),
                "agent_tokens": (usage.draw_agent_tokens, usage.build_agent_token_chart(session.agent_summaries)),
                "agent_swimlane": (swimlanes.draw_agent_swimlane, swimlanes.build_agent_swimlane_chart(session.steps)),
                "tool_outcomes": (swimlanes.draw_tool_outcomes, swimlanes.build_tool_outcome_timeline(session.steps)),
                "file_interactions": (
                    activity.draw_file_interactions,
                    activity.build_file_interaction_chart(session.file_interactions, session.target_files,
                                                          steps=session.steps),
                ),
            }
            for vid, (draw, dashboard_fig) in pairs.items():
                with self.subTest(run=path.name, view=vid):
                    self.assertEqual(draw(views[vid]).to_plotly_json(), dashboard_fig.to_plotly_json())


class ExportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "views"

    def test_export_writes_every_view_an_index_and_the_trajectory(self):
        run = tv.load(CC_RUN)
        paths = tv.export_json(run, self.out)
        names = {p.name for p in paths}
        self.assertEqual(names, {f"{vid}.json" for vid in tv.VIEW_CATALOG} | {"index.json", "trajectory.json"})
        index = json.loads((self.out / "index.json").read_text())
        self.assertEqual([v["id"] for v in index["views"]], list(tv.VIEW_CATALOG))
        self.assertEqual(index["sha256"], run.sha256)
        self.assertEqual(json.loads((self.out / "tool_usage.json").read_text()), run.view("tool_usage").to_dict())
        self.assertEqual(len(json.loads((self.out / "trajectory.json").read_text())["steps"]), len(run.steps))

    def test_export_can_be_restricted_and_repeated(self):
        run = tv.load(CC_RUN)
        tv.export_json(run, self.out, only=["health"])
        tv.export_json(run, self.out, only=["health"])  # replaces, no error
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["health.json", "index.json", "trajectory.json"])

    def test_comparison_views_export_without_a_trajectory_file(self):
        report = tv.compare(CC_RUN, CC_RUN)
        views = tv.comparison_views(report)
        self.assertIn("milestone_timeline", views)
        paths = tv.export_json(views, self.out)
        self.assertNotIn("trajectory.json", {p.name for p in paths})
        for view in views.values():
            json.loads(view.to_json())


class NoPresentationDependencyTests(unittest.TestCase):
    def test_views_and_export_need_no_third_party_package(self):
        prelude = textwrap.dedent(f"""
            import importlib.abc, sys, tempfile
            class _Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in {{"plotly", "gradio", "pygments", "requests", "pandas", "numpy"}}:
                        raise ModuleNotFoundError(name)
            sys.meta_path.insert(0, _Block())
            sys.path.insert(0, {str(REPO)!r})
            import agent_trajviz as tv
            run = tv.load({str(CC_RUN)!r})
            n = len(tv.export_json(run, tempfile.mkdtemp()))
            m = len(tv.export_json(tv.comparison_views(tv.compare(run, run)), tempfile.mkdtemp()))
            print("OK", n, m)
        """)
        proc = subprocess.run([sys.executable, "-c", prelude], capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.startswith("OK"), proc.stdout)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
