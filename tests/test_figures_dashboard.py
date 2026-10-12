"""Optional presentation layers built on the views: Plotly figures and Gradio.

`agent_trajviz.figures` (charts extra) draws any chart view with the
dashboard's own code; `agent_trajviz.dashboard` (app extra) launches the
dashboards and turns any view into a Gradio component. Neither is imported
by the core, and each names its extra when it is missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import agent_trajviz as tv

REPO = Path(__file__).resolve().parent.parent
CC_RUN = REPO / "tests" / "fixtures" / "reference" / "data" / "trajectory" / "claude_code" / "django__django-11477.json"


def _roundtrip(view):
    return tv.View(view.id, view.title, view.kind, view.description, json.loads(view.to_json())["data"])


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent_trajviz import figures

        cls.figures = figures
        cls.traj = tv.load(CC_RUN)

    def test_every_chart_view_has_a_figure(self):
        charts = {vid for vid, info in tv.VIEW_CATALOG.items() if info["kind"] == "chart"}
        comparison = {vid for vid, v in tv.comparison_views(tv.compare(self.traj, self.traj)).items()
                      if v.kind == "chart"}
        self.assertEqual(self.figures.FIGURE_VIEWS, charts | comparison)

    def test_plan_and_pressure_figures_match_the_dashboard(self):
        from agent_trajviz.insight.charts import activity
        from agent_trajviz.insight.context_usage import PRESSURE_ALL_AGENTS
        from agent_trajviz.insight.session import load_session

        session = load_session(str(CC_RUN))
        expected = {
            "plan": activity.build_plan_timeline_chart(session.plan_history, session.plan_metrics),
            "context_pressure": activity.build_context_pressure_chart(
                session.steps, agent_key=PRESSURE_ALL_AGENTS, raw=session.raw),
        }
        for vid, dashboard_fig in expected.items():
            with self.subTest(view=vid):
                fig = self.figures.figure(_roundtrip(self.traj.view(vid)))
                self.assertEqual(fig.to_plotly_json(), dashboard_fig.to_plotly_json())

    def test_a_table_view_has_no_figure(self):
        with self.assertRaises(ValueError):
            self.figures.figure(self.traj.view("health"))

    def test_figures_draws_only_the_charts(self):
        figs = self.figures.figures(self.traj.views(), dark=True)
        self.assertEqual(set(figs), {v for v in self.traj.views() if v in self.figures.FIGURE_VIEWS})


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from agent_trajviz import dashboard

        cls.dashboard = dashboard
        cls.traj = tv.load(CC_RUN)

    def test_view_component_types(self):
        import gradio as gr

        with gr.Blocks(analytics_enabled=False):
            plot = self.dashboard.view_component(self.traj.view("token_usage"))
            table = self.dashboard.view_component(self.traj.view("health"))
            other = self.dashboard.view_component(self.traj.view("metrics"))
        self.assertIsInstance(plot, gr.Plot)
        self.assertIsInstance(table, gr.Dataframe)
        self.assertIsInstance(other, gr.JSON)
        self.assertEqual(table.label, "Health checks")

    def test_both_dashboards_build_and_carry_their_presentation(self):
        import gradio as gr

        self.assertIsInstance(self.dashboard.insight_app(allow_server_paths=False), gr.Blocks)
        self.assertIsInstance(self.dashboard.converge_app(), gr.Blocks)
        self.assertIn("css", self.dashboard.insight_presentation())
        self.assertIn("css", self.dashboard.converge_presentation())

    def test_render_views_places_one_accordion_per_view(self):
        import gradio as gr

        with gr.Blocks(analytics_enabled=False) as demo:
            self.dashboard.render_views(self.traj.views(only=["summary", "health", "tool_usage"]))
        accordions = [b for b in demo.blocks.values() if isinstance(b, gr.Accordion)]
        self.assertEqual([a.label for a in accordions], ["Summary", "Health checks", "Tool calls"])


class OptionalImportTests(unittest.TestCase):
    def _run(self, code: str, blocked: set[str]) -> subprocess.CompletedProcess:
        prelude = textwrap.dedent(f"""
            import importlib.abc, sys
            class _Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in {blocked!r}:
                        raise ModuleNotFoundError(name)
            sys.meta_path.insert(0, _Block())
            sys.path.insert(0, {str(REPO)!r})
        """)
        return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)],
                              capture_output=True, text=True, timeout=180)

    def test_core_import_loads_neither_layer(self):
        proc = self._run("""
            import sys, agent_trajviz
            print(sorted(m for m in sys.modules if m in ("agent_trajviz.figures", "agent_trajviz.dashboard",
                                                       "plotly", "gradio")))
        """, set())
        self.assertEqual(proc.stdout.strip(), "[]", proc.stderr)

    def test_each_layer_names_its_extra(self):
        for module, blocked, extra in (("figures", {"plotly"}, "charts"), ("dashboard", {"gradio"}, "app")):
            with self.subTest(module=module):
                proc = self._run(f"""
                    import agent_trajviz
                    try:
                        import agent_trajviz.{module}
                    except agent_trajviz.MissingExtraError as exc:
                        print("MISSING", exc.extra)
                """, blocked)
                self.assertEqual(proc.stdout.strip(), f"MISSING {extra}", proc.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
