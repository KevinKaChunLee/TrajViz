"""Step labels and run groups through the public API, matching the dashboard.

The dashboard's labels panel and its Comparison tab (N-run scorecard and the
labelled pairwise phase charts) are now public: `label_views`,
`label_comparison_views`, `compare_runs` and `run_group_views`. Their figures
must be the dashboard's own.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import agent_trajviz as tv
from agent_trajviz.figures import figure
from agent_trajviz.insight.charts import (
    build_phase_count_comparison_chart,
    build_phase_duration_comparison_chart,
    build_run_group_agent_timeline,
)
from agent_trajviz.insight.presenters.label_ui import build_label_ui_payload
from agent_trajviz.insight.run_group import build_run_group_scorecard

REF = Path(__file__).resolve().parent / "fixtures" / "reference" / "data" / "trajectory" / "claude_code"
DJANGO, ASTROPY = REF / "django__django-11477.json", REF / "astropy__astropy-13033.json"
PHASES = [("understand", "code_reading"), ("plan", "task_planning"), ("implement", "code_editing"),
          ("validate", "test_running"), ("debug", "error_analysis")]


def write_labels(trajectory: Path, directory: Path, *, shift: int = 0) -> Path:
    """A v2 labels sidecar for *trajectory*, as agent-trajviz-label writes one."""
    run = tv.load(trajectory)
    records = []
    for i, step in enumerate(run.steps):
        if step["role"] == "user":
            phase, action = "user", "user_prompt"
        else:
            phase, action = PHASES[(i + shift) % len(PHASES)]
        records.append({"index": step["index"], "role": step["role"], "phase": phase, "action": action,
                        "duration_s": step.get("duration") or 0})
    path = directory / f"{trajectory.stem}_labeled.json"
    path.write_text(json.dumps({"schema_version": "trajectory_labels.v2", "taxonomy_version": "v1",
                                "model": "test-model", "trajectory_file": str(trajectory), "steps": records}))
    return path


class LabelViewTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.labels = write_labels(DJANGO, self.dir)

    def test_label_views_and_their_figures_match_the_labels_panel(self):
        views = tv.label_views(self.labels)
        self.assertEqual(list(views), ["labels_summary", "label_phase_counts", "label_action_counts",
                                       "label_phase_durations", "label_action_durations", "label_timeline"])
        panel = build_label_ui_payload(str(self.labels))
        for vid, key in (("label_phase_counts", "phase_count_fig"), ("label_action_counts", "action_count_fig"),
                         ("label_phase_durations", "phase_duration_fig"),
                         ("label_action_durations", "action_duration_fig"), ("label_timeline", "timeline_fig")):
            with self.subTest(view=vid):
                json.loads(views[vid].to_json())
                self.assertEqual(figure(views[vid]).to_plotly_json(), panel[key].to_plotly_json())
        self.assertEqual(views["labels_summary"].data["model"], "test-model")

    def test_a_file_that_is_not_a_labels_sidecar_is_a_load_error(self):
        bad = self.dir / "bad.json"
        bad.write_text('{"steps": []}')
        with self.assertRaises(tv.LoadError) as ctx:
            tv.label_views(bad)
        self.assertEqual(ctx.exception.code, "invalid_labels")

    def test_label_comparison_matches_the_comparison_tab_charts(self):
        other = write_labels(ASTROPY, self.dir, shift=2)
        views = tv.label_comparison_views(self.labels, other, reference_name="django", compared_name="astropy")
        from agent_trajviz.insight.labels import aggregate_labels, load_labeled_json

        ref, cmp_ = aggregate_labels(load_labeled_json(str(self.labels))), aggregate_labels(load_labeled_json(str(other)))
        expected = {
            "label_phase_count_comparison": build_phase_count_comparison_chart(
                ref["phase_counts"], cmp_["phase_counts"], ref_label="django", cmp_label="astropy"),
            "label_phase_duration_comparison": build_phase_duration_comparison_chart(
                ref["phase_durations"], cmp_["phase_durations"], ref_label="django", cmp_label="astropy"),
        }
        for vid, fig in expected.items():
            with self.subTest(view=vid):
                self.assertEqual(figure(views[vid]).to_plotly_json(), fig.to_plotly_json())


class RunGroupTests(unittest.TestCase):
    def test_compare_runs_scores_each_run_and_draws_the_dashboard_timeline(self):
        result = tv.compare_runs([DJANGO, ASTROPY])
        json.dumps(result, allow_nan=False)
        self.assertEqual([r["label"] for r in result["rows"]], [r["label"] for r in
                                                                build_run_group_scorecard([str(DJANGO), str(ASTROPY)])["rows"]])
        views = tv.run_group_views(result)
        self.assertEqual(list(views), ["run_scorecard", "run_behavior", "run_timelines"])
        dashboard = build_run_group_agent_timeline(
            build_run_group_scorecard([str(DJANGO), str(ASTROPY)])["timeline_runs"])
        self.assertEqual(figure(views["run_timelines"]).to_plotly_json(), dashboard.to_plotly_json())

    def test_trajectories_can_be_passed_loaded(self):
        result = tv.compare_runs([tv.load(DJANGO), tv.load(ASTROPY)], labels=["a", "b"])
        self.assertEqual([r["label"] for r in result["rows"]], ["a", "b"])

    def test_one_run_is_not_a_group(self):
        with self.assertRaises(ValueError):
            tv.compare_runs([DJANGO])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
