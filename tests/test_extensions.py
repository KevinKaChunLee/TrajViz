"""Extension points: formats, views and figures of your own, and installed plugins.

A project built on the library must be able to analyze an agent TrajViz does
not read and add analyses of its own without editing the package:
`Trajectory.from_steps()` takes steps parsed by the caller, `register_format()`
teaches `load()` a format, `register_view()` adds a view that every consumer
of the catalog (views(), export_json, the CLI) picks up, `register_figure()`
draws it, and `load_plugins()` runs the hooks installed packages declare.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent_trajviz as tv
from agent_trajviz import plugins
from agent_trajviz import views as views_mod

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CODEARTS = FIXTURES / "codearts_minimal.json"
HAS_PLOTLY = importlib.util.find_spec("plotly") is not None

T0 = 1_700_000_000_000
STEPS = [
    {"role": "user", "text_preview": "Fix the failing test", "time_created_ms": T0},
    {"role": "assistant", "tokens": {"input": 100, "output": 20},
     "time_created_ms": T0 + 1_000, "time_completed_ms": T0 + 3_500,
     "tool_calls": [{"tool_name": "bash", "input": {"command": "pytest -q"}, "output": "1 failed",
                     "error": "exit status 1"}]},
    {"role": "assistant", "tokens": {"input": 150, "output": 10, "cache_read": 40},
     "tool_calls": [{"tool_name": "edit", "input": {"file_path": "app.py"}, "output": "ok"}]},
    {"role": "assistant", "text_preview": "Fixed.", "tokens": {"total": 70, "input": 60, "output": 10}},
]


@contextlib.contextmanager
def _isolated_registries():
    """Restore every registry afterwards, so a test cannot leak into another."""
    saved = (dict(plugins._CUSTOM), dict(plugins._LABELS), list(plugins._LOADED),
             dict(views_mod._SPECS), dict(views_mod._CATALOG))
    drawn = None
    if HAS_PLOTLY:
        from agent_trajviz import figures

        drawn = dict(figures._DRAW)
    try:
        yield
    finally:
        for live, old in zip((plugins._CUSTOM, plugins._LABELS, plugins._LOADED,
                              views_mod._SPECS, views_mod._CATALOG), saved, strict=True):
            live.clear()
            live.update(old) if isinstance(live, dict) else live.extend(old)
        if drawn is not None:
            figures._DRAW.clear()
            figures._DRAW.update(drawn)


def _toy_steps(payload):
    turns = payload["turns"] if isinstance(payload, dict) else payload
    return [{"role": "user" if t["who"] == "user" else "assistant", "text_preview": t["say"],
             "tokens": {"output": t.get("used", 0)}} for t in turns]


TOY = tv.Format(key="toy", label="Toy Agent", parse=_toy_steps,
                detect=lambda p: isinstance(p, dict) and p.get("agent") == "toy")
TOY_DOC = {"agent": "toy", "turns": [{"who": "user", "say": "hi"}, {"who": "bot", "say": "hello", "used": 12}]}


class FromStepsTests(unittest.TestCase):
    def test_minimal_steps_are_completed_to_the_step_model(self):
        given = copy.deepcopy(STEPS)
        run = tv.Trajectory.from_steps(given, format="myagent")
        self.assertEqual(given, STEPS, "the caller's steps must not be modified")
        self.assertEqual(run.format, "myagent")
        self.assertEqual([s["index"] for s in run.steps], [0, 1, 2, 3])
        failed = run.steps[1]
        self.assertEqual(failed["tool_calls"][0]["status"], "error")
        self.assertEqual((failed["tool_call_count"], failed["error_count"]), (1, 1))
        self.assertEqual(failed["duration"], 2.5)
        self.assertEqual(failed["tokens"]["total"], 120)
        self.assertEqual(run.steps[2]["tool_calls"][0]["status"], "completed")
        self.assertEqual(run.steps[3]["tokens"]["total"], 70, "a reported total is kept")
        self.assertEqual(run.steps[0]["parts"][0]["text"], "Fix the failing test")
        self.assertEqual(run.steps[2]["text_preview"], "[Tool: edit] ")

    def test_every_key_a_loader_fills_is_filled(self):
        loaded = tv.load(CODEARTS).steps[0]
        made = tv.Trajectory.from_steps(STEPS).steps[0]
        # Absent on purpose unless given: without it a non-empty `agent` names a sub-agent.
        self.assertEqual(set(loaded) - set(made), {"is_sub_agent"})
        call = next(tc for s in tv.load(CODEARTS).steps for tc in s["tool_calls"])
        self.assertEqual(set(call) - set(tv.Trajectory.from_steps(STEPS).steps[1]["tool_calls"][0]), set())

    def test_every_view_builds_and_the_metrics_add_up(self):
        run = tv.Trajectory.from_steps(STEPS)
        for vid, view in run.views().items():
            with self.subTest(view=vid):
                json.loads(view.to_json())
        self.assertEqual(run.metrics["tokens"]["total"], 120 + 200 + 70)
        self.assertEqual(run.view("tool_usage").data["tools"][0]["calls"], 1)

    def test_completing_is_idempotent(self):
        run = tv.Trajectory.from_steps(STEPS)
        again = tv.Trajectory.from_steps(json.loads(json.dumps(run.steps)))
        self.assertEqual(json.dumps(again.steps, sort_keys=True), json.dumps(run.steps, sort_keys=True))

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            tv.Trajectory.from_steps([])
        with self.assertRaises(TypeError):
            tv.Trajectory.from_steps([{"role": "user"}, "not a step"])


class FormatRegistryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_isolated_registries())
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _write(self, name, text):
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_load_detects_a_registered_format(self):
        tv.register_format(TOY)
        path = self._write("run.json", json.dumps(TOY_DOC))
        run = tv.load(path)
        self.assertEqual((run.format, run.format_label), ("toy", "Toy Agent"))
        self.assertEqual(tv.FORMATS["toy"], "Toy Agent")
        self.assertEqual(run.source, str(path))
        self.assertEqual(run.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual([s["text_preview"] for s in run.steps], ["hi", "hello"])
        self.assertEqual(tv.load(path, format="toy").format, "toy")
        json.loads(run.view("summary").to_json())

    def test_json_lines_reach_the_format_as_records(self):
        seen = []
        tv.register_format(tv.Format("toyl", "Toy Lines", parse=_toy_steps,
                                     detect=lambda p: seen.append(p) or (isinstance(p, list) and p[0].get("who") == "user")))
        path = self._write("run.jsonl", "\n".join(json.dumps(t) for t in TOY_DOC["turns"]) + "\n")
        self.assertEqual(tv.load(path).format, "toyl")
        self.assertIsInstance(seen[0], list)

    def test_a_format_without_detect_is_used_only_when_required(self):
        tv.register_format(tv.Format("toy", "Toy Agent", parse=_toy_steps))
        path = self._write("run.json", json.dumps(TOY_DOC))
        with self.assertRaises(tv.LoadError):
            tv.load(path)
        self.assertEqual(tv.load(path, format="toy").format, "toy")

    def test_built_in_formats_win_over_a_greedy_detect(self):
        tv.register_format(tv.Format("greedy", "Greedy", parse=_toy_steps, detect=lambda p: True))
        self.assertEqual(tv.load(CODEARTS).format, "codearts")

    def test_a_failing_parse_is_a_load_error_and_a_failing_detect_no_match(self):
        def broken(payload):
            raise KeyError("turns")

        tv.register_format(tv.Format("toy", "Toy Agent", parse=broken, detect=TOY.detect))
        tv.register_format(tv.Format("crashy", "Crashy", parse=_toy_steps, detect=lambda p: p["missing"]))
        with self.assertRaises(tv.LoadError) as caught:
            tv.load(self._write("run.json", json.dumps(TOY_DOC)))
        self.assertEqual(caught.exception.code, "parse_failed")
        with self.assertRaises(tv.LoadError) as caught:
            tv.load(self._write("other.json", json.dumps({"something": "else"})))
        self.assertNotEqual(caught.exception.code, "parse_failed")

    def test_registration_rules(self):
        with self.assertRaises(ValueError):
            tv.register_format(tv.Format("codex", "Not Codex", parse=_toy_steps))
        tv.register_format(TOY)
        with self.assertRaises(ValueError):
            tv.register_format(TOY)
        tv.register_format(tv.Format("toy", "Toy v2", parse=_toy_steps), replace=True)
        self.assertEqual(tv.FORMATS["toy"], "Toy v2")
        for bad in ("", "unknown", "Has Space"):
            with self.subTest(key=bad), self.assertRaises(ValueError):
                tv.register_format(tv.Format(bad, "x", parse=_toy_steps))
        with self.assertRaises(TypeError):
            tv.register_format({"key": "toy"})


class ViewRegistryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_isolated_registries())
        self.run = tv.load(CODEARTS)

    @staticmethod
    def _tool_count(run, top=3):
        counts = {}
        for step in run.steps:
            for call in step["tool_calls"]:
                counts[call["tool_name"]] = counts.get(call["tool_name"], 0) + 1
        return {"tools": sorted(counts.items(), key=lambda kv: -kv[1])[:top]}

    def test_a_registered_view_joins_the_catalog_and_every_consumer(self):
        before = list(tv.VIEW_CATALOG)
        tv.register_view("tool_count", self._tool_count, title="Tool count", kind="table",
                         description="Calls per tool.", options={"top": "How many tools to keep."})
        self.assertEqual(list(tv.VIEW_CATALOG), [*before, "tool_count"])
        self.assertEqual(dict(tv.VIEW_CATALOG["tool_count"]["options"]), {"top": "How many tools to keep."})
        self.assertEqual(list(self.run.views())[-1], "tool_count")
        self.assertEqual(len(self.run.view("tool_count", top=1).data["tools"]), 1)
        with self.assertRaises(TypeError):
            self.run.view("tool_count", bottom=1)
        with tempfile.TemporaryDirectory() as tmp:
            tv.export_json(self.run, tmp)
            index = json.loads((Path(tmp) / "index.json").read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / "tool_count.json").is_file())
        self.assertIn("tool_count", [v["id"] for v in index["views"]])

    def test_view_data_must_be_json(self):
        tv.register_view("opaque", lambda run: object(), title="Opaque")
        with self.assertRaises(TypeError):
            self.run.view("opaque")

    def test_registration_rules(self):
        with self.assertRaises(ValueError):
            tv.register_view("summary", self._tool_count, title="Not the summary")
        for bad in ("", "../escape", "index", "trajectory", "a b"):
            with self.subTest(view_id=bad), self.assertRaises(ValueError):
                tv.register_view(bad, self._tool_count, title="x")
        with self.assertRaises(ValueError):
            tv.register_view("tool_count", self._tool_count, title="x", kind="picture")
        tv.register_view("tool_count", self._tool_count, title="x")
        with self.assertRaises(ValueError):
            tv.register_view("tool_count", self._tool_count, title="x")
        tv.register_view("tool_count", self._tool_count, title="y", replace=True)
        self.assertEqual(tv.VIEW_CATALOG["tool_count"]["title"], "y")


@unittest.skipUnless(HAS_PLOTLY, "needs the charts extra")
class FigureRegistryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_isolated_registries())

    def test_a_registered_figure_draws_a_registered_view(self):
        import plotly.graph_objects as go

        from agent_trajviz import figures

        tv.register_view("tool_bar", ViewRegistryTests._tool_count, title="Tools", kind="chart")
        figures.register_figure("tool_bar", lambda data, dark: go.Figure(go.Bar(
            x=[t for t, _ in data["tools"]], y=[n for _, n in data["tools"]])))
        self.assertIn("tool_bar", figures.FIGURE_VIEWS)
        fig = figures.figure(tv.load(CODEARTS).view("tool_bar"))
        self.assertEqual(fig.data[0].type, "bar")
        with self.assertRaises(ValueError):
            figures.register_figure("token_usage", lambda data, dark: go.Figure())


class _EntryPoint:
    def __init__(self, name, hook):
        self.name, self.value, self._hook = name, f"pkg:{name}", hook

    def load(self):
        return self._hook


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(_isolated_registries())

    def test_installed_hooks_run_once(self):
        calls = []

        def hook():
            calls.append(1)
            tv.register_format(TOY)

        with mock.patch.object(plugins, "entry_points", return_value=[_EntryPoint("toy", hook)]) as eps:
            self.assertEqual(tv.load_plugins(), ["toy"])
            self.assertEqual(tv.load_plugins(), ["toy"])
        eps.assert_called_with(group="agent_trajviz.plugins")
        self.assertEqual(calls, [1])
        self.assertIn("toy", tv.FORMATS)

    def test_a_failing_hook_names_the_plugin(self):
        def hook():
            raise RuntimeError("boom")

        with mock.patch.object(plugins, "entry_points", return_value=[_EntryPoint("broken", hook)]), \
                self.assertRaises(tv.PluginError) as caught:
            tv.load_plugins()
        self.assertIn("broken", str(caught.exception))
        self.assertIsInstance(caught.exception, tv.TrajVizError)

    def test_the_command_line_loads_plugins(self):
        from agent_trajviz.insight import __main__ as cli

        with mock.patch.object(plugins, "load_plugins", return_value=[]) as hook, \
                contextlib.redirect_stdout(None):
            cli.main(["--list-views"])
        hook.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
