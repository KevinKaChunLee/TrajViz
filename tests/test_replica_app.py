"""End to end: the TrajViz dashboard rebuilt on the public API alone.

examples/trajviz_app reproduces the dashboard -- KPI strip, Overview
sections, Patterns, Attribution, Comparison, Workflow, Raw Data and the AI
sidebar -- with nothing but the public API. These tests check that by AST,
then launch it on a local port and drive every interaction over HTTP through
Gradio's own client, and check its charts are the original dashboard's.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import agent_trajviz as tv

REPO = Path(__file__).resolve().parent.parent
APP_DIR = REPO / "examples" / "trajviz_app"
REF = REPO / "tests" / "fixtures" / "reference" / "data" / "trajectory" / "claude_code"
DJANGO, ASTROPY = REF / "django__django-11477.json", REF / "astropy__astropy-13033.json"
PUBLIC = {"agent_trajviz", "agent_trajviz.figures", "agent_trajviz.llm", "agent_trajviz.attribution",
          "agent_trajviz.dashboard"}


class PublicApiOnlyTests(unittest.TestCase):
    def test_the_replica_imports_nothing_internal(self):
        for module in sorted(APP_DIR.glob("*.py")):
            tree = ast.parse(module.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                for name in names:
                    if name.startswith("agent_trajviz"):
                        with self.subTest(module=module.name, imports=name):
                            self.assertIn(name, PUBLIC)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tv":
                    with self.subTest(module=module.name, attr=node.attr):
                        self.assertIn(node.attr, tv.__all__)


def _load_app():
    sys.path.insert(0, str(APP_DIR))
    try:
        spec = importlib.util.spec_from_file_location("trajviz_replica_app", APP_DIR / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(APP_DIR))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _labels(trajectory: Path, directory: Path) -> Path:
    run = tv.load(trajectory)
    phases = [("understand", "code_reading"), ("implement", "code_editing"), ("validate", "test_running")]
    records = [{"index": s["index"], "role": s["role"],
                **({"phase": "user", "action": "user_prompt"} if s["role"] == "user"
                   else dict(zip(("phase", "action"), phases[i % 3], strict=True))),
                "duration_s": s.get("duration") or 0} for i, s in enumerate(run.steps)]
    path = directory / f"{trajectory.stem}_labeled.json"
    path.write_text(json.dumps({"schema_version": "trajectory_labels.v2", "taxonomy_version": "v1",
                                "model": "m", "trajectory_file": str(trajectory), "steps": records}))
    return path


class LiveReplicaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gradio_client import Client, handle_file

        cls.file = staticmethod(handle_file)
        cls.module = _load_app()
        cls.app = cls.module.build_app()
        cls.app.launch(server_name="127.0.0.1", server_port=_free_port(), prevent_thread_lock=True, quiet=True)
        cls.client = Client(cls.app.local_url, verbose=False)
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.app.close()
        cls.tmp.cleanup()

    def _load(self):
        return self.client.predict(self.file(str(DJANGO)), "", api_name="/load")

    def test_load_fills_every_panel_with_the_original_dashboards_numbers(self):
        out = self._load()
        text = json.dumps(out, default=str)
        for needle in ("Claude Code", "602,021", "38.3 gen tok/s", "95.8%", "EISDIR", "Showing 24 of 24 steps",
                       "Session details"):
            with self.subTest(needle=needle):
                self.assertIn(needle, text)

    def test_charts_are_the_original_dashboards(self):
        from agent_trajviz.insight.presenters.overview import build_chart_outputs
        from agent_trajviz.insight.session import load_session

        out = self._load()
        plots = [o for o in out if isinstance(o, dict) and o.get("type") == "plotly"]
        self.assertGreaterEqual(len(plots), 10)
        original = build_chart_outputs(load_session(str(DJANGO)))
        replica_token, replica_duration = json.loads(plots[0]["plot"]), json.loads(plots[1]["plot"])
        for replica, fig in ((replica_token, original["tok_fig"]), (replica_duration, original["dur_fig"])):
            expected = json.loads(fig.to_json())
            self.assertEqual(replica["data"], expected["data"])
            self.assertEqual(replica["layout"]["title"], expected["layout"]["title"])

    def test_labels_run_group_compare_and_step_detail(self):
        self._load()
        labels = self.client.predict(self.file(str(_labels(DJANGO, Path(self.tmp.name)))), api_name="/labels")
        self.assertIn("Labels loaded", json.dumps(labels))

        group = self.client.predict([self.file(str(ASTROPY))], "", api_name="/run_group")
        self.assertIn("astropy__astropy-13033", group[0])
        self.assertIn("django__django-11477", group[0])

        compared = self.client.predict(self.file(str(ASTROPY)), None, None, "", api_name="/compare")
        self.assertIn("Alignment", compared[0])

        detail = self.client.predict(1, api_name="/step")
        self.assertIn("Step #1", detail)
        self.assertIn("translate_url", detail)

    def test_export_writes_the_html_report(self):
        self._load()
        exported = self.client.predict(api_name="/export")
        path = exported["value"] if isinstance(exported, dict) else exported
        self.assertIn("<html", Path(path["path"] if isinstance(path, dict) else path).read_text()[:2000].lower())

    def test_llm_features_without_configuration_answer_with_a_message(self):
        self._load()
        history, _ = self.client.predict("What is slow?", [], api_name="/ask")
        content = history[-1]["content"]  # Gradio 6 sends message content as parts
        text = content if isinstance(content, str) else " ".join(part.get("text", "") for part in content)
        self.assertEqual(history[-1]["role"], "assistant")
        self.assertIn("not configured", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
