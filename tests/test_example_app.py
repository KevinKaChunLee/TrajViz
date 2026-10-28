"""End to end: a Gradio app built only on the public API, driven over HTTP.

examples/views_app.py is what another project would write on top of the
library. These tests check it imports nothing internal, then launch it on a
local port and use it through Gradio's own client -- upload a trajectory,
read the summary and tables, draw views, download the JSON export, compare
two runs -- exactly as a browser session would.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import agent_trajviz as tv

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "views_app.py"
CC_RUN = REPO / "tests" / "fixtures" / "native_claude_code" / "django__django-11477.jsonl"
OTHER_RUN = REPO / "tests" / "fixtures" / "native_claude_code" / "astropy__astropy-13033.jsonl"
PUBLIC_MODULES = {"agent_trajviz", "agent_trajviz.figures", "agent_trajviz.dashboard"}


def _load_example():
    spec = importlib.util.spec_from_file_location("trajviz_example_app", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PublicApiOnlyTests(unittest.TestCase):
    def test_the_example_imports_only_public_names(self):
        tree = ast.parse(EXAMPLE.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agent_trajviz"):
                if node.module == "agent_trajviz":
                    for alias in node.names:
                        self.assertIn(alias.name, set(tv.__all__) | {"dashboard", "figures"}, alias.name)
                else:
                    self.assertIn(node.module, PUBLIC_MODULES)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("agent_trajviz"):
                        self.assertIn(alias.name, PUBLIC_MODULES)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "tv":
                self.assertIn(node.attr, tv.__all__, f"tv.{node.attr} is not public")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveAppTests(unittest.TestCase):
    """Launch the example on a free local port and use it through gradio_client."""

    @classmethod
    def setUpClass(cls):
        try:
            from gradio_client import Client, handle_file
        except ImportError:  # pragma: no cover - gradio ships it
            raise unittest.SkipTest("gradio_client unavailable") from None
        cls.handle_file = staticmethod(handle_file)
        cls.example = _load_example()
        cls.app = cls.example.build_app()
        cls.app.launch(server_name="127.0.0.1", server_port=_free_port(), prevent_thread_lock=True,
                       quiet=True, show_error=True)
        cls.client = Client(cls.app.local_url, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.app.close()

    def test_load_draw_export_and_compare(self):
        # Upload: summary line, health table and issue table come back.
        # The handlers return gr.update(...), so each output arrives as an
        # update envelope whose "value" is the component's serialized value.
        summary, health, issues = self.client.predict(self.handle_file(str(CC_RUN)), api_name="/load")
        self.assertIn("Claude Code", summary)
        self.assertIn("24 steps", summary)
        self.assertEqual(len(health["value"]["data"]), len(tv.load(CC_RUN).health))
        self.assertEqual(health["value"]["headers"], ["metric", "status", "label", "detail"])
        self.assertEqual(issues["value"]["headers"], ["kind", "title", "detail", "steps"])

        # Views are served for the same session: a chart, then a table.
        plot, table, text = self.client.predict("token_usage", api_name="/view")
        self.assertTrue(plot["visible"])
        self.assertEqual(plot["value"]["type"], "plotly")
        self.assertFalse(table["visible"])
        self.assertEqual(json.loads(text)["id"], "token_usage")
        plot, table, text = self.client.predict("health", api_name="/view")
        self.assertFalse(plot["visible"])
        self.assertTrue(table["visible"])
        self.assertEqual(json.loads(text)["data"], json.loads(tv.load(CC_RUN).view("health").to_json())["data"])

        # Export: a zip holding every view and the index.
        zipped = self.client.predict(api_name="/export")
        with zipfile.ZipFile(zipped) as zf:
            names = set(zf.namelist())
        self.assertTrue({f"{vid}.json" for vid in tv.VIEW_CATALOG} <= names)
        self.assertIn("index.json", names)

        # Compare two runs of the same task.
        summary, milestones, divergence = self.client.predict(
            self.handle_file(str(CC_RUN)), self.handle_file(str(OTHER_RUN)), api_name="/compare")
        self.assertEqual(summary["evidence_level"], "single_pair_hypothesis")
        self.assertTrue(milestones and divergence)

    def test_multi_session_capture_requires_selection_and_keeps_unknown_tokens(self):
        from tests.test_otlp_e2e import _bundle

        with tempfile.TemporaryDirectory() as tmp:
            _, archive = _bundle(Path(tmp), partial=True)
            inspected = self.client.predict(self.handle_file(str(archive)), api_name="/sessions")
            self.assertTrue(inspected["visible"])
            self.assertIsNone(inspected["value"])
            self.assertEqual({item[1] for item in inspected["choices"]}, {"demo-session", "second-session"})
            summary, _, _ = self.client.predict(self.handle_file(str(archive)), api_name="/load")
            self.assertIn("multiple sessions", summary.lower())
            summary, _, _ = self.client.predict(self.handle_file(str(archive)), "second-session",
                                                api_name="/select_session")
            self.assertIn("n/a tokens", summary)
            _, _, view = self.client.predict("summary", api_name="/view")
            self.assertIsNone(json.loads(view)["data"]["tokens"]["total"])
            exported = self.client.predict(api_name="/export")
            with zipfile.ZipFile(exported) as archive:
                restored = tv.Trajectory.from_dict(json.loads(archive.read("trajectory.json")))
            self.assertEqual(restored.raw["_otel"]["session_id"], "second-session")
            self.assertIsNone(restored.metrics["tokens"]["total"])
            hidden = self.client.predict(self.handle_file(str(CC_RUN)), api_name="/sessions")
            self.assertFalse(hidden["visible"])
            self.assertIsNone(hidden["value"])

    def test_native_summary_identifies_estimates_and_incomplete_tool_evidence(self):
        from tests.test_cursor_evidence import capture

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "cursor.json"
            source.write_text(json.dumps(capture()), encoding="utf-8")
            summary, _, _ = self.client.predict(self.handle_file(str(source)), api_name="/load")
            self.assertIn("estimated tokens", summary)
            self.assertIn("limited evidence", summary)
            self.assertIn("n/a success", summary)

    def test_compressed_native_file_is_accepted_by_the_upload_component(self):
        from tests.test_dsh_native import events, write

        with tempfile.TemporaryDirectory() as tmp:
            source = write(Path(tmp) / "native", events())
            summary, _, _ = self.client.predict(self.handle_file(str(source)), api_name="/load")
            self.assertIn("DeepSeek Harness", summary)
            self.assertIn("70 tokens", summary)

    def test_a_non_trajectory_upload_is_a_message_not_a_crash(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write('{"hello": "world"}')
        summary, _, _ = self.client.predict(self.handle_file(fh.name), api_name="/load")
        self.assertIn("Could not load the file", summary)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
