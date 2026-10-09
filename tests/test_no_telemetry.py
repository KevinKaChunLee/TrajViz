"""Building a dashboard must not phone home.

Gradio enables usage analytics by default: constructing a ``gr.Blocks``
starts a thread that fetches a version check from its servers, and
launching reports the launch. TrajViz promises that nothing leaves the
machine unless the user asks, so both dashboards turn analytics off. The
subprocess test records socket activity with an audit hook while both UIs are
built, so a future Gradio default or a new Blocks cannot slip past it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class NoTelemetryTests(unittest.TestCase):
    def test_both_apps_disable_gradio_analytics(self):
        from agent_trajviz.converge.app import build_ui as converge_ui
        from agent_trajviz.insight.insight import build_ui as insight_ui

        for build in (insight_ui, converge_ui):
            with self.subTest(app=build.__module__):
                self.assertFalse(build().analytics_enabled)

    def test_building_the_apps_opens_no_connection(self):
        script = textwrap.dedent("""
            import os, sys, threading
            os.environ.pop("GRADIO_ANALYTICS_ENABLED", None)  # Gradio's own default: on
            seen = []
            def hook(name, args):
                if name in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname"):
                    seen.append(name)
            sys.addaudithook(hook)
            from agent_trajviz.insight.insight import build_ui as insight_ui
            from agent_trajviz.converge.app import build_ui as converge_ui
            insight_ui()
            converge_ui()
            for t in threading.enumerate():
                if t is not threading.main_thread():
                    t.join(timeout=5)
            print("EVENTS", sorted(set(seen)))
        """)
        proc = subprocess.run(
            [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("EVENTS []", proc.stdout, proc.stdout + proc.stderr)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
