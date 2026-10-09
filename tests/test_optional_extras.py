"""The core runs without third-party packages; a feature that needs one names its extra.

Each check runs in a subprocess where the chosen packages cannot be imported,
so it sees what a fresh ``pip install agent-trajviz`` (without that extra) sees.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "reference" / "data" / "trajectory" / "claude_code" / "django__django-11477.json"
THIRD_PARTY = ("gradio", "plotly", "pandas", "numpy", "pygments", "requests", "awe")


def _run(code: str, blocked: tuple[str, ...] = THIRD_PARTY) -> subprocess.CompletedProcess:
    prelude = textwrap.dedent(f"""
        import importlib.abc, sys
        _BLOCKED = {set(blocked)!r}

        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in _BLOCKED:
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)

        sys.meta_path.insert(0, _Block())
        sys.path.insert(0, {str(REPO)!r})
    """)
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)],
        capture_output=True, text=True, timeout=180, cwd=str(REPO),
    )


class CoreWithoutExtrasTests(unittest.TestCase):
    def test_core_modules_import_with_no_third_party_package(self):
        proc = _run("""
            import importlib
            for name in ("agent_trajviz", "agent_trajviz.api", "agent_trajviz.insight.loaders", "agent_trajviz.insight.parser",
                         "agent_trajviz.insight.metrics", "agent_trajviz.insight.session",
                         "agent_trajviz.insight.context_usage", "agent_trajviz.insight.patterns",
                         "agent_trajviz.insight.run_group", "agent_trajviz.converge.alignment",
                         "agent_trajviz.converge.cli", "agent_trajviz.collect.cursor",
                         "agent_trajviz.collect.opencode", "agent_trajviz.collect.codearts"):
                importlib.import_module(name)
            print("ok")
        """)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)

    def test_a_trajectory_loads_and_measures_with_no_third_party_package(self):
        proc = _run(f"""
            from agent_trajviz.insight.session import load_session
            session = load_session({str(FIXTURE)!r})
            print(len(session.steps), session.metrics["tokens"]["total"] > 0)
        """)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertRegex(proc.stdout.strip(), r"^[1-9][0-9]* True$")

    def test_the_public_api_loads_and_compares_with_no_third_party_package(self):
        proc = _run(f"""
            import agent_trajviz as tv
            run = tv.load({str(FIXTURE)!r})
            print(run.format, run.metrics["tokens"]["total"] > 0, "alignment" in tv.compare(run, run))
        """)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "ccsession True True")

    def test_write_report_without_charts_raises_missing_extra(self):
        proc = _run(f"""
            import agent_trajviz as tv
            try:
                tv.write_report({str(FIXTURE)!r})
            except tv.MissingExtraError as exc:
                print("extra:", exc.extra)
        """, blocked=("plotly",))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "extra: charts")

    def test_the_collection_tools_run_with_no_third_party_package(self):
        proc = _run("""
            from agent_trajviz.collect.__main__ import main
            raise SystemExit(main(["--help"]))
        """)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("cursor", proc.stdout)


class MissingExtraHintTests(unittest.TestCase):
    def _assert_hint(self, proc: subprocess.CompletedProcess, extra: str) -> None:
        self.assertEqual(proc.returncode, 2, proc.stderr)
        self.assertIn(f'pip install "agent-trajviz[{extra}]"', proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_the_dashboard_names_the_app_extra(self):
        self._assert_hint(_run("""
            import sys
            from agent_trajviz.insight.__main__ import main
            sys.argv = ["agent-trajviz"]
            main()
        """, blocked=("gradio",)), "app")

    def test_the_html_report_names_the_charts_extra(self):
        self._assert_hint(_run(f"""
            import sys
            from agent_trajviz.insight.__main__ import main
            sys.argv = ["agent-trajviz", "--report", {str(FIXTURE)!r}]
            main()
        """, blocked=("plotly",)), "charts")

    def test_the_converge_dashboard_names_the_app_extra(self):
        self._assert_hint(_run("""
            from agent_trajviz.converge.__main__ import main
            main()
        """, blocked=("gradio",)), "app")

    def test_the_labeler_names_the_llm_extra(self):
        self._assert_hint(_run(f"""
            import sys
            from agent_trajviz.labeling.labeler import main
            sys.argv = ["agent-trajviz-label", {str(FIXTURE)!r}]
            main()
        """, blocked=("requests",)), "llm")

    def test_the_labeler_help_needs_no_extra(self):
        proc = _run("""
            import sys
            from agent_trajviz.labeling.labeler import main
            sys.argv = ["agent-trajviz-label", "--help"]
            main()
        """, blocked=("requests",))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("usage: agent-trajviz-label", proc.stdout)


if __name__ == "__main__":
    unittest.main()
