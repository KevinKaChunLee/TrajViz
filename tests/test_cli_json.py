"""`agent-trajviz --json` writes every chart and table without the dashboard."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import agent_trajviz as tv
from agent_trajviz.insight.__main__ import main

CC_RUN = (Path(__file__).resolve().parent / "fixtures" / "reference" / "data" / "trajectory"
          / "claude_code" / "django__django-11477.json")


def _run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class JsonExportCliTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_writes_every_view_and_prints_the_directory(self):
        out_dir = self.dir / "views"
        code, out, err = _run("--json", str(CC_RUN), "-o", str(out_dir))
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), str(out_dir))
        index = json.loads((out_dir / "index.json").read_text())
        self.assertEqual([v["id"] for v in index["views"]], list(tv.VIEW_CATALOG))

    def test_view_restricts_the_export(self):
        out_dir = self.dir / "views"
        code, _, err = _run("--json", str(CC_RUN), "-o", str(out_dir), "--view", "health", "--view", "tool_usage")
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(p.name for p in out_dir.iterdir()),
                         ["health.json", "index.json", "tool_usage.json", "trajectory.json"])

    def test_default_directory_is_named_after_the_trajectory(self):
        with contextlib.chdir(self.dir):
            code, out, err = _run("--json", str(CC_RUN))
        self.assertEqual(code, 0, err)
        self.assertEqual(Path(out.strip()).name, f"{CC_RUN.stem}.views")

    def test_an_unreadable_file_is_status_1_with_a_sentence(self):
        code, out, err = _run("--json", str(self.dir / "missing.json"), "-o", str(self.dir / "v"))
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.strip())

    def test_an_unknown_view_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(io.StringIO()):
            main(["--json", str(CC_RUN), "-o", str(self.dir / "v"), "--view", "nope"])
        self.assertEqual(ctx.exception.code, 2)

    def test_list_views_prints_the_catalog(self):
        code, out, _ = _run("--list-views")
        self.assertEqual(code, 0)
        for vid in tv.VIEW_CATALOG:
            self.assertIn(vid, out)

    def test_json_and_report_are_exclusive(self):
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(io.StringIO()):
            main(["--json", str(CC_RUN), "--report", str(CC_RUN)])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
