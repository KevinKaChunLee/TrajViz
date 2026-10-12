"""A trajectory survives serialization: to_dict -> JSON -> from_dict / load.

Other projects cache and transmit analyses. `to_dict(include_raw=True)` is
self-contained, `Trajectory.from_dict()` rebuilds it, `load()` recognizes an
exported `trajectory.json`, and `export_json()` writes one -- so an export
directory reloads to identical analysis: every view of the reloaded run must
equal the original's.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import agent_trajviz as tv

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RUNS = [FIXTURES / "reference" / "data" / "trajectory" / "claude_code" / "django__django-11477.json",
        FIXTURES / "reference" / "data" / "trajectory" / "claude_code" / "astropy__astropy-13033.json",
        FIXTURES / "codearts_minimal.json", FIXTURES / "cursor_minimal.json", FIXTURES / "icode_minimal.json"]


def _views(run: tv.Trajectory) -> dict:
    return {vid: json.loads(v.to_json())["data"] for vid, v in run.views().items()}


class RoundTripTests(unittest.TestCase):
    def test_from_dict_reproduces_every_view(self):
        for path in RUNS:
            with self.subTest(run=path.name):
                run = tv.load(path)
                data = json.loads(json.dumps(run.to_dict(include_raw=True)))
                again = tv.Trajectory.from_dict(data)
                self.assertEqual((again.format, again.source, again.sha256, again.warnings),
                                 (run.format, run.source, run.sha256, run.warnings))
                self.assertEqual(_views(again), _views(run))

    def test_an_export_directory_reloads(self):
        run = tv.load(RUNS[0])
        with tempfile.TemporaryDirectory() as tmp:
            tv.export_json(run, tmp)
            again = tv.load(Path(tmp) / "trajectory.json")
        self.assertEqual(_views(again), _views(run))
        self.assertEqual(again.sha256, run.sha256)

    def test_without_raw_the_steps_still_round_trip(self):
        run = tv.load(RUNS[0])
        again = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict())))
        self.assertEqual(json.dumps(again.steps, sort_keys=True), json.dumps(run.steps, sort_keys=True))
        self.assertEqual(again.raw, {})

    def test_a_document_from_another_layout_is_refused(self):
        data = tv.load(RUNS[0]).to_dict()
        for broken in ({**data, "schema_version": 99}, {k: v for k, v in data.items() if k != "steps"}, {"x": 1}):
            with self.subTest(keys=sorted(broken)[:3]), self.assertRaises(ValueError):
                tv.Trajectory.from_dict(broken)

    def test_load_marks_an_exported_trajectory_by_type(self):
        data = tv.load(RUNS[0]).to_dict()
        self.assertEqual(data["type"], "agent-trajviz-trajectory")

    def test_load_reads_an_export_written_with_sorted_keys(self):
        run = tv.load(RUNS[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(run.to_dict(include_raw=True), sort_keys=True), encoding="utf-8")
            again = tv.load(path)
        self.assertEqual(_views(again), _views(run))

    def test_load_checks_a_required_format_and_the_version(self):
        run = tv.load(RUNS[0])
        other = next(f for f in tv.FORMATS if f != run.format)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            path.write_text(json.dumps(run.to_dict()), encoding="utf-8")
            self.assertEqual(tv.load(path, format=run.format).format, run.format)
            with self.assertRaises(tv.LoadError) as caught:
                tv.load(path, format=other)
            self.assertEqual(caught.exception.code, "mismatch")
            path.write_text(json.dumps({**run.to_dict(), "schema_version": 99}), encoding="utf-8")
            with self.assertRaises(tv.LoadError) as caught:
                tv.load(path)
            self.assertEqual(caught.exception.code, "unsupported_schema")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
