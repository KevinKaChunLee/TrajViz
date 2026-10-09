"""The Converge web app must not read outside its data root or leak internals.

Batch manifests name trajectory files on the server. The app resolved them
against the upload's temporary directory (so relative paths never worked) and
accepted any absolute path, which made every reachable Converge instance a
probe for arbitrary files. Manifest paths now resolve against the app's data
root and must stay inside it, symlinks included. Errors render the message
-- with server paths cut back to what the user supplied -- and never a
traceback; a cleared Token Rate box is a message, not a TypeError.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from agent_trajviz.converge import app as cvg_app
from agent_trajviz.converge.batch import parse_manifest


def _trajectory(path: Path, *, steps: int = 3) -> Path:
    entries = []
    for i in range(steps):
        entries.append({
            "type": "assistant", "uuid": f"u{i}",
            "message": {
                "id": f"m{i}", "role": "assistant",
                "content": [{"type": "text", "text": f"step {i}"},
                            {"type": "tool_use", "id": f"t{i}", "name": "Bash",
                             "input": {"command": f"echo {i}"}}],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trajectory": entries}))
    return path


class _Dirs(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name).resolve()
        self.root = base / "data"
        self.outside = base / "elsewhere"
        self.uploads = base / "uploads"  # where the web server keeps an upload
        for d in (self.root, self.outside, self.uploads):
            d.mkdir()
        _trajectory(self.root / "runs" / "ref.json")
        _trajectory(self.root / "runs" / "cmp.json", steps=4)
        _trajectory(self.outside / "secret.json")

    def manifest(self, entries, name="manifest.json") -> str:
        path = self.uploads / name
        path.write_text(json.dumps(entries))
        return str(path)


class ManifestConfinementTests(_Dirs):
    def test_relative_paths_resolve_against_the_base_dir(self):
        m = self.manifest([{"task_id": "t", "reference": "runs/ref.json", "compared": "runs/cmp.json"}])
        (entry,) = parse_manifest(m, base_dir=self.root, root=self.root)
        self.assertEqual(entry.reference, str(self.root / "runs" / "ref.json"))

    def test_absolute_path_outside_the_root_is_refused(self):
        m = self.manifest([{"task_id": "t", "reference": str(self.outside / "secret.json"),
                            "compared": "runs/cmp.json"}])
        with self.assertRaises(ValueError) as ctx:
            parse_manifest(m, base_dir=self.root, root=self.root)
        self.assertIn("outside", str(ctx.exception))

    def test_dot_dot_cannot_climb_out(self):
        m = self.manifest([{"task_id": "t", "reference": "../elsewhere/secret.json",
                            "compared": "runs/cmp.json"}])
        with self.assertRaises(ValueError):
            parse_manifest(m, base_dir=self.root, root=self.root)

    def test_a_symlink_inside_the_root_cannot_point_out(self):
        link = self.root / "runs" / "link.json"
        try:
            os.symlink(self.outside / "secret.json", link)
        except (OSError, NotImplementedError):  # pragma: no cover - platform
            self.skipTest("symlinks unavailable")
        m = self.manifest([{"task_id": "t", "reference": "runs/link.json", "compared": "runs/cmp.json"}])
        with self.assertRaises(ValueError):
            parse_manifest(m, base_dir=self.root, root=self.root)

    def test_anchor_paths_are_confined_too(self):
        m = self.manifest([{"task_id": "t", "reference": "runs/ref.json", "compared": "runs/cmp.json",
                            "anchor": str(self.outside / "secret.json")}])
        with self.assertRaises(ValueError):
            parse_manifest(m, base_dir=self.root, root=self.root)

    def test_the_command_line_keeps_unconfined_manifest_relative_paths(self):
        # The CLI runs as the user on the user's files: no root, and relative
        # paths still resolve against the manifest's own directory.
        local = self.root / "local.json"
        local.write_text(json.dumps([{"task_id": "t", "reference": "runs/ref.json",
                                      "compared": str(self.outside / "secret.json")}]))
        (entry,) = parse_manifest(str(local))
        self.assertEqual(entry.reference, str(self.root / "runs" / "ref.json"))
        self.assertEqual(entry.compared, str(self.outside / "secret.json"))


class BatchViewTests(_Dirs):
    def test_batch_runs_with_paths_relative_to_the_data_root(self):
        m = self.manifest([{"task_id": "t", "reference": "runs/ref.json", "compared": "runs/cmp.json"}])
        html = cvg_app.batch_report_html(m, 50.0, False, data_root=self.root)
        self.assertIn("1 of 1 tasks compared", html)

    def test_batch_refuses_a_path_outside_the_data_root(self):
        m = self.manifest([{"task_id": "t", "reference": str(self.outside / "secret.json"),
                            "compared": "runs/cmp.json"}])
        html = cvg_app.batch_report_html(m, 50.0, False, data_root=self.root)
        self.assertIn("outside", html)
        self.assertNotIn("Traceback", html)

    def test_before_after_is_confined_the_same_way(self):
        good = self.manifest([{"task_id": "t", "reference": "runs/ref.json", "compared": "runs/cmp.json"}],
                             name="before.json")
        bad = self.manifest([{"task_id": "t", "reference": str(self.outside / "secret.json"),
                              "compared": "runs/cmp.json"}], name="after.json")
        html = cvg_app.before_after_report_html(good, bad, 50.0, False, data_root=self.root)
        self.assertIn("outside", html)

    def test_error_messages_do_not_carry_server_paths(self):
        m = self.manifest([{"task_id": "t", "reference": "runs/missing.json", "compared": "runs/cmp.json"}])
        html = cvg_app.batch_report_html(m, 50.0, False, data_root=self.root)
        self.assertIn("runs/missing.json", html)
        self.assertNotIn(str(self.root), html)
        self.assertNotIn(str(self.uploads), html)


class CompareViewTests(_Dirs):
    def test_a_cleared_token_rate_is_a_message(self):
        ref, cmp_ = str(self.root / "runs" / "ref.json"), str(self.root / "runs" / "cmp.json")
        for rate in (None, 0, -5, float("nan"), float("inf")):
            with self.subTest(rate=rate):
                html = cvg_app.compare_uploads(ref, cmp_, None, rate, False)[0]
                self.assertIn("Token rate", html)
                self.assertNotIn("Traceback", html)
                self.assertNotIn("TypeError", html)

    def test_batch_rate_is_validated_too(self):
        m = self.manifest([{"task_id": "t", "reference": "runs/ref.json", "compared": "runs/cmp.json"}])
        self.assertIn("Token rate", cvg_app.batch_report_html(m, None, False, data_root=self.root))

    def test_a_bad_upload_shows_the_reason_without_a_traceback_or_server_path(self):
        bad = self.uploads / "notes.json"
        bad.write_text('{"hello": 1}')
        html = cvg_app.compare_uploads(str(bad), str(self.root / "runs" / "cmp.json"), None, 50.0, False)[0]
        self.assertIn("no steps parsed", html)
        self.assertIn("notes.json", html)
        self.assertNotIn("Traceback", html)
        self.assertNotIn(str(self.uploads), html)

    def test_an_unexpected_failure_reports_only_its_type(self):
        ref, cmp_ = str(self.root / "runs" / "ref.json"), str(self.root / "runs" / "cmp.json")
        from unittest import mock

        with mock.patch("agent_trajviz.converge.alignment.build_comparison_report",
                        side_effect=KeyError("internal_key /srv/secret")), \
                self.assertLogs("agent_trajviz", level="ERROR"):
            html = cvg_app.compare_uploads(ref, cmp_, None, 50.0, False)[0]
        self.assertIn("KeyError", html)
        self.assertNotIn("/srv/secret", html)
        self.assertNotIn("Traceback", html)

    def test_the_app_module_no_longer_formats_tracebacks(self):
        source = Path(cvg_app.__file__).read_text()
        self.assertNotIn("format_exc", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
