"""Launch options shared by the two dashboards.

Both apps bind to 127.0.0.1 by default. Once a user passes `--share` or a
non-loopback `--host`, anyone who can reach the port can upload files, spend
the LLM key configured on this machine, and -- in Converge -- make the server
read trajectory files named in a manifest. That must not happen silently:
the launcher warns unless a login is configured, a login can be required
through the environment (never a command-line password, which `ps` and shell
history would show), uploads are size-capped, and server-path inputs are
switched off for exposed launches.
"""

from __future__ import annotations

import argparse
import logging
import os
import unittest
from unittest import mock

from agent_trajviz import _serve


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _serve.add_server_arguments(parser, default_port=7860)
    return parser.parse_args(list(argv))


class LoopbackTests(unittest.TestCase):
    def test_loopback_hosts(self):
        for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost"):
            with self.subTest(host=host):
                self.assertTrue(_serve.is_loopback(host))

    def test_everything_else_is_exposed(self):
        for host in ("0.0.0.0", "::", "192.168.1.5", "10.0.0.1", "example.com", ""):
            with self.subTest(host=host):
                self.assertFalse(_serve.is_loopback(host))


class AuthTests(unittest.TestCase):
    def _auth(self, value):
        env = {} if value is None else {_serve.AUTH_ENV: value}
        with mock.patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop(_serve.AUTH_ENV, None)
            return _serve.auth_from_env()

    def test_unset_means_no_login(self):
        self.assertIsNone(self._auth(None))
        self.assertIsNone(self._auth(""))

    def test_user_and_password(self):
        self.assertEqual(self._auth("alice:s3cret"), ("alice", "s3cret"))

    def test_the_password_may_contain_a_colon(self):
        self.assertEqual(self._auth("alice:a:b"), ("alice", "a:b"))

    def test_malformed_values_are_rejected_not_ignored(self):
        for value in ("alice", "alice:", ":s3cret", ":"):
            with self.subTest(value=value), self.assertRaises(ValueError) as ctx:
                self._auth(value)
            # The message must not echo the secret back.
            self.assertNotIn("s3cret", str(ctx.exception))


class LaunchKwargsTests(unittest.TestCase):
    def _kwargs(self, *argv, auth=None):
        with (
            mock.patch.object(_serve, "auth_from_env", return_value=auth),
            self.assertLogs("agent_trajviz", level="WARNING") as logs,
        ):
            # assertLogs fails on silence, so log a sentinel and drop it below.
            logging.getLogger("agent_trajviz").warning("sentinel")
            kwargs = _serve.launch_kwargs(_args(*argv))
        return kwargs, [r.getMessage() for r in logs.records if r.getMessage() != "sentinel"]

    def test_local_launch_is_quiet(self):
        kwargs, warnings = self._kwargs()
        self.assertEqual(warnings, [])
        self.assertEqual(kwargs["server_name"], "127.0.0.1")
        self.assertEqual(kwargs["server_port"], 7860)
        self.assertFalse(kwargs["share"])
        self.assertIsNone(kwargs["auth"])

    def test_exposed_launch_without_login_warns(self):
        for argv in (("--share",), ("--host", "0.0.0.0")):
            with self.subTest(argv=argv):
                _, warnings = self._kwargs(*argv)
                self.assertEqual(len(warnings), 1)
                self.assertIn(_serve.AUTH_ENV, warnings[0])

    def test_exposed_launch_with_login_is_quiet(self):
        kwargs, warnings = self._kwargs("--share", auth=("alice", "pw"))
        self.assertEqual(warnings, [])
        self.assertEqual(kwargs["auth"], ("alice", "pw"))

    def test_uploads_are_capped_by_default(self):
        kwargs, _ = self._kwargs()
        self.assertEqual(kwargs["max_file_size"], _serve.DEFAULT_MAX_FILE_SIZE)
        kwargs, _ = self._kwargs("--max-file-size", "2gb")
        self.assertEqual(kwargs["max_file_size"], "2gb")

    def test_exposed_reports_the_exposure(self):
        self.assertFalse(_serve.is_exposed(_args()))
        self.assertTrue(_serve.is_exposed(_args("--share")))
        self.assertTrue(_serve.is_exposed(_args("--host", "0.0.0.0")))


class _FakeApp:
    def __init__(self):
        self.launched_with = None

    def launch(self, **kwargs):
        self.launched_with = kwargs


class InsightLauncherTests(unittest.TestCase):
    def _run(self, *argv, env=None):
        from agent_trajviz.insight import __main__ as entry
        from agent_trajviz.insight import insight

        app = _FakeApp()
        calls = {}

        def fake_build_ui(**kwargs):
            calls.update(kwargs)
            return app

        with mock.patch.object(insight, "build_ui", fake_build_ui), \
                mock.patch.dict(os.environ, env or {}, clear=False):
            if not env:
                os.environ.pop(_serve.AUTH_ENV, None)
            code = entry.main(list(argv))
        return code, calls, app.launched_with

    def test_local_launch_keeps_server_path_inputs(self):
        code, calls, launched = self._run()
        self.assertEqual(code, 0)
        self.assertTrue(calls["allow_server_paths"])
        self.assertEqual(launched["server_name"], "127.0.0.1")
        self.assertIn("css", launched)  # presentation still applied

    def test_exposed_launch_turns_server_path_inputs_off(self):
        with self.assertLogs("agent_trajviz", level="WARNING"):
            code, calls, launched = self._run("--share")
        self.assertEqual(code, 0)
        self.assertFalse(calls["allow_server_paths"])
        self.assertTrue(launched["share"])

    def test_bad_login_variable_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run(env={_serve.AUTH_ENV: "nopassword"})
        self.assertEqual(ctx.exception.code, 2)


class ConvergeLauncherTests(unittest.TestCase):
    def _run(self, *argv):
        from agent_trajviz.converge import __main__ as entry
        from agent_trajviz.converge import app as cvg_app

        app = _FakeApp()
        calls = {}

        def fake_build_ui(**kwargs):
            calls.update(kwargs)
            return app

        with mock.patch.object(cvg_app, "build_ui", fake_build_ui), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_serve.AUTH_ENV, None)
            code = entry.main(list(argv))
        return code, calls, app.launched_with

    def test_defaults(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            code, calls, launched = self._run("--data-root", tmp)
            self.assertEqual(code, 0)
            self.assertEqual(calls["data_root"], Path(tmp).resolve())
        self.assertEqual(launched["server_name"], "127.0.0.1")
        self.assertEqual(launched["server_port"], 7861)  # beside Insight's 7860
        self.assertEqual(launched["max_file_size"], _serve.DEFAULT_MAX_FILE_SIZE)
        self.assertIn("css", launched)

    def test_exposed_launch_warns(self):
        with self.assertLogs("agent_trajviz", level="WARNING"):
            self._run("--host", "0.0.0.0")

    def test_missing_data_root_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            self._run("--data-root", "/definitely/not/here")
        self.assertEqual(ctx.exception.code, 2)


class ServerPathInputTests(unittest.TestCase):
    """The Attribution tab's reference-data root names a path on this machine."""

    def _app(self, allow):
        from agent_trajviz.insight.insight import build_ui

        return build_ui(allow_server_paths=allow)

    def _root_boxes(self, app):
        return [b for b in app.blocks.values() if getattr(b, "label", None) == "Reference data root"]

    def _diagnose_roots(self, allow, sent_root):
        from agent_trajviz.insight import attribution

        app = self._app(allow)
        fns = app.fns.values() if isinstance(app.fns, dict) else app.fns
        on_diagnose = next(f.fn for f in fns if getattr(f.fn, "__name__", "") == "on_diagnose")
        seen = []

        def fake_diagnose(**kwargs):
            seen.append(kwargs["reference_root"])
            return attribution.AttributionResult(False, reason="stub")

        with mock.patch.object(attribution, "diagnose", fake_diagnose):
            on_diagnose({"_source_path": "/x/agent/inst.json"}, "", "", sent_root)
        return seen

    def test_a_local_launch_offers_the_box(self):
        (box,) = self._root_boxes(self._app(True))
        self.assertTrue(box.visible)
        self.assertEqual(self._diagnose_roots(True, "/srv/reference"), ["/srv/reference"])

    def test_an_exposed_launch_hides_the_box_and_ignores_a_sent_value(self):
        (box,) = self._root_boxes(self._app(False))
        self.assertFalse(box.visible)
        self.assertFalse(box.interactive)
        # A client can call the endpoint directly; the value must still be dropped.
        self.assertEqual(self._diagnose_roots(False, "/etc"), [None])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
