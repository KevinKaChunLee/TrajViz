"""Every command follows one convention.

Console-script targets take ``argv`` (default: ``sys.argv[1:]``) and return
the exit status: 0 success, 1 a failure the input caused, 2 a usage error.
Before this the commands mixed ``sys.exit`` calls, ``None`` returns and
printed tracebacks, so none could be driven in-process and the exit status
depended on which command you ran.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _scripts() -> dict[str, str]:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["scripts"]


def _quiet(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*args)
    return code, out.getvalue(), err.getvalue()


class ScriptTests(unittest.TestCase):
    def test_every_command_carries_the_distribution_prefix(self):
        for name in _scripts():
            with self.subTest(script=name):
                self.assertTrue(name == "agent-trajviz" or name.startswith("agent-trajviz-"))

    def test_every_target_takes_argv_and_returns_the_status(self):
        for name, target in _scripts().items():
            with self.subTest(script=name):
                module, func = target.split(":")
                main = getattr(importlib.import_module(module), func)
                params = list(inspect.signature(main).parameters.values())
                self.assertEqual([p.name for p in params], ["argv"])
                self.assertIsNone(params[0].default)
                self.assertIn(inspect.signature(main).return_annotation, (int, "int"))


class ConvergeCliTests(unittest.TestCase):
    def setUp(self):
        from agent_trajviz.converge import cli

        self.cli = cli

    def test_a_missing_file_is_status_1_with_a_sentence(self):
        code, out, err = _quiet(self.cli.main, ["/no/such/ref.json", "/no/such/cmp.json"])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("Error:", err)

    def test_a_non_positive_token_rate_is_a_usage_error(self):
        for rate in ("0", "-1", "nan", "inf", "fast"):
            with self.subTest(rate=rate), self.assertRaises(SystemExit) as ctx:
                _quiet(self.cli.main, ["a.json", "b.json", "--token-rate", rate])
            self.assertEqual(ctx.exception.code, 2)

    def test_a_bug_is_not_dressed_up_as_an_input_error(self):
        from unittest import mock

        with mock.patch.object(self.cli, "build_comparison_report", side_effect=KeyError("bug")), \
                self.assertRaises(KeyError):
            _quiet(self.cli.main, ["a.json", "b.json"])


class OpenCodeCollectorTests(unittest.TestCase):
    def test_no_arguments_is_a_usage_error(self):
        from agent_trajviz.collect import opencode

        code, _, err = _quiet(opencode.main, [])
        self.assertEqual(code, 2)
        self.assertIn("agent-trajviz-collect opencode", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
