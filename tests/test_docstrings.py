"""The API reference is generated from docstrings, so every public name has one.

`agent-trajviz`'s reference (built with pdoc in CI, see the README) documents
the top-level package and the optional public modules. A public function,
class, method or property without a docstring would be an empty entry there.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import unittest

PUBLIC_MODULES = ["agent_trajviz", "agent_trajviz.figures", "agent_trajviz.dashboard", "agent_trajviz.llm",
                  "agent_trajviz.attribution"]


def _undocumented(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    missing = [] if (inspect.getdoc(module) or "").strip() else [module_name]
    for name in module.__all__:
        obj = getattr(module, name)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if not (inspect.getdoc(obj) or "").strip():
            missing.append(f"{module_name}.{name}")
        if not inspect.isclass(obj):
            continue
        for member_name, member in vars(obj).items():
            if member_name.startswith("_"):
                continue
            target = getattr(member, "__func__", None) or getattr(member, "func", None) or member
            if (isinstance(member, property) or inspect.isfunction(target)) and not (inspect.getdoc(target) or "").strip():
                missing.append(f"{module_name}.{name}.{member_name}")
    return missing


class DocstringTests(unittest.TestCase):
    def test_every_public_name_is_documented(self):
        for module_name in PUBLIC_MODULES:
            with self.subTest(module=module_name):
                try:
                    missing = _undocumented(module_name)
                except ImportError as exc:  # an optional module whose extra is absent
                    self.skipTest(str(exc))
                self.assertEqual(missing, [])

    def test_every_public_module_declares_its_names(self):
        for module_name in PUBLIC_MODULES:
            if module_name != "agent_trajviz" and importlib.util.find_spec(module_name) is None:
                continue
            with self.subTest(module=module_name):
                try:
                    module = importlib.import_module(module_name)
                except ImportError as exc:
                    self.skipTest(str(exc))
                self.assertTrue(module.__all__)
                self.assertEqual([n for n in module.__all__ if not hasattr(module, n)], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
