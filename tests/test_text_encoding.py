"""Every text file the package reads or writes names its encoding.

Without ``encoding=``, Python 3.11-3.14 use the locale's encoding, which on
Windows is usually a legacy code page: a UTF-8 trajectory, patch or manifest
that loads on Linux and macOS fails there (or is written in a way other
platforms misread). Binary I/O is exempt.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "agent_trajviz"
_NOT_FILE_OPENERS = {"os", "webbrowser", "zipfile", "zf", "tarfile", "sqlite3", "gzip", "resource"}


def _text_io_without_encoding(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        keywords = {k.arg for k in node.keywords}
        if "encoding" in keywords:
            continue
        if name in ("read_text", "write_text"):
            if isinstance(func, ast.Attribute) and ast.unparse(func.value) in _NOT_FILE_OPENERS:
                continue
            found.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno} {name}()")
        elif name == "open":
            if isinstance(func, ast.Attribute) and ast.unparse(func.value) in _NOT_FILE_OPENERS:
                continue
            mode = next((a.value for a in node.args[1:2] if isinstance(a, ast.Constant)), None)
            mode = next((k.value.value for k in node.keywords
                         if k.arg == "mode" and isinstance(k.value, ast.Constant)), mode)
            if isinstance(mode, str) and "b" in mode:
                continue
            found.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno} {ast.unparse(node)[:80]}")
    return found


class TextEncodingTests(unittest.TestCase):
    def test_text_io_names_its_encoding(self):
        found = [hit for path in sorted(PACKAGE.rglob("*.py")) for hit in _text_io_without_encoding(path)]
        self.assertEqual(found, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
