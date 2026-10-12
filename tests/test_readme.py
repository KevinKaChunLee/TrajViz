"""The README is the package's PyPI page: its links must work there and its quickstart must run.

PyPI shows README.md without the repository around it, so a relative link or
image (``](examples/)``, ``![..](architecture.svg)``) is broken on the
project page. The quickstart is the first code a user copies.
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")
FIXTURE = REPO / "tests" / "fixtures" / "codearts_minimal.json"


class ReadmeTests(unittest.TestCase):
    def test_links_and_images_are_absolute(self):
        targets = re.findall(r"\]\(([^)\s]+)\)", README) + re.findall(r'src="([^"]+)"', README)
        relative = [t for t in targets if not t.startswith(("https://", "http://", "mailto:"))]
        self.assertEqual(relative, [], "PyPI cannot resolve these; link to the repository instead")

    def test_the_quickstart_runs(self):
        code = re.search(r"```python\n(import agent_trajviz as tv\n.*?)```", README, re.S).group(1)
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                Path("trajectory.json").write_bytes(FIXTURE.read_bytes())
                with redirect_stdout(StringIO()) as out:
                    exec(compile(code, "README quickstart", "exec"), {})
                self.assertTrue((Path(tmp) / "out" / "trajectory.json").is_file())
            finally:
                os.chdir(cwd)
        self.assertIn("CodeArts", out.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
