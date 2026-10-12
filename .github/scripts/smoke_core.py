"""The public API end to end from an installed wheel without extras.

CI runs this on each operating system the package declares (the test suite
itself runs on Linux and macOS): every fixture format loads, every view is
JSON, an export reloads to the same views, runs compare, schemas and the
command line work.
"""

from __future__ import annotations

import os
import subprocess
import sysconfig
import tempfile
from pathlib import Path

import agent_trajviz as tv

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
RUNS = [FIXTURES / "codearts_minimal.json", FIXTURES / "cursor_minimal.json", FIXTURES / "icode_minimal.json",
        *sorted((FIXTURES / "reference" / "data" / "trajectory").rglob("*.json"))]


def _views(run: tv.Trajectory) -> dict[str, str]:
    return {vid: view.to_json() for vid, view in run.views().items()}


def main() -> None:
    assert len(RUNS) >= 5, RUNS
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        loaded = []
        for path in RUNS:
            run = tv.load(path)
            tv.export_json(run, out / path.stem)
            assert _views(tv.load(out / path.stem / "trajectory.json")) == _views(run), path
            loaded.append(run)
        reference, compared = loaded[-2], loaded[-1]
        for view in tv.comparison_views(tv.compare(reference, compared)).values():
            view.to_json()
        group = tv.compare_runs(loaded[-2:] + [out / "missing.json"])
        assert [bool(row["error"]) for row in group["rows"]] == [False, False, True], group["rows"]
        made = tv.Trajectory.from_steps([
            {"role": "user", "text_preview": "Fix the test"},
            {"role": "assistant", "tokens": {"input": 10, "output": 5},
             "tool_calls": [{"tool_name": "bash", "input": {"command": "pytest"}, "error": "exit 1"}]},
        ])
        assert made.view("tool_usage").data["tools"][0]["calls"] == 1
        assert set(tv.VIEW_CATALOG) <= set(tv.view_schemas())

        scripts = Path(sysconfig.get_path("scripts"))
        command = str(scripts / ("agent-trajviz.exe" if os.name == "nt" else "agent-trajviz"))
        subprocess.run([command, "--json", str(RUNS[0]), "-o", str(out / "cli")], check=True, capture_output=True)
        assert len(list((out / "cli").glob("*.json"))) == len(tv.VIEW_CATALOG) + 2
        subprocess.run([command, "--schemas", str(out / "schemas")], check=True, capture_output=True)
        assert len(list((out / "schemas").glob("*.schema.json"))) == len(tv.view_schemas())
    print(f"core smoke passed: {len(RUNS)} runs, {len(tv.VIEW_CATALOG)} views each")


if __name__ == "__main__":
    main()
