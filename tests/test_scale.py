"""Analysis scales linearly with the number of steps.

Long runs are synthesized by replaying a recorded Claude Code session's turns
with unique ids and shifted clocks. A run four times longer must take well under sixteen
times as long (what a quadratic pass would cost) to load and to produce
every view -- the margin is wide so a busy CI machine cannot make it flaky.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import agent_trajviz as tv

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "native_claude_code" / "django__django-11477.jsonl")
_ID = re.compile(r'"((?:toolu_|msg_|req_)[A-Za-z0-9_]+|00000000-0000-4000-8000-[0-9]{12})"')
_TS = re.compile(r'"timestamp": "([0-9T:\-\.]+)Z"')


def replayed(copies: int, directory: Path) -> Path:
    """The fixture's turns replayed *copies* times as one longer session."""
    data = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    session_id = data[0]["sessionId"]
    head, body = data[:6], data[6:]
    template = json.dumps(body)
    entries = list(head)
    for k in range(copies):
        def new_id(m: re.Match[str], k: int = k) -> str:
            return f'"{m.group(1)}-c{k}"'

        def shifted(m: re.Match[str], k: int = k) -> str:
            stamp = datetime.fromisoformat(m.group(1)) + timedelta(seconds=110 * k)
            return f'"timestamp": "{stamp.isoformat(timespec="milliseconds")}Z"'

        copied = json.loads(_TS.sub(shifted, _ID.sub(new_id, template)))
        # Message and tool IDs change per replay; the session remains one run.
        for entry in copied:
            entry["sessionId"] = session_id
        entries.extend(copied)
    path = directory / f"replayed_{copies}.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def _time_everything(path: Path) -> tuple[float, int]:
    start = time.perf_counter()
    run = tv.load(path)
    views = run.views()
    for view in views.values():
        view.to_json()
    return time.perf_counter() - start, len(run.steps)


class ScaleTests(unittest.TestCase):
    def test_load_and_every_view_scale_linearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            small, small_steps = _time_everything(replayed(30, Path(tmp)))
            large, large_steps = _time_everything(replayed(120, Path(tmp)))
        self.assertGreater(large_steps, 2_500)
        self.assertAlmostEqual(large_steps / small_steps, 4, delta=0.3)
        self.assertLess(large / small, 10, f"{small:.2f}s for {small_steps} steps, {large:.2f}s for {large_steps}")

    def test_a_long_run_exports_completely(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = tv.load(replayed(40, Path(tmp)))
            paths = tv.export_json(run, Path(tmp) / "out")
        self.assertEqual(len(paths), len(tv.VIEW_CATALOG) + 2)
        self.assertEqual(len(run.view("steps").data), len(run.steps))
        original_session = json.loads(FIXTURE.read_text().splitlines()[0])["sessionId"]
        self.assertEqual({step["session_id"] for step in run.steps}, {original_session})
        lanes = run.view("agent_swimlane").data["lanes"]
        self.assertEqual(len([lane for lane in lanes if lane["color_index"] is not None]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
