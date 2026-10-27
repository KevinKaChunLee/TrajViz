"""Current Codex code-mode outcomes survive native public loading."""
from __future__ import annotations

import json

import pytest

import agent_trajviz as tv


def _block(text):
    return {"type": "input_text", "text": text}


def _result(code, output="quiet"):
    return _block(json.dumps({"chunk_id": "test", "exit_code": code, "output": output}))


@pytest.mark.parametrize(("output", "expected"), [
    ([_block("Script completed\nOutput:\n"), _result(7)], "error"),
    ([_block("Error: warning text"), _result(0)], "success"),
    ([_result(0), _result(9)], "error"),
    ([_result(-9)], "error"),
    ([_result(0, '{"exit_code": 8, "error": "fixture"}')], "success"),
    ([_block("Script failed\nError executing code: unavailable")], "error"),
    ([_block("Script failed\nError executing code: boom"), _result(0)], "error"),
    ("failed to spawn code-mode host /scratch/host: No such file or directory (os error 2)", "error"),
    ([_block("All checks passed: Found 0 errors, error-free.")], "success"),
    ([_block("x" * 300 + "\nProcess exited with code 10")], "error"),
    ([_block("Process exited with code 0\nProcess exited with code 2")], "error"),
    ([_block(json.dumps({"metadata": {"exit_code": 0}, "output": "Error: fixture"}))], "success"),
    ([_block(json.dumps({"metadata": {"exit_code": 2}, "output": "quiet"}))], "error"),
    ([{"type": "input_image", "text": "Error: not a text result"}], "success"),
    ([{"type": "text", "text": "quiet", "isError": True}], "error"),
])
def test_native_code_mode_outcomes(tmp_path, output, expected):
    events = [
        {"type": "session_meta", "payload": {"id": "test", "cli_version": "0.164.0"}},
        {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec",
            "call_id": "cmd", "input": 'text(await tools.exec_command({cmd: "test-command"}));'}},
        {"type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "cmd", "output": output}},
        {"type": "event_msg", "payload": {"type": "task_complete"}},
    ]
    source = tmp_path / "rollout.jsonl"
    source.write_text("\n".join(json.dumps(event) for event in events))
    run = tv.load(source)
    calls = [call for step in run.steps for call in step["tool_calls"]]
    assert len(calls) == 1
    assert calls[0]["status"] == expected
    assert sum(step["error_count"] for step in run.steps) == (expected == "error")
    if isinstance(output, list):
        assert calls[0]["output"] == output
    assert tv.Trajectory.from_dict(run.to_dict(include_raw=True)).metrics == run.metrics
