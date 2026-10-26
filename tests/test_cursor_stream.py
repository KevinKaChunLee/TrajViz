"""Cursor CLI protocol regressions, based on the 2026.10.22 live capture."""

from __future__ import annotations

import json
from copy import deepcopy

import jsonschema
import pytest

import agent_trajviz as tv
from agent_trajviz.insight.formats.cursor_stream import convert_cursor_stream, looks_like_cursor_stream
from agent_trajviz.insight.loaders import detect_format
from agent_trajviz.insight.parser import parse_steps

SID = "cursor-audit-session"


def header():
    return {"type": "system", "subtype": "init", "session_id": SID, "cwd": "/workspace",
            "model": "fixture-model", "apiKeySource": "login", "permissionMode": "default"}


def assistant(text, model="model-1", stamp=1100):
    event = {"type": "assistant", "session_id": SID,
             "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}
    if model is not None:
        event["model_call_id"] = model
    if stamp is not None:
        event["timestamp_ms"] = stamp
    return event


def tool(subtype, *, call_id="read-1", model="model-1", name="readToolCall", args=None, result=None):
    payload = {"args": args if args is not None else {"path": "parser.py"}}
    if result is not None:
        payload["result"] = result
    wrapper = {name: payload, "toolCallId": call_id, "startedAtMs": 1200}
    if subtype == "completed":
        wrapper["completedAtMs"] = 1500
    return {"type": "tool_call", "subtype": subtype, "session_id": SID, "call_id": call_id,
            "model_call_id": model, "timestamp_ms": 1500 if subtype == "completed" else 1200,
            "tool_call": wrapper}


def terminal():
    return {"type": "result", "subtype": "success", "session_id": SID,
            "is_error": False, "duration_ms": 2000, "result": "I will inspect. Done."}


def events():
    return [header(), {"type": "user", "session_id": SID, "message": {
        "role": "user", "content": [{"type": "text", "text": "Inspect the parser."}]}},
        {"type": "thinking", "subtype": "delta", "session_id": SID, "text": "Check the parser.", "timestamp_ms": 1000},
        {"type": "thinking", "subtype": "completed", "session_id": SID, "text": "", "timestamp_ms": 1050},
        assistant("I will inspect."), tool("started"),
        tool("completed", result={"success": {"content": "def parse(): pass"}}),
        assistant("Done.", "model-2", 1800), terminal()]


def text(step):
    return "".join(p.get("text", "") for p in step["parts"] if p.get("type") == "text")


@pytest.mark.parametrize("extension", [".jsonl", ".json"])
def test_public_native_stream_loading_preserves_messages_tools_and_estimate_labels(tmp_path, extension):
    path = tmp_path / ("cursor" + extension)
    path.write_text("".join(json.dumps(e) + "\n" for e in events()) if extension == ".jsonl"
                    else json.dumps(events()))
    run = tv.load(path)
    assert run.format == "cursor" and run.tokens_estimated is True
    assert [step["role"] for step in run.steps] == ["user", "assistant", "assistant"]
    assert text(run.steps[1]) == "I will inspect." and text(run.steps[2]) == "Done."
    call, = run.steps[1]["tool_calls"]
    assert call["tool_name"] == "Read" and call["input"]["file_path"] == "parser.py"
    assert call["output"] == "def parse(): pass" and call["status"] == "completed"
    assert call["duration_ms"] == 300 and call["time_start"] == 1200 and call["time_end"] == 1500
    assert run.steps[1]["duration"] is None  # No fabricated model generation interval.
    assert run.steps[1]["has_reasoning"] is True
    assert run.raw["metadata"]["token_semantics"] == "estimated_log_tokens"
    assert run.raw["_capabilities"]["has_runtime_token_usage"] is False
    assert run.raw["timing"]["total_duration"] == 2.0
    for view_id, view in run.views().items():
        jsonschema.validate(view.to_dict(), tv.view_schema(view_id))
    restored = tv.Trajectory.from_dict(run.to_dict(include_raw=True))
    assert {k: v.to_dict() for k, v in restored.views().items()} == {
        k: v.to_dict() for k, v in run.views().items()}


def test_cursor_discriminator_does_not_claim_claude_cli_or_generic_chat():
    assert looks_like_cursor_stream(events()) is True
    assert detect_format(events()) == "cursor"
    assert looks_like_cursor_stream([header(), assistant("Hello"), terminal()]) is False
    claude = [header(), {"type": "assistant", "session_id": SID, "message": {
        "id": "msg-fixture", "type": "message", "model": "claude-fixture", "content": []}}]
    assert looks_like_cursor_stream(claude) is False
    assert looks_like_cursor_stream([{"role": "user", "content": "Hello"}]) is False


def test_model_call_identifies_one_step_and_replayed_tools_are_not_added_twice():
    stream = events()
    stream.insert(7, deepcopy(stream[6]))
    stream.insert(8, deepcopy(stream[4]))
    steps = parse_steps(convert_cursor_stream(stream))
    assert len(steps) == 3
    assert sum(step["tool_call_count"] for step in steps) == 1
    assert text(steps[1]) == "I will inspect."


def test_older_unkeyed_complete_messages_remain_separate_between_tools():
    stream = [header(), assistant("First", None, None),
              {k: v for k, v in tool("started").items() if k != "model_call_id"},
              {k: v for k, v in tool("completed", result={"success": {"content": "ok"}}).items()
               if k != "model_call_id"}, assistant("Second", None, None), terminal()]
    steps = parse_steps(convert_cursor_stream(stream))
    assert [text(step) for step in steps] == ["First", "Second"]
    assert [step["tool_call_count"] for step in steps] == [1, 0]


def test_partial_deltas_buffered_flush_and_terminal_flush_count_text_once():
    stream = [header(), assistant("Hel", None, 1000), assistant("lo", None, 1001),
              assistant("Hello", "model-1", 1002), tool("started"),
              tool("completed", result={"success": {"content": "ok"}}),
              assistant("Done", None, 1600), assistant("Done", None, None), terminal()]
    steps = parse_steps(convert_cursor_stream(stream))
    assert [text(step) for step in steps] == ["Hello", "Done"]
    assert sum(step["tool_call_count"] for step in steps) == 1


def test_thinking_deltas_are_retained_and_completed_snapshot_is_not_duplicated():
    stream = [header(),
              {"type": "thinking", "subtype": "delta", "session_id": SID, "text": "Plan "},
              {"type": "thinking", "subtype": "delta", "session_id": SID, "text": "first."},
              {"type": "thinking", "subtype": "completed", "session_id": SID, "text": "Plan first."},
              assistant("Done"), terminal()]
    raw = convert_cursor_stream(stream)
    step, = parse_steps(raw)
    assert [p["text"] for p in step["parts"] if p["type"] == "reasoning"] == ["Plan first."]
    assert step["tokens"]["reasoning"] == 3
    assert step["tokens"]["output"] >= 3


@pytest.mark.parametrize("result,status", [
    ({"success": {"stdout": "", "stderr": "failed", "exitCode": 2}}, "error"),
    ({"failure": {"message": "not found"}}, "error"),
    ({"error": {"message": "execution failed"}}, "error"),
    ({"rejected": {"message": "denied"}}, "failed"),
    ({"cancelled": {}}, "cancelled"),
])
def test_explicit_tool_outcomes_and_shell_exit_codes_are_preserved(result, status):
    raw = convert_cursor_stream([header(), tool("started", name="shellToolCall", args={"command": "false"}),
                                 tool("completed", name="shellToolCall", args={"command": "false"}, result=result), terminal()])
    call = parse_steps(raw)[0]["tool_calls"][0]
    assert call["status"] == status
    if "success" in result:
        assert call["metadata"]["exit"] == 2
        assert call["output"] == "failed"
    if "rejected" in result:
        assert call["metadata"]["cursor_raw_status"] == "rejected"
        assert call["error"] == "denied"


def test_missing_tool_result_is_unknown_and_blob_reference_is_not_content():
    raw = convert_cursor_stream([header(), tool("started"),
                                 tool("completed", result={"success": {"contentBlobId": "blob-fixture", "totalLines": 10}})])
    step, = parse_steps(raw)
    assert step["tool_calls"][0]["output"] == ""
    assert step["tool_calls"][0]["metadata"]["output_available"] is False
    assert any("no terminal result" in warning for warning in raw["_load_warnings"])
    raw = convert_cursor_stream([header(), tool("started")])
    assert parse_steps(raw)[0]["tool_calls"][0]["status"] == "unknown"
    assert any("no recorded outcome" in warning for warning in raw["_load_warnings"])


def test_completion_without_repeated_arguments_keeps_initial_input():
    completed = tool("completed", result={"success": {"content": "ok"}})
    del completed["tool_call"]["readToolCall"]["args"]
    step = parse_steps(convert_cursor_stream([header(), tool("started"), completed, terminal()]))[0]
    assert step["tool_calls"][0]["input"]["file_path"] == "parser.py"


def test_explicit_subagent_summary_has_parent_identity_and_partial_warning():
    result = {"success": {"agentId": "reviewer", "conversationSteps": [
        {"assistantMessage": {"text": "Review complete."}}], "durationMs": 5000}}
    stream = [header(), tool("started", name="taskToolCall", args={"prompt": "Review"}),
              tool("completed", name="taskToolCall", args={"prompt": "Review"}, result=result), terminal()]
    raw = convert_cursor_stream(stream)
    steps = parse_steps(raw)
    child = next(step for step in steps if step["is_sub_agent"])
    assert child["session_id"] == "reviewer" and child["parent_session_id"] == SID
    assert child["duration"] is None
    assert text(child) == "Review complete."
    assert raw["metadata"]["session_count"] == 2
    assert raw["metadata"]["sub_agent_count"] == 1
    assert steps[0]["tool_calls"][0]["metadata"]["sessionId"] == "reviewer"
    assert any("summaries" in warning for warning in raw["_load_warnings"])


def test_multiple_unrelated_sessions_are_not_silently_merged():
    raw = convert_cursor_stream([header(), {**assistant("Other"), "session_id": "different"}])
    assert raw["_error_code"] == "ambiguous_session"


def test_function_payload_and_json_arguments_support_extension_tools():
    event = {"type": "tool_call", "subtype": "completed", "call_id": "function-1", "session_id": SID,
             "tool_call": {"function": {"name": "Bash", "arguments": '{"command":"pwd"}',
                                         "result": {"success": {"stdout": "/workspace", "exitCode": 0}}}}}
    step = parse_steps(convert_cursor_stream([header(), event, terminal()]))[0]
    call, = step["tool_calls"]
    assert call["tool_name"] == "Bash" and call["input"]["command"] == "pwd"
    assert call["output"] == "/workspace" and call["status"] == "completed"
