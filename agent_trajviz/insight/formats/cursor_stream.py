"""Cursor Agent CLI ``--output-format stream-json`` → Cursor trajectory.

The CLI protocol differs from the IDE's transcript mirror. Tool IDs join
started/completed events; model_call_id joins a response and its tools. Newer
CLI builds also emit thinking deltas and explicit tool execution clocks.
Token counts are estimates of captured content, never reported API usage.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

from ...collect.cursor import _estimate_step_tokens
from .cursor import _convert_cursor_to_internal

_TOOLS = {
    "readToolCall": "Read", "writeToolCall": "Write", "editToolCall": "Edit",
    "editFileToolCall": "Edit", "strReplaceToolCall": "Edit", "deleteToolCall": "Delete",
    "shellToolCall": "Bash", "grepToolCall": "Grep", "globToolCall": "Glob",
    "taskToolCall": "Task", "todoToolCall": "TodoWrite", "writeTodosToolCall": "TodoWrite",
    "getMcpToolsToolCall": "GetDynamicTools", "callMcpToolCall": "CallDynamicTool",
    "webSearchToolCall": "WebSearch", "webFetchToolCall": "WebFetch", "readLintsToolCall": "ReadLints",
}


def looks_like_cursor_stream(events: list) -> bool:
    """Require positive Cursor protocol evidence; Claude's init is similar."""
    if not isinstance(events, list):
        return False
    initialized = any(isinstance(e, dict) and e.get("type") == "system" and e.get("subtype") == "init"
                      and isinstance(e.get("session_id"), str) and e["session_id"] for e in events)
    cursor_tools = any(isinstance(e, dict) and e.get("type") == "tool_call"
                       and isinstance(e.get("tool_call"), dict) and isinstance(e.get("call_id"), str)
                       and e["call_id"] for e in events)
    return bool(initialized and cursor_tools)


def _text(value):
    return value if isinstance(value, str) else ""


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _content(event):
    message = event.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content", [])
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [deepcopy(block) for block in content if isinstance(block, dict)] if isinstance(content, list) else []


def _plain(parts):
    return "".join(_text(p.get("text")) for p in parts if p.get("type") == "text")


def _tool_payload(event):
    wrapper = event.get("tool_call")
    wrapper = wrapper if isinstance(wrapper, dict) else {}
    for key, value in wrapper.items():
        if key.endswith("ToolCall") and isinstance(value, dict):
            return wrapper, _TOOLS.get(key, key.removesuffix("ToolCall")), value
    function = wrapper.get("function")
    if isinstance(function, dict):
        return wrapper, _text(function.get("name")) or "?", function
    return wrapper, "?", wrapper


def _arguments(payload, name):
    args = payload.get("args", payload.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, RecursionError):
            args = {}
    args = deepcopy(args) if isinstance(args, dict) else {}
    if name in {"Read", "Write", "Edit", "Delete"}:
        for key in ("path", "filePath", "relativeWorkspacePath", "targetFile"):
            if key in args and "file_path" not in args:
                args["file_path"] = args[key]
    for original, normalized in (("oldString", "old_string"), ("newString", "new_string"),
                                 ("globPattern", "pattern"), ("targetDirectory", "path"),
                                 ("subagentType", "subagent_type")):
        if original in args:
            args.setdefault(normalized, args[original])
    if name == "Bash" and "command" not in args:
        for key in ("cmd", "full_command", "bash_command"):
            if key in args:
                args["command"] = args[key]
                break
    return args


def _output_text(result):
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(text for item in result if (text := _output_text(item)))
    if isinstance(result, dict):
        for key in ("interleavedOutput", "content", "output", "text", "diff", "message"):
            if key in result:
                text = _output_text(result[key])
                if text:
                    return text
        return "\n".join(text for key in ("stdout", "stderr") if (text := _text(result.get(key))))
    return ""


def _update_tool(part, event):
    wrapper, name, payload = _tool_payload(event)
    state = part["state"]
    if name != "?":
        part["tool"] = name
    if "args" in payload or "arguments" in payload:
        state["input"] = _arguments(payload, name)
    metadata = state["metadata"]
    metadata["cursor_raw_tool"] = deepcopy(wrapper)
    start = _number(wrapper.get("startedAtMs"))
    end = _number(wrapper.get("completedAtMs"))
    if start is None and event.get("subtype") == "started":
        start = _number(event.get("timestamp_ms"))
    if start is not None:
        state.setdefault("time", {})["start"] = start
    if end is None and event.get("subtype") == "completed":
        end = _number(event.get("timestamp_ms"))
    if end is not None:
        state.setdefault("time", {})["end"] = end
    span = state.get("time", {})
    if span.get("start") is not None and span.get("end") is not None and span["end"] >= span["start"]:
        metadata["timing_source"] = "cursor_cli_execution"
        metadata["totalDurationMs"] = span["end"] - span["start"]
    result = payload.get("result")
    if not isinstance(result, dict):
        return
    metadata["cursor_raw_result"] = deepcopy(result)
    outcome = next((key for key in ("failure", "error", "rejected", "cancelled", "success") if key in result), None)
    data = result.get(outcome) if outcome else result
    state["output"] = _output_text(data)
    metadata["output_available"] = bool(state["output"])
    if outcome == "success":
        state["status"] = "completed"
    elif outcome in {"failure", "error"}:
        state["status"] = "error"
        state["error"] = _output_text(data) or outcome
    elif outcome == "rejected":
        metadata["cursor_raw_status"] = outcome
        state["status"] = "failed"
        state["error"] = _output_text(data) or "Tool call rejected"
    elif outcome == "cancelled":
        state["status"] = outcome
    if isinstance(data, dict):
        exit_code = data.get("exitCode")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            metadata["exit"] = exit_code
            if exit_code:
                state["status"] = "error"
                state["error"] = state["output"] or f"Command exited with code {exit_code}"


def convert_cursor_stream(events: list, *, source_path: str | None = None) -> dict[str, Any]:
    """Convert one captured CLI session; never scan Cursor's storage or blobs."""
    records = [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []
    header = next((e for e in records if e.get("type") == "system" and e.get("subtype") == "init"), {})
    sid = _text(header.get("session_id")) or next((_text(e.get("session_id")) for e in records
                                                  if _text(e.get("session_id"))), "")
    if not sid:
        return {"_error": "Cursor CLI stream must contain a nonempty session_id.", "_error_code": "unknown_format"}
    if any(_text(e.get("session_id")) not in ("", sid) for e in records):
        return {"_error": "Cursor CLI stream contains multiple sessions; load one captured stream per session.",
                "_error_code": "ambiguous_session"}
    messages, by_model, calls = [], {}, {}
    warnings = []
    current = None
    pending_thinking, pending_deltas = [], []
    partial = any(e.get("type") == "assistant" and "timestamp_ms" in e and not e.get("model_call_id")
                  for e in records)
    terminal = next((e for e in reversed(records) if e.get("type") == "result"), None)

    def message(role, index, *, model_id="", parts=None, stamp=None):
        info = {"role": role, "id": f"{sid}:{role}:{model_id or index}", "sessionID": sid,
                "agent": "cursor", "isSubAgent": False, "modelID": _text(header.get("model")),
                "sourcePath": source_path or "", "rawIndex": index}
        if stamp is not None:
            info["time"] = {"created": stamp}
        record = {"info": info, "parts": parts or []}
        messages.append(record)
        return record

    def assistant(index, model_id="", stamp=None):
        nonlocal current, pending_thinking
        if model_id:
            current = by_model.get(model_id)
        if current is None or current["info"]["role"] != "assistant":
            current = message("assistant", index, model_id=model_id, stamp=stamp)
            if model_id:
                by_model[model_id] = current
        if pending_thinking:
            current["parts"].append({"type": "reasoning", "text": "".join(pending_thinking)})
            pending_thinking = []
        return current

    for index, event in enumerate(records):
        kind = event.get("type")
        stamp = _number(event.get("timestamp_ms"))
        model_id = _text(event.get("model_call_id"))
        if kind == "user":
            current = message("user", index, parts=_content(event), stamp=stamp)
        elif kind == "thinking":
            if event.get("subtype") == "delta" and isinstance(event.get("text"), str):
                pending_thinking.append(event["text"])
            elif event.get("subtype") == "completed" and event.get("text"):
                pending_thinking = [_text(event["text"])]
        elif kind == "assistant":
            parts = _content(event)
            if partial and not model_id and stamp is not None:
                pending_deltas.append(_plain(parts))
                continue
            if partial and not model_id and stamp is None:
                if not pending_deltas:
                    continue  # Terminal duplicate flush; result text is also cumulative.
                parts = [{"type": "text", "text": "".join(pending_deltas)}]
                current = None
            elif not model_id:
                current = None  # Older complete-message protocol has no model_call_id.
            target = assistant(index, model_id, stamp)
            pending_deltas = []
            text = _plain(parts)
            old = _plain(target["parts"])
            if text and text != old:
                if old and text.startswith(old):
                    target["parts"] = [p for p in target["parts"] if p.get("type") != "text"]
                target["parts"].extend(p for p in parts if p.get("type") == "text")
        elif kind == "tool_call":
            call_id = _text(event.get("call_id"))
            if not call_id:
                continue
            if call_id not in calls:
                target = assistant(index, model_id, stamp)
                part = {"type": "tool", "tool": "?", "callID": call_id, "id": call_id,
                        "state": {"status": "unknown", "input": {}, "output": "", "metadata": {}}}
                target["parts"].append(part)
                calls[call_id] = part
            _update_tool(calls[call_id], event)
    if pending_deltas:
        current = None
        assistant(len(records))["parts"].append({"type": "text", "text": "".join(pending_deltas)})
    if pending_thinking:
        assistant(len(records))
    if terminal is None:
        warnings.append("Cursor CLI stream has no terminal result; the captured run may be incomplete.")
    if not any(_plain(m["parts"]) for m in messages if m["info"]["role"] == "assistant") and terminal:
        text = _text(terminal.get("result"))
        if text:
            current = None
            assistant(len(records))["parts"].append({"type": "text", "text": text})
    unfinished = sum(part["state"]["status"] == "unknown" for part in calls.values())
    if unfinished:
        warnings.append(f"{unfinished} Cursor tool calls have no recorded outcome.")
    sessions = [{"id": sid, "parent_id": None, "depth": 0}]
    children = set()
    for call_id, part in calls.items():
        wrapper = part["state"]["metadata"].get("cursor_raw_tool", {})
        task = wrapper.get("taskToolCall", {})
        result = task.get("result", {}).get("success", {}) if isinstance(task, dict) else {}
        child_id = _text(result.get("agentId")) if isinstance(result, dict) else ""
        if not child_id:
            continue
        part["state"]["metadata"].update(sessionId=child_id, parentSessionId=sid)
        if child_id not in children:
            sessions.append({"id": child_id, "parent_id": sid, "depth": 1})
            children.add(child_id)
        conversation = result.get("conversationSteps", [])
        if not isinstance(conversation, list):
            continue
        for position, child_step in enumerate(conversation):
            if not isinstance(child_step, dict):
                continue
            body = child_step.get("assistantMessage")
            role = "assistant"
            if not isinstance(body, dict):
                body, role = child_step.get("userMessage"), "user"
            if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                continue
            record = message(role, len(records) + position,
                             model_id=f"{call_id}:child:{position}", parts=[{"type": "text", "text": body["text"]}])
            record["info"].update(sessionID=child_id, parentSessionID=sid, isSubAgent=True, sessionDepth=1,
                                 agent="cursor (subagent)", id=f"{child_id}:{call_id}:{position}")
            record["info"]["cursor_capture_kind"] = "subagent_summary"
    if children:
        warnings.append("Cursor CLI stream includes captured subagent summaries; collect native child logs for detailed activity.")
    for record in messages:
        tokens = _estimate_step_tokens(record["info"]["role"], record["parts"])
        reasoning_chars = sum(len(_text(p.get("text"))) for p in record["parts"] if p.get("type") == "reasoning")
        if reasoning_chars:
            reasoning = (reasoning_chars + 3) // 4
            tokens["reasoning"] = reasoning
            tokens["output"] = tokens.get("output", 0) + reasoning
            tokens["total"] = tokens.get("total", 0) + reasoning
        record["info"]["tokens"] = tokens
    raw = {
        "info": {"id": sid, "model": _text(header.get("model")), "directory": _text(header.get("cwd"))},
        "messages": messages,
        "sessions": sessions,
        "statistics": {"sessions": len(sessions), "subagent_sessions": len(children)},
        "export_metadata": {"schema_version": 1, "source_format": "cursor_composer",
                            "generator_name": "cursor_cli_stream", "token_semantics": "estimated_log_tokens",
                            "complete": terminal is not None, "warnings": warnings},
    }
    converted = _convert_cursor_to_internal(raw)
    converted["_cursor_cli_stream"] = True
    converted["_cursor_cli_events"] = deepcopy(records)
    warnings.append("Cursor CLI captures tool execution clocks but not model request intervals; generation timing is unavailable.")
    converted["_load_warnings"] = list(dict.fromkeys(converted.get("_load_warnings", []) + warnings))
    converted["metadata"]["event_count"] = len(records)
    converted["_capabilities"]["has_reasoning_content"] = any(
        p.get("type") == "reasoning" for message in messages for p in message["parts"])
    converted["_capabilities"]["has_model_timing"] = False
    duration = _number((terminal or {}).get("duration_ms"))
    if duration is not None:
        converted["timing"]["total_duration"] = duration / 1000
        converted["_capabilities"]["has_timing"] = True
    if terminal and terminal.get("is_error") is True:
        converted["output"]["error"] = _text(terminal.get("result")) or "Cursor CLI run failed"
    return converted
