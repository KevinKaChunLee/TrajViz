"""Claude OTLP mapping: explicit joins, graph evidence and unknown values."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_trajviz.errors import LoadError
from agent_trajviz.insight.formats.otlp_claude import convert_otlp_claude, list_otlp_sessions


TRACE = "1" * 32
OTHER_TRACE = "2" * 32
BASE = 1_760_000_000_000_000_001


def span(kind, sid, *, trace=TRACE, parent="", session="session-one", start=0, end=1, **attrs):
    return {
        "trace_id": trace, "span_id": sid, "parent_span_id": parent,
        "name": f"claude_code.{kind}", "start_time_unix_nano": BASE + start * 1_000_000_000,
        "end_time_unix_nano": BASE + end * 1_000_000_000,
        "attributes": {**({"session.id": session} if session else {}), **attrs},
        "events": [], "status": {"code": 0, "message": ""},
        "resource": {"service.name": "claude-code"}, "scope": {"name": "claude-code"},
        "schema_url": "", "raw": {}, "source": {"path": "traces.jsonl", "line": 1, "index": 0},
    }


def log(name, *, trace=TRACE, sid="", session="session-one", **attrs):
    return {"trace_id": trace, "span_id": sid, "attributes": {
        "event.name": name, **({"session.id": session} if session else {}), **attrs},
        "body": f"claude_code.{name}", "time_unix_nano": BASE, "observed_time_unix_nano": BASE,
        "resource": {"service.name": "claude-code"}, "scope": {}, "raw": {},
        "source": {"path": "logs.jsonl", "line": 1, "index": 0}}


def capture(spans, logs=()):
    return SimpleNamespace(spans=spans, logs=list(logs), raw={"source_files": ["traces.jsonl"]},
                           warnings=[], sha256="abc123")


def steps(cap, **kwargs):
    return convert_otlp_claude(cap, **kwargs)["_otlp_parsed_steps"]


def usage(**extra):
    return {"input_tokens": 10, "output_tokens": 5, "cache_read_tokens": 20,
            "cache_creation_tokens": 2, **extra}


def test_multiple_prompt_traces_group_by_session_not_cloud_umbrella():
    cap = capture([
        span("interaction", "a", **{"ccr.session.id": "cloud-umbrella"}),
        span("interaction", "b", trace=OTHER_TRACE, start=2, end=3,
             **{"ccr.session.id": "cloud-umbrella"}),
        span("interaction", "c", trace="3" * 32, session="session-two",
             **{"ccr.session.id": "cloud-umbrella"}),
    ])
    sessions = list_otlp_sessions(cap)
    assert [item["id"] for item in sessions] == ["session-one", "session-two"]
    assert sessions[0]["trace_ids"] == [TRACE, OTHER_TRACE]
    with pytest.raises(LoadError) as caught:
        convert_otlp_claude(cap)
    assert caught.value.code == "ambiguous_session"
    assert caught.value.sessions == sessions
    assert len(steps(cap, session_id="session-one")) == 2
    assert len(steps(cap, session_id="session-two")) == 1


def test_missing_session_is_separate_trace_not_time_or_resource_group():
    cap = capture([span("interaction", "a", session=""),
                   span("interaction", "b", trace=OTHER_TRACE, session="")])
    assert [item["id"] for item in list_otlp_sessions(cap)] == [f"trace:{TRACE}", f"trace:{OTHER_TRACE}"]
    result = convert_otlp_claude(cap, session_id=f"trace:{TRACE}")
    assert result["_otel"]["incomplete"] is True
    assert "No session.id" in result["_load_warnings"][0]


def test_session_inherited_from_parent_without_dropping_graph_records():
    cap = capture([span("interaction", "root"),
                   span("llm_request", "model", parent="root", session="", **usage()),
                   {**span("http", "http", parent="model", session=""), "name": "HTTP POST"}])
    result = convert_otlp_claude(cap)
    assert len(result["_otlp_parsed_steps"]) == 2
    assert len(result["_otel"]["spans"]) == 3


def test_span_tokens_and_duplicate_api_log_are_counted_once():
    cap = capture([span("llm_request", "model", request_id="req-1", **usage())],
                  [log("api_request", request_id="req-1", **usage())])
    result = convert_otlp_claude(cap)
    step = result["_otlp_parsed_steps"][0]
    assert step["tokens"] == {"total": 37, "input": 10, "output": 5, "cache_read": 20, "cache_write": 2}
    assert result["token_usage"]["total_tokens"] == 37
    assert step["_otel"]["fields"]["tokens.reasoning"] == "unavailable"
    assert step["_otel"]["fields"]["tokens.total"] == "observed"


def test_correlated_logs_fill_missing_usage_and_response_without_new_model_step():
    cap = capture([span("llm_request", "model", request_id="req-1")], [
        log("api_request", request_id="req-1", **usage()),
        log("assistant_response", request_id="req-1", response="The parser is fixed."),
        log("assistant_response", request_id="different", response="Must not be attached."),
    ])
    step, = steps(cap)
    assert step["tokens"]["total"] == 37
    assert step["text_preview"] == "The parser is fixed."
    assert step["_otel"]["fields"]["text"] == "observed"


@pytest.mark.parametrize("value", [None, -1, True, "invalid", float("inf")])
def test_missing_or_invalid_cache_fields_do_not_invent_measured_zero_total(value):
    step, = steps(capture([span("llm_request", "model", input_tokens=10, output_tokens=5,
                                cache_read_tokens=value)]))
    assert step["_otel"]["fields"]["tokens.input"] == "observed"
    assert step["_otel"]["fields"]["tokens.cache_read"] == "unavailable"
    assert step["_otel"]["fields"]["tokens.cache_write"] == "unavailable"
    assert step["_otel"]["fields"]["tokens.total"] == "unavailable"


def test_empty_redacted_and_truncated_content_are_distinct():
    result = steps(capture([
        span("interaction", "empty", user_prompt=""),
        span("interaction", "redacted", user_prompt="<REDACTED>"),
        span("interaction", "truncated", user_prompt="Known prefix", user_prompt_truncated=True),
    ]))
    by_span = {s["_otel"]["span_id"]: s for s in result}
    assert by_span["empty"]["_otel"]["fields"]["text"] == "observed"
    assert by_span["redacted"]["parts"] == []
    assert by_span["redacted"]["_otel"]["reasons"]["text"] == "redacted"
    assert by_span["truncated"]["_otel"]["fields"]["text"] == "partial"


def test_tool_execution_wait_and_wall_time_do_not_double_count():
    tool = span("tool", "tool", start=2, end=9, tool_name="Bash", tool_use_id="call-1",
                tool_input='{"command":"pwd"}')
    tool["events"] = [{"name": "tool.output", "time_unix_nano": BASE + 9_000_000_000,
                       "attributes": {"output": "/workspace"}}]
    cap = capture([tool,
                   span("tool.blocked_on_user", "wait", parent="tool", start=2, end=6, decision="accept"),
                   span("tool.execution", "exec", parent="tool", start=6, end=9, success="true")])
    step, = steps(cap)
    call, = step["tool_calls"]
    assert call["duration_ms"] == 3000
    assert call["time_end"] - call["time_start"] == 3000
    assert "time_created" not in call and "time_updated" not in call
    assert call["metadata"]["timing_source"] == "otlp_execution"
    assert call["metadata"]["_otel"]["wall_duration_ms"] == 7000
    assert call["metadata"]["_otel"]["permission_wait_ms"] == 4000
    assert call["status"] == "success" and call["output"] == "/workspace"
    assert call["input"] == {"command": "pwd"}


def test_parent_unset_and_false_string_do_not_claim_success():
    unknown = span("tool", "unknown", tool_name="Read", tool_use_id="read")
    failed = span("tool", "failed", tool_name="Bash", tool_use_id="bash")
    result = steps(capture([unknown, failed,
                           span("tool.execution", "exec", parent="failed", success="false", error="ShellError")]))
    calls = {c["tool_id"]: c for s in result for c in s["tool_calls"]}
    assert calls["read"]["status"] == "unknown"
    assert calls["read"]["metadata"]["_otel"]["fields"]["tool_outcomes"] == "unavailable"
    assert calls["bash"]["status"] == "error"
    assert calls["bash"]["error"] == "ShellError"


def test_tool_joins_only_explicit_request_not_nearest_model():
    cap = capture([
        span("llm_request", "model-1", start=0, end=2, request_id="req-1", **usage()),
        span("llm_request", "model-2", start=3, end=4, request_id="req-2", **usage()),
        span("tool", "tool-1", start=5, end=6, tool_name="Read", tool_use_id="call-1"),
        span("tool", "tool-2", start=6, end=7, tool_name="Bash", tool_use_id="call-2"),
    ], [log("tool_result", tool_use_id="call-1", request_id="req-1", success="true", duration_ms=1000)])
    result = steps(cap)
    assert len(result) == 3
    by_span = {step["_otel"]["span_id"]: step for step in result}
    assert by_span["model-1"]["tool_calls"][0]["tool_id"] == "call-1"
    assert by_span["model-1"]["_otel"]["fields"]["tool_outcomes"] == "observed"
    assert by_span["model-2"]["tool_calls"] == []
    assert by_span["tool-2"]["role"] == "tool"


def test_agent_ids_are_not_prefix_normalized_and_graph_sets_parent():
    cap = capture([
        span("interaction", "root"),
        span("tool", "spawn", parent="root", tool_name="Agent", tool_use_id="spawn-child"),
        span("llm_request", "child", parent="spawn", agent_id="agent-child", **usage()),
        span("tool", "nested-spawn", parent="child", agent_id="agent-child", tool_name="Agent"),
        span("llm_request", "nested", parent="nested-spawn", agent_id="child", **usage()),
        span("tool", "nested-tool", parent="nested", tool_name="Read"),
    ])
    result = convert_otlp_claude(cap)
    by_span = {s["_otel"]["span_id"]: s for s in result["_otlp_parsed_steps"]}
    assert by_span["child"]["agent"] == "agent-child"
    assert by_span["nested"]["agent"] == "child"
    assert by_span["nested"]["parent_agent_id"] == "agent-child"
    assert by_span["nested-tool"]["agent"] == "child"
    assert by_span["nested"]["session_depth"] == 2
    child = next(a for a in result["_otlp_sub_agents"] if a["agent_id"] == "agent-child")
    assert child["spawned_by"] == "spawn-child"


def test_missing_parent_reports_incomplete_and_nanoseconds_survive():
    cap = capture([span("llm_request", "model", parent="not-exported", **usage())])
    original = deepcopy(cap.spans)
    result = convert_otlp_claude(cap)
    assert result["_otel"]["incomplete"] is True
    assert "parents absent" in result["_load_warnings"][0]
    assert result["_otel"]["spans"][0]["start_time_unix_nano"] == BASE
    assert result["_source_sha256"] == "abc123"
    assert cap.spans == original


def test_retries_and_ttft_are_metadata_not_extra_requests():
    model = span("llm_request", "model", request_id="req-1", attempt=3, ttft_ms=500,
                 duration_ms=1000, first_content_ms=700, **usage())
    model["events"] = [{"name": "gen_ai.request.attempt", "time_unix_nano": BASE,
                        "attributes": {"attempt": 1, "client_request_id": "client-1"}}]
    step, = steps(capture([model]))
    assert step["_otel"]["attempts"] == 3
    assert step["_otel"]["ttft_ms"] == 500
    assert step["_otel"]["first_content_ms"] == 700
    assert len(step["_otel"]["retry_events"]) == 1
    assert step["tokens"]["total"] == 37


@pytest.mark.parametrize("cap,code", [
    (capture([], [log("api_request", **usage())]), "missing_traces"),
    (capture([{**span("other", "x"), "name": "generic.agent"}]), "unsupported_producer"),
    (capture([span("hook", "hook")]), "no_steps"),
])
def test_unsupported_capture_has_actionable_error(cap, code):
    with pytest.raises(LoadError) as caught:
        convert_otlp_claude(cap)
    assert caught.value.code == code


def test_unknown_session_is_not_silently_replaced():
    with pytest.raises(LoadError) as caught:
        convert_otlp_claude(capture([span("interaction", "root")]), session_id="unknown")
    assert caught.value.code == "session_not_found"


def test_sanitized_saved_fixture_matches_shared_native_facts():
    import agent_trajviz as tv
    from agent_trajviz.insight.formats.otlp_files import read_otlp_capture

    fixture = Path(__file__).parent / "fixtures" / "otlp"
    cap = read_otlp_capture(fixture)
    result = convert_otlp_claude(cap)
    native = tv.load(fixture / "native" / "demo-session.jsonl")
    assert len(cap.spans) == 13 and len(cap.logs) == 5
    assert result["token_usage"]["total_tokens"] == native.metrics["tokens"]["total"] == 57
    assert result["stats"]["total_tool_calls"] == native.metrics["tool_call_count"] == 3
    assert {a["agent_id"] for a in result["_otlp_sub_agents"]} == {"agent-child"}
    calls = {call["tool_id"]: call for s in result["_otlp_parsed_steps"] for call in s["tool_calls"]}
    assert calls["call-bash"]["duration_ms"] == 2000
    assert calls["call-bash"]["metadata"]["_otel"]["permission_wait_ms"] == 2000
    assert calls["call-read"]["output"] == "def parse(): pass"
    saved = tv.load(fixture)
    assert [row["errors"] for row in saved.view("steps").data if row["tool_calls"]] == [0, 0]


def test_correlations_do_not_cross_traces_when_producer_ids_repeat():
    cap = capture([
        span("llm_request", "model", request_id="same-request", **usage()),
        span("llm_request", "model", trace=OTHER_TRACE, request_id="same-request"),
        span("tool", "tool", tool_name="Read", tool_use_id="same-call"),
        span("tool", "tool", trace=OTHER_TRACE, tool_name="Read", tool_use_id="same-call"),
    ], [log("api_request", request_id="same-request", **usage()),
        log("tool_result", tool_use_id="same-call", success="false")])
    result = steps(cap)
    other_model = next(s for s in result if s["_otel"]["trace_id"] == OTHER_TRACE
                       and s["_otel"]["kind"] == "model_request")
    assert other_model["_otel"]["fields"]["tokens.input"] == "unavailable"
    other_tool = next(s for s in result if s["_otel"]["trace_id"] == OTHER_TRACE
                      and s["_otel"]["kind"] == "tool")
    assert other_tool["tool_calls"][0]["status"] == "unknown"


def test_conflicting_span_log_usage_is_disclosed_without_adding_counts():
    cap = capture([span("llm_request", "model", request_id="req-1", **usage())],
                  [log("api_request", request_id="req-1", **usage(input_tokens=99))])
    result = convert_otlp_claude(cap)
    assert result["token_usage"]["total_tokens"] == 37
    assert result["_otlp_parsed_steps"][0]["_otel"]["conflicts"] == ["input_tokens"]
    assert any("Conflicting span/log usage" in warning for warning in result["_load_warnings"])


def test_selected_session_export_does_not_embed_other_session_requests():
    cap = capture([span("interaction", "a", user_prompt="selected"),
                   span("interaction", "b", trace=OTHER_TRACE, session="private-session",
                        user_prompt="unrelated private prompt")])
    cap.raw["requests"] = [{"unrelated": "unrelated private prompt"}]
    result = convert_otlp_claude(cap, session_id="session-one")
    assert "requests" not in result["_otel"]["source"]
    assert {s["attributes"]["session.id"] for s in result["_otel"]["spans"]} == {"session-one"}


def test_reversed_deep_graph_inherits_session_and_agent_without_recursion():
    depth = 1500
    records = [span("interaction", "root", agent_id="child")]
    records += [span("llm_request", f"span-{i}", parent="root" if i == 1 else f"span-{i - 1}",
                     session="", **usage()) for i in range(1, depth)]
    result = convert_otlp_claude(capture(list(reversed(records))))
    assert len(result["_otlp_parsed_steps"]) == depth
    assert {s["agent"] for s in result["_otlp_parsed_steps"]} == {"child"}


def test_reported_duration_survives_missing_clock_bounds_without_inventing_bounds():
    model = span("llm_request", "model", duration_ms=1234, **usage())
    tool = span("tool", "tool", tool_name="Read", tool_use_id="call")
    execution = span("tool.execution", "exec", parent="tool", success=True, duration_ms=300)
    wait = span("tool.blocked_on_user", "wait", parent="tool")
    for record in (model, tool, execution, wait):
        record["start_time_unix_nano"] = record["end_time_unix_nano"] = 0
    result = steps(capture([model, tool, execution, wait]))
    model_step = next(s for s in result if s["role"] == "assistant")
    call = next(call for s in result for call in s["tool_calls"])
    assert model_step["duration"] == 1.234
    assert model_step["time_created_ms"] is None and model_step["time_completed_ms"] is None
    assert model_step["_otel"]["fields"]["duration"] == "observed"
    assert call["duration_ms"] == 300
    assert call["time_start"] is None and call["time_end"] is None
    assert call["metadata"]["_otel"]["permission_wait_ms"] is None


@pytest.mark.parametrize("value", [1.5, "1.5", True, "9" * 4000, 1e308, str(2**63)])
def test_invalid_saved_token_counts_remain_unavailable_and_all_views_validate(tmp_path, value):
    import json

    import jsonschema

    import agent_trajviz as tv

    fixture = Path(__file__).parent / "fixtures" / "otlp" / "claude_code_traces.jsonl"
    requests = [json.loads(line) for line in fixture.read_text().splitlines()]
    for request in requests:
        for resource in request["resourceSpans"]:
            for scope in resource["scopeSpans"]:
                for item in scope["spans"]:
                    if item["name"] != "claude_code.llm_request":
                        continue
                    for attribute in item["attributes"]:
                        if attribute["key"] == "input_tokens":
                            kind = "boolValue" if isinstance(value, bool) else (
                                "doubleValue" if isinstance(value, float) else "stringValue")
                            attribute["value"] = {kind: value}
    path = tmp_path / "invalid-counts.jsonl"
    path.write_text("".join(json.dumps(request) + "\n" for request in requests))
    run = tv.load(path)
    assert run.metrics["tokens"]["input"] is None
    assert run.metrics["tokens"]["total"] is None
    for step in run.steps:
        if step["_otel"]["kind"] == "model_request":
            assert step["_otel"]["fields"]["tokens.input"] == "unavailable"
            assert step["_otel"]["reasons"]["tokens.input"] == "invalid"
    for view_id, view in run.views().items():
        jsonschema.validate(view.to_dict(), tv.view_schema(view_id))


def test_integral_double_counts_normalize_to_integers_without_rounding_durations():
    model = span("llm_request", "model", **usage(input_tokens=10.0), duration_ms=1234.5)
    model["start_time_unix_nano"] = model["end_time_unix_nano"] = 0
    step, = steps(capture([model]))
    assert step["tokens"]["input"] == 10 and isinstance(step["tokens"]["input"], int)
    assert step["tokens"]["total"] == 37 and isinstance(step["tokens"]["total"], int)
    assert step["duration"] == 1.2345
