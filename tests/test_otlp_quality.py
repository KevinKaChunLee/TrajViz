"""Missing telemetry is evidence about coverage, never a measured zero."""

from __future__ import annotations

from pathlib import Path

import jsonschema
import pytest

import agent_trajviz as tv
from agent_trajviz.insight import chart_data
from agent_trajviz.insight.analysis import Analysis
from agent_trajviz.insight.formats.claude_code import _cc_build_step
from agent_trajviz.insight.metrics import compute_metrics, generation_seconds
from agent_trajviz.insight.telemetry_quality import QUALITY_FIELDS, data_quality

FIXTURE = Path(__file__).parent / "fixtures" / "otlp" / "manifest.json"


def model(index=0, *, observed=None, duration=2.0):
    step = _cc_build_step([], role="assistant", agent="", session_id="session", timestamp_ms=1000 * index)
    step.update(index=index, duration=duration, time_completed_ms=1000 * index + int((duration or 0) * 1000))
    step["tokens"] = dict.fromkeys(("input", "output", "cache_read", "cache_write", "reasoning", "total"), 0)
    step["_otel"] = {
        "kind": "model_request", "fields": dict.fromkeys(QUALITY_FIELDS, "unavailable"),
        "reasons": dict.fromkeys(QUALITY_FIELDS, "not_recorded"), "trace_id": "a" * 32, "span_id": f"{index + 1:016x}",
    }
    for field, value in (observed or {}).items():
        step["_otel"]["fields"][field] = "observed"
        step["_otel"]["reasons"].pop(field, None)
        if field.startswith("tokens."):
            step["tokens"][field.removeprefix("tokens.")] = value
    return step


def run(steps):
    return tv.Trajectory("otlp", steps, raw={"_otel": {
        "profile": "claude_code", "profile_version": 1, "mapping": "operation",
        "limitations": ["Saved capture completeness cannot be established"],
    }})


def tool(*, outcome=None):
    return {
        "tool_name": "Read", "tool_id": "tool-one", "status": outcome or "unknown",
        "input": {}, "output": "", "error": None, "error_type": None,
        "duration_ms": 1000, "time_start": 1000, "time_end": 2000,
        "metadata": {"_otel": {"fields": {
            "tool_outcomes": "observed" if outcome else "unavailable",
            "tool_inputs": "unavailable", "tool_outputs": "unavailable", "duration": "observed",
        }, "reasons": {"tool_inputs": "redacted", "tool_outputs": "truncated"}}},
    }


def strict(schema):
    if isinstance(schema, dict):
        schema = {key: strict(value) for key, value in schema.items()}
        if "properties" in schema and "additionalProperties" not in schema:
            schema["additionalProperties"] = False
    elif isinstance(schema, list):
        schema = [strict(value) for value in schema]
    return schema


def test_missing_usage_never_becomes_zero_or_healthy():
    step = model()
    # Even a nonzero placeholder must be ignored without observation evidence.
    step["tokens"] = dict.fromkeys(step["tokens"], 999)
    trajectory = run([step])
    assert all(value is None for value in trajectory.metrics["tokens"].values())
    assert trajectory.metrics["output_tokens_per_sec"] is None
    assert trajectory.metrics["total_duration"] is None
    assert all(verdict["status"] == "unavailable" for verdict in trajectory.health)
    assert trajectory.view("steps").data[0]["tokens_total"] is None
    assert trajectory.view("step_analytics").data[0]["tok_total"] is None
    assert trajectory.view("token_usage").data["steps"][0]["total"] is None
    assert trajectory.view("agents").data[0]["total_tokens"] is None
    assert trajectory.view("hotspots").data["heaviest"] == []


def test_observed_zero_remains_zero_and_partial_subtotals_are_labeled():
    first = model(observed={"tokens.input": 0, "tokens.output": 0, "duration": 2})
    second = model(1, observed={"tokens.input": 10, "duration": 2})
    trajectory = run([first, second])
    assert trajectory.metrics["tokens"]["input"] == 10
    assert trajectory.metrics["tokens"]["output"] == 0
    assert trajectory.metrics["metric_coverage"]["tokens.output"] == "partial"
    assert trajectory.metrics["output_tokens_per_sec"] == 0
    assert trajectory.metrics["output_throughput_timed_steps"] == 1
    assert trajectory.metrics["output_throughput_total_steps"] == 2
    assert trajectory.view("steps").data[0]["tokens_input"] == 0
    assert trajectory.view("token_usage").data["steps"][1]["fresh_input"] == 10
    assert trajectory.metrics["output_input_ratio"] is None
    assert next(v for v in trajectory.health if v["metric"] == "Throughput")["status"] == "unavailable"


def test_unknown_tool_outcomes_are_excluded_from_the_rate():
    step = model()
    step["tool_calls"] = [tool(outcome="completed"), tool(outcome="error"), tool()]
    step["tool_call_count"] = 3
    trajectory = run([step])
    assert trajectory.metrics["tool_success"] == 1
    assert trajectory.metrics["tool_fail"] == 1
    assert trajectory.metrics["tool_success_rate"] == 50
    assert trajectory.metrics["tool_observed_outcomes"] == 2
    assert trajectory.metrics["tool_unknown_outcomes"] == 1
    assert [call["ok"] for call in trajectory.view("tool_outcomes").data["calls"]] == [True, False, None]
    assert trajectory.view("tool_outcomes").availability["status"] == "partial"
    assert next(v for v in trajectory.health if v["metric"] == "Tool Success")["status"] == "unavailable"


def test_generation_uses_model_request_duration_without_parent_or_tool_overlap():
    request = model(observed={"tokens.output": 20, "duration": 2})
    request["tool_calls"] = [tool(outcome="completed")]
    request["tool_call_count"] = 1
    prompt = model(1, observed={"duration": 20})
    prompt["role"], prompt["_otel"]["kind"] = "user", "prompt"
    standalone = model(2, observed={"duration": 10})
    standalone["role"], standalone["_otel"]["kind"] = "tool", "tool"
    trajectory = run([request, prompt, standalone])
    assert generation_seconds(request) == (2.0, 0.0)
    assert generation_seconds(standalone) is None
    assert trajectory.metrics["total_duration"] == 2
    assert trajectory.metrics["output_tokens_per_sec"] == 10
    assert trajectory.metrics["output_throughput_total_steps"] == 1


def test_attached_agent_call_is_not_subtracted_from_model_operation_chart():
    request = model(observed={"tokens.output": 20, "duration": 2})
    delegation = tool(outcome="completed")
    delegation.update(tool_name="Agent", duration_ms=4000, time_end=5000)
    request["tool_calls"], request["tool_call_count"] = [delegation], 1
    trajectory = run([request])
    row = trajectory.view("step_durations").data["steps"][0]
    assert row["duration_s"] == 2
    assert row["spawn_adjusted"] is False
    assert trajectory.metrics["output_tokens_per_sec"] == 10
    assert trajectory.metrics["delegation_time_total"] == 4


def test_incomplete_content_gates_detectors_and_preserves_reasons():
    step = model()
    call = tool()
    call["input"] = {"file_path": "sensitive.py"}
    step["tool_calls"], step["tool_call_count"] = [call], 1
    analysis = run([step])._analysis
    assert analysis.file_interactions == []
    assert analysis.repeated_searches == []
    assert analysis.plan_history == []
    assert analysis.premature_compactions == []
    assert analysis.failure_patterns == []
    assert analysis.diagnostic_metrics["edit_precision"] is None
    assert analysis.data_quality["fields"]["tool_inputs"]["reasons"] == {"redacted": 1}
    assert analysis.data_quality["capture_complete"] is None


def test_unobserved_tool_duration_cannot_leak_placeholder_times():
    step = model()
    call = tool(outcome="completed")
    call["metadata"]["_otel"]["fields"]["duration"] = "unavailable"
    step["tool_calls"], step["tool_call_count"] = [call], 1
    trajectory = run([step])
    assert trajectory.metrics["tool_time_total"] is None
    assert trajectory.view("tool_durations").data["tools"] == []
    assert trajectory.view("steps").data[0]["tool_time_sum"] is None
    assert trajectory.view("step_analytics").data[0]["tool_time_ms"] is None


def test_complete_token_totals_cannot_use_partial_duration_denominator():
    observed = {f"tokens.{key}": value for key, value in {
        "input": 10, "output": 5, "cache_read": 20, "cache_write": 2, "total": 37,
    }.items()}
    trajectory = run([model(observed={**observed, "duration": 2}), model(1, observed=observed, duration=None)])
    assert trajectory.metrics["tokens"]["total"] == 74
    assert trajectory.metrics["metric_coverage"]["duration"] == "partial"
    assert trajectory.metrics["tokens_per_second"] is None
    assert trajectory.metrics["output_tokens_per_sec"] == 2.5


def test_captured_response_text_does_not_prove_no_thinking_or_patch_blocks():
    trajectory = tv.load(FIXTURE)
    for key in ("reasoning_parts", "patch_steps", "has_patch", "patch_lines"):
        assert trajectory.metrics[key] is None
    for row in trajectory.view("steps").data:
        assert row["reasoning_parts"] is None
        assert row["patch_parts"] is None


def test_partial_and_unknown_coverage_remain_distinct():
    step = model()
    step["_otel"]["fields"]["text"] = "partial"
    step["_otel"]["reasons"]["text"] = "truncated"
    step["_otel"]["fields"].pop("context")
    quality = data_quality([step])
    assert quality["fields"]["text"]["status"] == "partial"
    assert quality["fields"]["text"]["partial"] == 1
    assert quality["fields"]["context"]["status"] == "unknown"


@pytest.mark.parametrize("kind", ["missing", "partial", "full_fixture"])
def test_every_otel_view_satisfies_schema_v2_and_roundtrips(kind):
    if kind == "full_fixture":
        trajectory = tv.load(FIXTURE)
    else:
        first = model(observed={"tokens.input": 10, "duration": 2} if kind == "partial" else {})
        first["tool_calls"], first["tool_call_count"] = [tool()], 1
        trajectory = run([first, model(1)])
    assert tv.VIEWS_SCHEMA_VERSION == 2
    assert trajectory.to_dict()["schema_version"] == tv.SCHEMA_VERSION == 1
    errors = []
    for view in trajectory.views().values():
        validator = jsonschema.Draft202012Validator(strict(tv.view_schema(view.id)))
        errors.extend(f"{view.id}: {'/'.join(map(str, error.absolute_path))}: {error.message}"
                      for error in validator.iter_errors(view.to_dict()))
    assert errors == []
    restored = tv.Trajectory.from_dict(trajectory.to_dict())
    assert restored.view("data_quality").data == trajectory.view("data_quality").data
    assert restored.metrics == trajectory.metrics


def test_native_metrics_remain_unchanged_without_telemetry_evidence():
    step = model(observed={"tokens.input": 10, "tokens.output": 5, "tokens.total": 15})
    step.pop("_otel")
    metrics = compute_metrics([step], {})
    assert metrics["tokens"]["total"] == 15
    assert metrics["tokens"]["reasoning"] == 0
    assert "data_quality" not in metrics
    assert Analysis([step], {}, "claude_code").data_quality["source"] == "native"
    assert chart_data.token_usage([step])["steps"][0]["total"] == 15
