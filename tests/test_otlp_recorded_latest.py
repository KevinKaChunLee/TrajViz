"""Sanitized excerpts recorded from Claude Code 2.1.310 / Collector 0.163.2.

The source captures were independently matched against native parent/reviewer
sessions. Excerpts preserve recorded field shapes and timing; identities, counts
and all captured content use controlled replacements (see provenance.json).
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

import agent_trajviz as tv

FIXTURES = Path(__file__).parent / "fixtures"
REDACTED = FIXTURES / "otlp_claude_2_1_310"
CONTENT = FIXTURES / "otlp_claude_2_1_310_content"


def test_recorded_redaction_and_missing_content_are_distinct_from_observed_zero():
    run = tv.load(REDACTED)
    assert run.format == "otlp"
    assert run.metrics["tokens"] == {
        "total": 36, "input": 11, "output": 7, "reasoning": None, "cache_read": 13, "cache_write": 5,
    }
    assert len(run.steps) == 2
    assert run.metrics["tool_call_count"] == 1
    assert run.metrics["tool_success"] == 1
    quality = run.view("data_quality").data["fields"]
    assert quality["text"]["status"] == "unavailable"
    assert quality["text"]["reasons"] == {"redacted": 2}
    assert quality["tool_inputs"]["reasons"] == {"not_recorded": 1}
    assert quality["tool_outputs"]["reasons"] == {"not_recorded": 1}
    assert quality["tokens.reasoning"]["status"] == "unavailable"
    model = next(step for step in run.steps if step["role"] == "assistant")
    assert model["duration"] == pytest.approx(23.382564417)
    assert model["_otel"]["end_time_unix_nano"] - model["_otel"]["start_time_unix_nano"] == 23_382_564_417
    call = model["tool_calls"][0]
    assert call["duration_ms"] == pytest.approx(775.868792)
    assert run.view("tool_durations").data["tools"][0]["total_s"] == pytest.approx(0.775868792)
    assert not run.warnings


def test_recorded_reviewer_scope_content_and_usage_are_preserved_once():
    run = tv.load(CONTENT)
    assert len(run.steps) == 5
    assert run.metrics["tokens"] == {
        "total": 144, "input": 44, "output": 28, "reasoning": None, "cache_read": 52, "cache_write": 20,
    }
    assert run.metrics["tool_call_count"] == 5
    assert run.metrics["tool_success"] == 5
    assert len(run.agents) == 2
    assert len(run.raw["_otel"]["spans"]) == 20
    assert len(run.raw["_otel"]["logs"]) == 17
    assert run.raw["metadata"]["sub_agent_count"] == 1
    reviewer = run.raw["_otel"]["agents"][0]
    assert reviewer["agent_id"] == "recorded-reviewer"
    assert reviewer["spawned_by"]
    lanes = run.view("agent_tokens").data["agents"]
    assert {row["agent_id"]: row["total"] for row in lanes} == {"": 72, "recorded-reviewer": 72}
    quality = run.view("data_quality").data["fields"]
    assert quality["text"]["status"] == "partial" and quality["text"]["observed"] == 3
    assert quality["tool_inputs"]["status"] == "observed" and quality["tool_inputs"]["total"] == 5
    assert quality["tool_outputs"]["status"] == "partial" and quality["tool_outputs"]["observed"] == 4
    assert quality["tool_outcomes"]["status"] == "observed"
    calls = [call for step in run.steps for call in step["tool_calls"]]
    reads = [call for call in calls if call["tool_name"] == "Read"]
    assert len(reads) == 4
    assert all(call["input"] == {"file_path": "example.py"} for call in reads)
    assert all(call["output"] == "def parse_example(value): return value\n" for call in reads)
    assert not run.warnings


@pytest.mark.parametrize("fixture", [REDACTED, CONTENT], ids=["redacted", "reviewer-content"])
@pytest.mark.parametrize("include_raw", [False, True], ids=["lightweight", "full"])
def test_recorded_selected_views_roundtrip_and_relocated_archive_identity(tmp_path, fixture, include_raw):
    original = tv.load(fixture)
    sid = tv.list_sessions(fixture)[0]["id"]
    source = tmp_path / "relocated.zip"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("manifest.json", "traces.jsonl", "logs.jsonl"):
            archive.write(fixture / name, f"capture/{name}")
    selected = tv.load(source, session_id=sid)
    assert selected.sha256 == original.sha256
    assert tv.list_sessions(source) == tv.list_sessions(fixture)
    export = tmp_path / "trajectory.json"
    export.write_text(json.dumps(selected.to_dict(include_raw=include_raw), sort_keys=True))
    restored = tv.load(export, format="otlp", session_id=sid)
    assert restored.sha256 == selected.sha256
    assert restored.metrics == selected.metrics
    assert {name: view.to_dict() for name, view in restored.views().items()} == {
        name: view.to_dict() for name, view in selected.views().items()
    }


def test_recorded_rotated_replays_do_not_duplicate_reviewer_usage(tmp_path):
    capture = tmp_path / "capture"
    shutil.copytree(CONTENT, capture)
    for name in ("traces.jsonl", "logs.jsonl"):
        source = capture / name
        original = source.read_bytes()
        source.write_bytes(original + original)
    run = tv.load(capture)
    assert run.metrics["tokens"]["total"] == 144
    assert run.metrics["tool_call_count"] == 5
    assert len(run.agents) == 2
    assert len(run.raw["_otel"]["spans"]) == 20
    assert len(run.raw["_otel"]["logs"]) == 17
    assert all(len(span["sources"]) == 2 for span in run.raw["_otel"]["spans"])
    assert not run.warnings


@pytest.mark.parametrize("fixture", [REDACTED, CONTENT], ids=["redacted", "reviewer-content"])
def test_recorded_fixtures_keep_only_sanitized_resource_and_controlled_content(fixture):
    for name in ("traces.jsonl", "logs.jsonl"):
        data = (fixture / name).read_text()
        for forbidden in ("user.email", "user.id", "user.account", "organization.id", "/Users/", "/private/", "@"):
            assert forbidden not in data
        for line in data.splitlines():
            request = json.loads(line)
            for resource in request.get("resourceSpans", request.get("resourceLogs", [])):
                assert {entry["key"] for entry in resource["resource"]["attributes"]} == {
                    "service.name", "service.version",
                }
