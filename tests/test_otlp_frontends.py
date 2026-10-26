"""Saved captures use the same library, command, dashboard and report pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.__main__ import main

FIXTURE = Path(__file__).parent / "fixtures" / "otlp"


def incomplete_capture(tmp_path):
    batches = [json.loads(line) for line in (FIXTURE / "claude_code_traces.jsonl").read_text().splitlines()]
    for batch in batches:
        for resource in batch["resourceSpans"]:
            for scope in resource["scopeSpans"]:
                for span in scope["spans"]:
                    span["attributes"] = [attribute for attribute in span.get("attributes", [])
                                          if attribute["key"] not in {
                                              "input_tokens", "output_tokens", "cache_read_tokens",
                                              "cache_creation_tokens", "success", "error", "error_class",
                                          }]
                    span["status"] = {"code": 0}
                    span["events"] = []
    path = tmp_path / "partial.jsonl"
    path.write_text("\n".join(json.dumps(batch) for batch in batches) + "\n")
    return path


def test_cli_lists_sessions_and_exports_selected_run(tmp_path, capsys):
    assert main(["--list-sessions", str(FIXTURE)]) == 0
    sessions = json.loads(capsys.readouterr().out)
    assert len(sessions) == 1
    assert main(["--json", str(FIXTURE), "--format", "otlp", "--session-id", sessions[0]["id"],
                 "--view", "data_quality", "-o", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "data_quality.json").read_text())["schema_version"] == 2
    assert tv.load(tmp_path / "trajectory.json").metrics["tokens"]["total"] == 57


def test_cli_invalid_session_and_format_are_actionable(capsys):
    assert main(["--json", str(FIXTURE), "--session-id", "absent"]) == 1
    assert "session" in capsys.readouterr().err.lower()
    with pytest.raises(SystemExit) as error:
        main(["--json", str(FIXTURE), "--format", "not-a-format"])
    assert error.value.code == 2


@pytest.mark.parametrize("missing", [False, True])
def test_dashboard_pack_and_report_support_partial_evidence(tmp_path, missing):
    from agent_trajviz.insight.report import build_report_html
    from agent_trajviz.insight.session import session_from_run
    from agent_trajviz.insight.ui.load import pack_load_outputs

    run = tv.load(incomplete_capture(tmp_path) if missing else FIXTURE)
    session = session_from_run(run)
    packed = pack_load_outputs(session)
    assert packed["state_run"] is run
    report = build_report_html(session, inline_plotly=True)
    assert "OpenTelemetry capture" in report
    assert "capture completeness is unknown" in report
    if missing:
        assert run.metrics["tokens"]["total"] is None
        assert "Failure-pattern checks unavailable" in report
        assert "No errors detected" not in report


def test_dashboard_inspection_does_not_analyse_the_capture():
    from agent_trajviz.insight.ui.load import _inspect_sessions

    update = _inspect_sessions(str(FIXTURE))
    assert update["visible"] is True
    assert len(update["choices"]) == 1
    assert update["value"] == tv.list_sessions(FIXTURE)[0]["id"]
    assert _inspect_sessions(None)["visible"] is False


def test_otlp_does_not_make_transcript_comparison_or_attribution_claims():
    from agent_trajviz.attribution import diagnose

    run = tv.load(FIXTURE)
    with pytest.raises(ValueError, match="operation-based"):
        tv.compare(run, run)
    with pytest.raises(ValueError, match="None of the trajectories"):
        tv.compare_runs([run, run])
    result = diagnose(run)
    assert result["available"] is False
    assert "OTLP" in result["reason"]
    assert run.context_pressure()["availability"]["status"] == "unavailable"
