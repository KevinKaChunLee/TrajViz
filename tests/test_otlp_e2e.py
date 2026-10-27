"""Run saved-capture workflows through the real command-line process boundary."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import jsonschema
import pytest

import agent_trajviz as tv

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "otlp"


def _cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "agent_trajviz.insight", *map(str, args)],
        cwd=cwd or ROOT, env=env, capture_output=True, text=True, timeout=90,
    )


def _second_session(value, key=""):
    if isinstance(value, dict):
        return {name: _second_session(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_second_session(item, key) for item in value]
    if isinstance(value, str):
        if key in {"traceId", "spanId", "parentSpanId"} and value:
            return f"{int(value, 16) ^ (1 << (4 * len(value) - 1)):0{len(value)}x}"
        if key.endswith("TimeUnixNano") or key == "timeUnixNano":
            return str(int(value) + 10_000_000_000)
        return value.replace("demo-session", "second-session")
    return value


def _bundle(tmp_path: Path, *, partial=False) -> tuple[Path, Path]:
    capture = tmp_path / "capture with spaces"
    capture.mkdir()
    (capture / "manifest.json").write_bytes((FIXTURE / "manifest.json").read_bytes())
    for name in ("claude_code_traces.jsonl", "claude_code_logs.jsonl"):
        batches = [json.loads(line) for line in (FIXTURE / name).read_text().splitlines()]
        if partial:
            for batch in batches:
                for resource in batch.get("resourceSpans", batch.get("resourceLogs", [])):
                    for scope in resource.get("scopeSpans", resource.get("scopeLogs", [])):
                        for record in scope.get("spans", scope.get("logRecords", [])):
                            record["attributes"] = [attribute for attribute in record.get("attributes", [])
                                                    if attribute["key"] not in {
                                                        "input_tokens", "output_tokens", "cache_read_tokens",
                                                        "cache_creation_tokens", "success", "error", "error_class",
                                                    }]
                            if "spans" in scope:
                                record["status"] = {"code": 0}
                                record["events"] = []
        batches.extend(_second_session(copy.deepcopy(batches)))
        (capture / name).write_text("\n".join(json.dumps(batch) for batch in batches) + "\n")
    archive = tmp_path / "capture with spaces.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in capture.iterdir():
            handle.write(path, f"nested capture/{path.name}")
    return capture, archive


@pytest.mark.parametrize("source_kind", ["directory", "manifest", "zip"])
@pytest.mark.parametrize("partial", [False, True], ids=["observed", "partial"])
def test_real_cli_discovers_selects_exports_every_view_and_reloads(tmp_path, source_kind, partial):
    directory, archive = _bundle(tmp_path, partial=partial)
    source = {"directory": directory, "manifest": directory / "manifest.json", "zip": archive}[source_kind]
    # Run away from the checkout and include spaces in all paths, as a user does.
    listed = _cli("--list-sessions", source, cwd=tmp_path)
    assert listed.returncode == 0, listed.stderr
    assert listed.stderr == ""
    sessions = json.loads(listed.stdout)
    assert [row["id"] for row in sessions] == ["demo-session", "second-session"]
    assert sessions == tv.list_sessions(source)

    ambiguous = _cli("--json", source, "-o", tmp_path / "ambiguous")
    assert ambiguous.returncode == 1 and ambiguous.stdout == ""
    assert "multiple sessions" in ambiguous.stderr and "Traceback" not in ambiguous.stderr
    assert not (tmp_path / "ambiguous").exists()

    output = tmp_path / "exported views"
    selected = _cli("--json", source, "--format", "otlp", "--session-id", "second-session", "-o", output)
    assert selected.returncode == 0, selected.stderr
    assert selected.stdout.strip() == str(output) and selected.stderr == ""
    run = tv.load(source, session_id="second-session")
    restored = tv.load(output / "trajectory.json", format="otlp", session_id="second-session")
    assert restored.sha256 == run.sha256
    assert restored.to_dict(include_raw=True) == run.to_dict(include_raw=True)
    index = json.loads((output / "index.json").read_text())
    assert [row["id"] for row in index["views"]] == list(tv.VIEW_CATALOG)
    for view_id in tv.VIEW_CATALOG:
        document = json.loads((output / f"{view_id}.json").read_text())
        jsonschema.validate(document, tv.view_schema(view_id))
        assert document == restored.view(view_id).to_dict() == run.view(view_id).to_dict()
        json.dumps(document, allow_nan=False)
    assert run.metrics["tokens"]["total"] == (None if partial else 57)
    assert run.raw["_otel"]["session_id"] == "second-session"
    assert all(step["session_id"] == "second-session" for step in run.steps)
    assert all(span["attributes"].get("session.id") != "demo-session" for span in run.raw["_otel"]["spans"])
    assert all(isinstance(span["start_time_unix_nano"], int) for span in restored.raw["_otel"]["spans"])


@pytest.mark.parametrize("include_raw", [False, True], ids=["lightweight", "full"])
def test_multisession_selected_roundtrip_preserves_all_views(tmp_path, include_raw):
    source, _ = _bundle(tmp_path)
    run = tv.load(source, session_id="second-session")
    normalized = tmp_path / "normalized.json"
    normalized.write_text(json.dumps(run.to_dict(include_raw=include_raw), sort_keys=True))
    restored = tv.load(normalized, session_id="second-session")
    assert restored.sha256 == run.sha256
    assert {key: view.to_dict() for key, view in restored.views().items()} == {
        key: view.to_dict() for key, view in run.views().items()
    }


@pytest.mark.parametrize("source_kind", ["directory", "zip"])
def test_real_report_cli_offline_partial_evidence_and_session_identity(tmp_path, source_kind):
    directory, archive = _bundle(tmp_path, partial=True)
    source = directory if source_kind == "directory" else archive
    destination = tmp_path / "offline report.html"
    result = _cli("--report", source, "--session-id", "second-session", "--inline-plotly", "--dark", "-o", destination)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(destination)
    html = destination.read_text()
    assert "OpenTelemetry capture" in html
    assert "capture completeness is unknown" in html
    assert "Failure-pattern checks unavailable" in html
    assert "No errors detected" not in html
    assert not re.search(r'<script\b[^>]*\bsrc=["\']https?://', html, re.IGNORECASE)
    assert tv.load(source, session_id="second-session").sha256 in html[:4096]


@pytest.mark.parametrize("mode", ["--json", "--report", "--list-sessions"])
@pytest.mark.parametrize("option", ["--session-id", "--format"])
def test_empty_cli_selectors_are_usage_errors_without_tracebacks(tmp_path, mode, option):
    result = _cli(mode, FIXTURE, option, "", "-o", tmp_path / "output")
    assert result.returncode == 2
    assert result.stdout == ""
    assert option in result.stderr or "format" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mode", ["--json", "--report"])
@pytest.mark.parametrize("failure", ["missing-session", "native-mismatch", "missing-source"])
def test_real_cli_input_failures_have_machine_readable_stdout(tmp_path, mode, failure):
    source = FIXTURE if failure != "missing-source" else tmp_path / "missing.zip"
    options = {"missing-session": ["--session-id", "absent"],
               "native-mismatch": ["--format", "claude_code"], "missing-source": []}[failure]
    output = tmp_path / "output"
    result = _cli(mode, source, *options, "-o", output)
    assert result.returncode == 1
    assert result.stdout == "" and result.stderr and "Traceback" not in result.stderr
    assert not output.exists()


def test_native_jsonl_cli_keeps_subagent_loading_and_format_gate(tmp_path):
    native = FIXTURE / "native" / "demo-session.jsonl"
    native_run = tv.load(native)
    result = _cli("--json", native, "--format", "claude_code", "-o", tmp_path / "native")
    assert result.returncode == 0, result.stderr
    restored = tv.load(tmp_path / "native" / "trajectory.json")
    assert restored.metrics == native_run.metrics
    assert len(restored.agents) == len(native_run.agents) == 2
    refused = _cli("--json", native, "--format", "otlp", "-o", tmp_path / "refused")
    assert refused.returncode == 1 and refused.stdout == ""
    assert "not otlp" in refused.stderr
    assert not (tmp_path / "refused").exists()
