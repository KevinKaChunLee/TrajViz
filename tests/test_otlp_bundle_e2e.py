"""Exercise selected saved captures through the installed command/public API.

These tests deliberately use filesystem and ZIP bytes rather than mocking a
reader, and check independently known usage and nanosecond values.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.formats import otlp_files

FIXTURE = Path(__file__).parent / "fixtures" / "otlp"
ROOT = Path(__file__).resolve().parents[1]


def _cli(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "agent_trajviz.insight", *map(str, arguments)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30,
    )


def _bundle(directory):
    directory.mkdir()
    for name in ("manifest.json", "claude_code_traces.jsonl", "claude_code_logs.jsonl"):
        shutil.copyfile(FIXTURE / name, directory / name)
    return directory


def _archive(path, directory, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for source in sorted(directory.iterdir(), reverse=True):
            archive.write(source, f"relocated/{source.name}")
    return path


@pytest.mark.parametrize("representation", [
    "raw-json", "raw-jsonl", "mixed-jsonl", "directory", "manifest",
    "zip-stored", "zip-deflated", "zip-bzip2", "zip-lzma",
])
def test_public_import_and_real_cli_export_matrix(tmp_path, representation):
    directory = _bundle(tmp_path / "capture")
    traces = (directory / "claude_code_traces.jsonl").read_bytes()
    if representation == "raw-json":
        requests = [json.loads(line) for line in traces.splitlines()]
        payload = {"resourceSpans": [resource for request in requests for resource in request["resourceSpans"]]}
        source = tmp_path / "capture.json"
        source.write_text(json.dumps(payload, indent=2))
    elif representation == "raw-jsonl":
        source = directory / "claude_code_traces.jsonl"
    elif representation == "mixed-jsonl":
        source = tmp_path / "capture.jsonl"
        # A duplicated rotation must retain provenance without inflating usage.
        source.write_bytes(b"\xef\xbb\xbf" + traces + b"\r\n" + traces
                           + (directory / "claude_code_logs.jsonl").read_bytes())
    elif representation == "directory":
        source = directory
    elif representation == "manifest":
        source = directory / "manifest.json"
    else:
        compression = {
            "zip-stored": zipfile.ZIP_STORED, "zip-deflated": zipfile.ZIP_DEFLATED,
            "zip-bzip2": zipfile.ZIP_BZIP2, "zip-lzma": zipfile.ZIP_LZMA,
        }[representation]
        if compression == zipfile.ZIP_BZIP2:
            pytest.importorskip("bz2")
        if compression == zipfile.ZIP_LZMA:
            pytest.importorskip("lzma")
        source = _archive(tmp_path / "capture.zip", directory, compression)

    sessions = tv.list_sessions(source)
    assert [session["id"] for session in sessions] == ["demo-session"]
    assert sessions[0]["span_count"] == 9
    assert sessions[0]["interaction_count"] == 2
    detected = tv.load(source)
    selected = tv.load(source, format="otlp", session_id="demo-session")
    assert detected.sha256 == selected.sha256
    assert detected.metrics["tokens"]["total"] == 57
    assert detected.metrics["tool_call_count"] == 3
    assert len(detected.steps) == (9 if representation in {"raw-json", "raw-jsonl"} else 6)
    assert detected.raw["_otel"]["spans"][0]["start_time_unix_nano"] == 1_760_000_000_000_000_001
    if representation == "mixed-jsonl":
        assert len(detected.raw["_otel"]["spans"][0]["sources"]) == 2
    if representation in {"directory", "manifest"} or representation.startswith("zip-"):
        assert detected.sha256 == tv.load(FIXTURE).sha256
        assert detected.raw["_otel"]["capture_sha256"] == tv.load(FIXTURE).raw["_otel"]["capture_sha256"]

    discovery = _cli("--list-sessions", source)
    assert discovery.returncode == 0, discovery.stderr
    assert json.loads(discovery.stdout) == sessions
    destination = tmp_path / "exported"
    command = _cli("--json", source, "--format", "otlp", "--session-id", "demo-session",
                   "--view", "metrics", "--view", "data_quality", "-o", destination)
    assert command.returncode == 0, command.stderr
    restored = tv.load(destination / "trajectory.json")
    assert restored.sha256 == detected.sha256
    assert restored.metrics == detected.metrics
    assert restored.raw["_otel"]["spans"] == detected.raw["_otel"]["spans"]
    assert json.loads((destination / "metrics.json").read_text())["schema_version"] == 2
    assert json.loads((destination / "data_quality.json").read_text())["schema_version"] == 2


def _corrupt_member(path, name):
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(name)
    data = bytearray(path.read_bytes())
    name_size, extra_size = struct.unpack_from("<HH", data, member.header_offset + 26)
    start = member.header_offset + 30 + name_size + extra_size
    data[start:start + member.compress_size] = b"\xff" * member.compress_size
    path.write_bytes(data)


@pytest.mark.parametrize("compression", [zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
@pytest.mark.parametrize("member", ["manifest.json", "claude_code_traces.jsonl"])
def test_corrupt_zip_members_remain_public_errors_and_cli_does_not_traceback(tmp_path, compression, member):
    if compression == zipfile.ZIP_BZIP2:
        pytest.importorskip("bz2")
    if compression == zipfile.ZIP_LZMA:
        pytest.importorskip("lzma")
    source = _archive(tmp_path / "corrupt.zip", _bundle(tmp_path / "capture"), compression)
    _corrupt_member(source, f"relocated/{member}")
    # Sniffing is a Boolean operation even when a compressed manifest is broken.
    assert otlp_files.looks_like_otlp_source(source) is (member != "manifest.json")
    operations = [lambda: tv.load(source, format="otlp"), lambda: tv.list_sessions(source)]
    if member != "manifest.json":
        operations.append(lambda: tv.load(source))
    for operation in operations:
        with pytest.raises(tv.LoadError) as caught:
            operation()
        assert caught.value.code == "load_failed"
        assert caught.value.path == str(source)
        assert member in str(caught.value)
    command = _cli("--json", source, "--format", "otlp", "-o", tmp_path / "output")
    assert command.returncode == 1
    assert member in command.stderr
    assert "Traceback" not in command.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("case,code", [
    ("truncated-tail", "invalid_otlp"), ("duplicate-json-key", "invalid_otlp"),
    ("binary-protobuf", "invalid_otlp"), ("metrics-only", "unsupported_otlp"),
    ("missing-member", "unsafe_bundle"), ("traversal", "unsafe_bundle"),
    ("duplicate-manifests", "invalid_otlp_bundle"),
])
def test_invalid_capture_public_api_and_cli_fail_without_partial_export(tmp_path, case, code):
    source = tmp_path / "bad.jsonl"
    if case == "truncated-tail":
        source.write_bytes((FIXTURE / "claude_code_traces.jsonl").read_bytes() + b'{"resourceSpans":[')
    elif case == "duplicate-json-key":
        source.write_bytes(b'{"resourceSpans":[],"resourceSpans":[]}')
    elif case == "binary-protobuf":
        source.write_bytes(b"\x0a\xff\x80\x00")
    elif case == "metrics-only":
        source.write_text('{"resourceMetrics":[]}')
    else:
        source = _bundle(tmp_path / "capture")
        manifest = source / "manifest.json"
        if case == "missing-member":
            (source / "claude_code_traces.jsonl").unlink()
        elif case == "traversal":
            payload = json.loads(manifest.read_text())
            payload["traces"] = ["../outside.jsonl"]
            manifest.write_text(json.dumps(payload))
        else:
            shutil.copyfile(manifest, source / "otlp-manifest.json")
    operations = [lambda: tv.load(source, format="otlp"), lambda: tv.list_sessions(source)]
    if case == "duplicate-manifests":
        operations.append(lambda: tv.load(source))
    for operation in operations:
        with pytest.raises(tv.LoadError) as caught:
            operation()
        assert caught.value.code == code
    command = _cli("--json", source, "--format", "otlp", "-o", tmp_path / "output")
    assert command.returncode == 1
    assert command.stderr.strip()
    assert "Traceback" not in command.stderr
    assert not (tmp_path / "output").exists()


def test_relocated_bundle_identity_ignores_unlisted_private_files(tmp_path):
    original = _bundle(tmp_path / "capture")
    first = tv.load(original)
    (original / "unlisted.txt").write_text("not telemetry; must not be collected")
    relocated = tmp_path / "new-parent" / "renamed"
    relocated.parent.mkdir()
    shutil.copytree(original, relocated)
    archived = _archive(tmp_path / "capture.zip", relocated, zipfile.ZIP_DEFLATED)
    for source in [original, relocated, archived]:
        run = tv.load(source)
        assert run.sha256 == first.sha256
        assert run.raw["_otel"]["capture_sha256"] == first.raw["_otel"]["capture_sha256"]
        assert tv.list_sessions(source) == tv.list_sessions(original)
        assert not any("unlisted.txt" in path for path in run.raw["_otel"]["source"])
    # Identity covers all declared bytes, even JSON whitespace with equal semantics.
    traces = relocated / "claude_code_traces.jsonl"
    traces.write_bytes(traces.read_bytes() + b"\n")
    changed = tv.load(relocated)
    assert changed.metrics == first.metrics
    assert changed.sha256 != first.sha256


@pytest.mark.parametrize("boundary", ["raw-member", "bundle-total", "zip-file", "zip-expanded-member", "zip-members"])
def test_public_size_boundaries_are_enforced_before_unbounded_reads(tmp_path, monkeypatch, boundary):
    directory = _bundle(tmp_path / "capture")
    if boundary == "raw-member":
        source = directory / "claude_code_traces.jsonl"
        monkeypatch.setattr(otlp_files, "MAX_MEMBER_BYTES", source.stat().st_size - 1)
    elif boundary == "bundle-total":
        source = directory
        total = sum(path.stat().st_size for path in directory.iterdir())
        monkeypatch.setattr(otlp_files, "MAX_TOTAL_BYTES", total - 1)
    else:
        source = _archive(tmp_path / "bounded.zip", directory, zipfile.ZIP_DEFLATED)
        if boundary == "zip-file":
            monkeypatch.setattr(otlp_files, "MAX_TOTAL_BYTES", source.stat().st_size - 1)
        elif boundary == "zip-members":
            monkeypatch.setattr(otlp_files, "MAX_MEMBERS", 1)
        else:
            # The compressed archive fits this budget; the trace member's
            # declared expanded size exceeds it. Refuse before decompressing.
            assert source.stat().st_size < 4096
            monkeypatch.setattr(otlp_files, "MAX_MEMBER_BYTES", 4096)
            original = otlp_files._zip_member_bytes

            def refuse_trace_decompression(archive, info, limit, label):
                if info.filename.endswith("claude_code_traces.jsonl"):
                    pytest.fail("oversized member must be refused before decompression")
                return original(archive, info, limit, label)

            monkeypatch.setattr(otlp_files, "_zip_member_bytes", refuse_trace_decompression)
    for operation in [lambda: tv.load(source, format="otlp"), lambda: tv.list_sessions(source)]:
        with pytest.raises(tv.LoadError) as caught:
            operation()
        assert caught.value.code == "size_limit"


@pytest.mark.parametrize("kind", ["selected-symlink", "member-symlink", "ancestor-symlink", "selected-fifo", "member-fifo"])
def test_public_api_and_real_cli_reject_filesystem_escape_and_blocking_sources(tmp_path, kind):
    directory = _bundle(tmp_path / "capture")
    trace = directory / "claude_code_traces.jsonl"
    code = "unsafe_bundle" if kind.endswith("symlink") else "load_failed"
    if kind == "selected-symlink":
        source = tmp_path / "linked.jsonl"
        source.symlink_to(trace)
    elif kind in {"selected-fifo", "member-fifo"}:
        if not hasattr(os, "mkfifo"):
            pytest.skip("platform has no named pipes")
        if kind == "selected-fifo":
            source = tmp_path / "pipe.jsonl"
            os.mkfifo(source)
        else:
            trace.unlink()
            os.mkfifo(trace)
            source = directory
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        shutil.copyfile(trace, outside / "data.jsonl")
        trace.unlink()
        source = directory
        if kind == "member-symlink":
            trace.symlink_to(outside / "data.jsonl")
        else:
            (directory / "linked").symlink_to(outside, target_is_directory=True)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["traces"] = ["linked/data.jsonl"]
            manifest_path.write_text(json.dumps(manifest))
    for operation in [lambda: tv.load(source, format="otlp"), lambda: tv.list_sessions(source)]:
        with pytest.raises(tv.LoadError) as caught:
            operation()
        assert caught.value.code == code
    command = _cli("--json", source, "--format", "otlp", "-o", tmp_path / "output")
    assert command.returncode == 1
    assert "Traceback" not in command.stderr
    assert not (tmp_path / "output").exists()
