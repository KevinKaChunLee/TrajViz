"""Transport precision, deduplication, provenance and explicit bundle safety."""

from __future__ import annotations

import copy
import hashlib
import json
import stat
import zipfile

import pytest

from agent_trajviz.errors import LoadError
from agent_trajviz.insight.formats import otlp, otlp_files
from agent_trajviz.insight.formats.otlp import decode_otlp_bytes, looks_like_otlp_payload
from agent_trajviz.insight.formats.otlp_files import looks_like_otlp_source, read_otlp_capture

TRACE = "a" * 32
SPAN = "b" * 16
NANO = 1_792_963_200_123_456_789


def attribute(key, value):
    return {"key": key, "value": value}


def span(**overrides):
    result = {
        "traceId": TRACE, "spanId": SPAN, "name": "claude_code.llm_request",
        "startTimeUnixNano": str(NANO), "endTimeUnixNano": str(NANO + 2_000_001),
        "attributes": [attribute("session.id", {"stringValue": "session-1"})],
    }
    result.update(overrides)
    return result


def traces(*spans):
    return {"resourceSpans": [{
        "resource": {"attributes": [attribute("service.name", {"stringValue": "claude-code"})]},
        "schemaUrl": "https://example.invalid/resource-schema",
        "scopeSpans": [{"scope": {"name": "claude-code", "version": "test"}, "spans": list(spans)}],
    }]}


def logs(*records):
    return {"resourceLogs": [{"scopeLogs": [{"logRecords": list(records)}]}]}


def encoded(payload):
    return json.dumps(payload).encode()


def write_manifest(directory, *, traces_list=None, logs_list=None, **overrides):
    manifest = {
        "type": "agent-trajviz-otlp-bundle", "schema_version": 1, "profile": "claude_code",
        "traces": ["traces.jsonl"] if traces_list is None else traces_list,
        "logs": [] if logs_list is None else logs_list,
    }
    manifest.update(overrides)
    path = directory / "manifest.json"
    path.write_bytes(encoded(manifest))
    return path


def test_integer_timestamps_ids_resource_scope_and_raw_are_preserved():
    original = traces(span(
        traceId=TRACE.upper(), spanId=SPAN.upper(), status={"code": "STATUS_CODE_ERROR", "message": "failed"},
        events=[{"name": "queued", "timeUnixNano": str(NANO + 1), "attributes": []}],
        links=[{"traceId": "c" * 32, "spanId": "d" * 16}],
    ))
    data = encoded(original)
    capture = decode_otlp_bytes(data, "selected.json")
    record = capture.spans[0]
    assert record["start_time_unix_nano"] == NANO
    assert record["end_time_unix_nano"] - record["start_time_unix_nano"] == 2_000_001
    assert record["events"][0]["time_unix_nano"] == NANO + 1
    assert record["links"][0]["trace_id"] == "c" * 32
    assert record["trace_id"] == TRACE
    assert record["span_id"] == SPAN
    assert record["parent_span_id"] == ""
    assert record["status"] == {"code": 2, "message": "failed"}
    assert record["resource"] == {"service.name": "claude-code"}
    assert record["scope"]["version"] == "test"
    assert record["schema_url"] == "https://example.invalid/resource-schema"
    assert record["source"] == {"path": "selected.json", "line": 1, "index": 0}
    assert record["raw"] == original["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert capture.raw["requests"] == [original]
    assert capture.sha256 == hashlib.sha256(data).hexdigest()


def test_all_anyvalue_shapes_remain_json_serializable():
    values = {
        "str": {"stringValue": "prompt"}, "bool": {"boolValue": False},
        "int": {"intValue": "-9223372036854775808"}, "double": {"doubleValue": 1.5},
        "nan": {"doubleValue": "NaN"}, "bytes": {"bytesValue": "AAE="},
        "array": {"arrayValue": {"values": [{"intValue": "4"}, {}]}},
        "map": {"kvlistValue": {"values": [attribute("unknown.vendor", {"stringValue": "retained"})]}},
    }
    record = span(attributes=[attribute(key, value) for key, value in values.items()])
    attrs = decode_otlp_bytes(encoded(traces(record))).spans[0]["attributes"]
    assert attrs == {
        "str": "prompt", "bool": False, "int": -(2**63), "double": 1.5, "nan": "NaN",
        "bytes": {"$otlp_bytes": "AAE="}, "array": [4, None], "map": {"unknown.vendor": "retained"},
    }
    json.dumps(attrs, allow_nan=False)


def test_mixed_jsonl_requests_lines_and_logs():
    event = {
        "timeUnixNano": str(NANO + 1), "observedTimeUnixNano": str(NANO + 2),
        "traceId": TRACE, "spanId": SPAN, "severityNumber": 9, "severityText": "INFO",
        "body": {"stringValue": "claude_code.api_request"},
        "attributes": [attribute("request_id", {"stringValue": "request-1"})],
    }
    data = b"\xef\xbb\xbf" + encoded(traces(span())) + b"\n\n" + encoded(logs(event)) + b"\n"
    capture = decode_otlp_bytes(data, "capture.jsonl")
    assert len(capture.spans) == len(capture.logs) == 1
    assert capture.logs[0]["source"] == {"path": "capture.jsonl", "line": 3, "index": 1}
    assert capture.logs[0]["body"] == "claude_code.api_request"
    assert capture.logs[0]["observed_time_unix_nano"] == NANO + 2
    assert capture.logs[0]["attributes"]["request_id"] == "request-1"


def test_unrelated_logs_can_omit_trace_ids_and_use_typed_body():
    capture = decode_otlp_bytes(encoded(logs({"body": {"kvlistValue": {
        "values": [attribute("event.name", {"stringValue": "api_request"})],
    }}})))
    assert capture.logs[0]["trace_id"] == capture.logs[0]["span_id"] == ""
    assert capture.logs[0]["body"] == {"event.name": "api_request"}


def test_original_proto_field_names_are_accepted():
    original = {"resource_spans": [{"scope_spans": [{"spans": [{
        "trace_id": TRACE, "span_id": SPAN, "name": "test", "start_time_unix_nano": str(NANO),
        "attributes": [attribute("n", {"int_value": "3"})],
    }]}]}]}
    capture = decode_otlp_bytes(encoded(original))
    assert capture.spans[0]["attributes"] == {"n": 3}
    assert capture.spans[0]["start_time_unix_nano"] == NANO


def test_duplicates_and_out_of_order_records_are_deterministic():
    parent = span(spanId="c" * 16, startTimeUnixNano=str(NANO - 100))
    child = span(parentSpanId="c" * 16)
    event = {"timeUnixNano": str(NANO), "body": {"stringValue": "result"}}
    requests = [traces(child), traces(parent), traces(copy.deepcopy(child)), logs(event), logs(event)]
    capture = decode_otlp_bytes(b"\n".join(encoded(request) for request in requests))
    assert [item["span_id"] for item in capture.spans] == ["c" * 16, SPAN]
    assert len(capture.logs) == 1
    assert len(capture.spans[1]["sources"]) == 2
    assert len(capture.logs[0]["sources"]) == 2
    assert not capture.warnings
    a = span(attributes=[attribute("tokens", {"intValue": "5"})])
    b = span(attributes=[attribute("tokens", {"intValue": "7"})])
    first = decode_otlp_bytes(encoded(traces(a, b)))
    second = decode_otlp_bytes(encoded(traces(b, a)))
    assert first.spans[0]["attributes"] == second.spans[0]["attributes"]
    assert first.warnings == second.warnings
    assert "Conflicting duplicate" in first.warnings[0]


def test_same_span_id_in_two_traces_is_not_a_duplicate():
    capture = decode_otlp_bytes(encoded(traces(span(), span(traceId="c" * 32))))
    assert len(capture.spans) == 2


@pytest.mark.parametrize("mutate", [
    lambda item: item.update(traceId="0" * 32),
    lambda item: item.update(spanId="not-hex"),
    lambda item: item.update(parentSpanId="0" * 16),
    lambda item: item.update(startTimeUnixNano=1.1),
    lambda item: item.update(startTimeUnixNano=True),
    lambda item: item.update(startTimeUnixNano=str(2**64)),
    lambda item: item.update(startTimeUnixNano="-1"),
    lambda item: item.update(status={"code": 99}),
    lambda item: item.update(attributes=[attribute("a", {"intValue": str(2**63)})]),
    lambda item: item.update(attributes=[attribute("a", {"boolValue": "true"})]),
    lambda item: item.update(attributes=[attribute("a", {"bytesValue": "??"})]),
    lambda item: item.update(attributes=[attribute("a", {"boolValue": True, "intValue": "1"})]),
    lambda item: item.update(attributes=[attribute("a", {}), attribute("a", {})]),
])
def test_invalid_wire_values_fail_clearly(mutate):
    record = span()
    mutate(record)
    with pytest.raises(LoadError) as error:
        decode_otlp_bytes(encoded(traces(record)), "bad.json")
    assert error.value.code == "invalid_otlp"
    assert error.value.path == "bad.json"


@pytest.mark.parametrize("data", [
    b"", b"\xff", b"{\"resourceSpans\": []}\n{bad", b'{"resourceSpans":[],"resourceSpans":[]}',
    b'{"resourceSpans":[{"scopeSpans":[{"spans":NaN}]}]}',
])
def test_invalid_json_and_binary_never_silently_load_partial_data(data):
    with pytest.raises(LoadError) as error:
        decode_otlp_bytes(data, "broken.jsonl")
    assert error.value.code == "invalid_otlp"


@pytest.mark.parametrize("payload", [{"resourceMetrics": []}, {"spans": []}, [], {"body": "debug exporter"}])
def test_non_transport_json_is_rejected(payload):
    assert not looks_like_otlp_payload(payload)
    with pytest.raises(LoadError) as error:
        decode_otlp_bytes(encoded(payload))
    assert error.value.code == "unsupported_otlp"


def test_record_attribute_and_depth_limits(monkeypatch):
    monkeypatch.setattr(otlp, "MAX_RECORDS", 1)
    with pytest.raises(LoadError, match="record limit"):
        decode_otlp_bytes(encoded(traces(span(), span(spanId="c" * 16))))
    monkeypatch.setattr(otlp, "MAX_ATTRIBUTES", 1)
    with pytest.raises(LoadError, match="item limit"):
        decode_otlp_bytes(encoded(traces(span(attributes=[attribute("a", {}), attribute("b", {})]))))
    monkeypatch.setattr(otlp, "MAX_DEPTH", 1)
    with pytest.raises(LoadError, match="nesting limit"):
        decode_otlp_bytes(encoded(traces(span(attributes=[attribute("a", {"arrayValue": {
            "values": [{"arrayValue": {"values": [{"stringValue": "deep"}]}}],
        }})]))))


def test_standalone_sniff_and_exact_bytes_hash(tmp_path):
    path = tmp_path / "data.jsonl"
    data = encoded(traces(span())) + b"\n"
    path.write_bytes(data)
    assert looks_like_otlp_source(path)
    capture = read_otlp_capture(path)
    assert capture.sha256 == hashlib.sha256(data).hexdigest()
    assert len(capture.spans) == 1
    path.write_bytes(b'{"native":true}\n')
    assert not looks_like_otlp_source(path)


def test_directory_zip_and_manifest_load_same_declared_data_with_same_identity(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    manifest = write_manifest(directory, logs_list=["logs.jsonl"])
    (directory / "traces.jsonl").write_bytes(encoded(traces(span())))
    (directory / "logs.jsonl").write_bytes(encoded(logs({"body": {"stringValue": "prompt"}})))
    # Unlisted files are neither parsed nor included in content identity.
    (directory / "private.jsonl").write_text("unrelated data")
    a = read_otlp_capture(directory)
    assert read_otlp_capture(manifest).sha256 == a.sha256
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in directory.iterdir():
            handle.write(path, f"relocated/{path.name}")
    b = read_otlp_capture(archive)
    assert len(a.spans) == len(b.spans) == len(a.logs) == len(b.logs) == 1
    assert a.sha256 == b.sha256
    assert a.raw["profile"] == "claude_code"
    assert "private.jsonl" not in repr(a.raw["source_files"])
    assert looks_like_otlp_source(directory) and looks_like_otlp_source(archive)
    (directory / "logs.jsonl").write_bytes(encoded(logs({"body": {"stringValue": "different"}})))
    assert read_otlp_capture(directory).sha256 != a.sha256


@pytest.mark.parametrize("member", ["../escape.jsonl", "/absolute.jsonl", "C:/file", "back\\slash", "./x", "a//b"])
def test_manifest_member_paths_cannot_escape_bundle(tmp_path, member):
    write_manifest(tmp_path, traces_list=[member])
    with pytest.raises(LoadError) as error:
        read_otlp_capture(tmp_path)
    assert error.value.code == "unsafe_bundle"


@pytest.mark.parametrize("overrides,code", [
    ({"schema_version": 2}, "unsupported_schema"),
    ({"schema_version": True}, "unsupported_schema"),
    ({"profile": "unknown"}, "unsupported_otlp"),
    ({"traces": []}, "invalid_otlp_bundle"),
    ({"traces": ["x", "x"]}, "invalid_otlp_bundle"),
    ({"traces": ["x"], "logs": ["x"]}, "invalid_otlp_bundle"),
])
def test_manifest_contract_is_explicit(tmp_path, overrides, code):
    write_manifest(tmp_path, **overrides)
    with pytest.raises(LoadError) as error:
        read_otlp_capture(tmp_path)
    assert error.value.code == code


def test_directory_does_not_discover_unlisted_telemetry(tmp_path):
    (tmp_path / "traces.jsonl").write_bytes(encoded(traces(span())))
    assert not looks_like_otlp_source(tmp_path)
    with pytest.raises(LoadError, match="requires exactly one"):
        read_otlp_capture(tmp_path)


def test_symlinked_files_and_ancestor_directories_are_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.jsonl").write_bytes(encoded(traces(span())))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "traces.jsonl").symlink_to(outside / "data.jsonl")
    write_manifest(bundle)
    with pytest.raises(LoadError) as error:
        read_otlp_capture(bundle)
    assert error.value.code == "unsafe_bundle"
    (bundle / "linked").symlink_to(outside, target_is_directory=True)
    write_manifest(bundle, traces_list=["linked/data.jsonl"])
    with pytest.raises(LoadError) as error:
        read_otlp_capture(bundle)
    assert error.value.code == "unsafe_bundle"


def test_zip_missing_duplicate_symlink_and_unsafe_members(tmp_path):
    path = tmp_path / "capture.zip"
    manifest = encoded({"type": "agent-trajviz-otlp-bundle", "schema_version": 1,
                        "profile": "claude_code", "traces": ["traces.jsonl"]})
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("manifest.json", manifest)
    with pytest.raises(LoadError, match="Missing"):
        read_otlp_capture(path)
    for name, is_symlink in [("../escape", False), ("traces.jsonl", True)]:
        with zipfile.ZipFile(path, "w") as handle:
            handle.writestr("manifest.json", manifest)
            info = zipfile.ZipInfo(name)
            if is_symlink:
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            handle.writestr(info, "target")
        with pytest.raises(LoadError) as error:
            read_otlp_capture(path)
        assert error.value.code == "unsafe_bundle"
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("manifest.json", manifest)
        handle.writestr("traces.jsonl", encoded(traces(span())))
        with pytest.warns(UserWarning):
            handle.writestr("traces.jsonl", encoded(traces(span())))
    with pytest.raises(LoadError, match="Duplicate"):
        read_otlp_capture(path)


def test_size_and_member_depth_limits(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    path.write_bytes(encoded(traces(span())))
    monkeypatch.setattr(otlp_files, "MAX_MEMBER_BYTES", 10)
    with pytest.raises(LoadError) as error:
        read_otlp_capture(path)
    assert error.value.code == "size_limit"
    monkeypatch.setattr(otlp_files, "MAX_MEMBER_BYTES", 1024 * 1024)
    write_manifest(tmp_path, traces_list=["a/b/c/data"])
    monkeypatch.setattr(otlp_files, "MAX_PATH_DEPTH", 2)
    with pytest.raises(LoadError) as error:
        read_otlp_capture(tmp_path)
    assert error.value.code == "size_limit"


def test_companion_batches_deduplicate_across_files(tmp_path):
    data = encoded(traces(span()))
    (tmp_path / "one.jsonl").write_bytes(data)
    (tmp_path / "two.jsonl").write_bytes(data)
    write_manifest(tmp_path, traces_list=["one.jsonl", "two.jsonl"])
    capture = read_otlp_capture(tmp_path)
    assert len(capture.spans) == 1
    assert len(capture.spans[0]["sources"]) == 2


def test_missing_file_uses_public_load_error(tmp_path):
    with pytest.raises(LoadError) as error:
        read_otlp_capture(tmp_path / "absent.json")
    assert error.value.code == "load_failed"


def test_large_pretty_envelope_sniff_does_not_claim_nested_native_keys(tmp_path):
    source = tmp_path / "large.json"
    source.write_text(json.dumps({"metadata": {"resourceSpans": []}, "padding": "x" * (2 * 1024 * 1024)}, indent=2))
    assert not looks_like_otlp_source(source)
    payload = traces(span(attributes=[attribute("padding", {"stringValue": "x" * (2 * 1024 * 1024)})]))
    source.write_text(json.dumps(payload, indent=2))
    assert looks_like_otlp_source(source)


def test_invalid_unicode_and_overflowing_float_raise_public_errors():
    payload = traces(span(name="\ud800"))
    with pytest.raises(LoadError) as error:
        decode_otlp_bytes(encoded(payload))
    assert error.value.code == "invalid_otlp"
    with pytest.raises(LoadError) as error:
        decode_otlp_bytes(b'{"resourceSpans": [], "unknown": 1e999}')
    assert error.value.code == "invalid_otlp"


def test_log_event_name_and_dropped_counts_survive_decoding():
    record = {"eventName": "claude_code.api_request", "flags": 1, "droppedAttributesCount": 2}
    result = decode_otlp_bytes(encoded(logs(record))).logs[0]
    assert result["event_name"] == "claude_code.api_request"
    assert result["flags"] == 1 and result["dropped_attributes_count"] == 2


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_line_separators_are_content_not_jsonl_record_delimiters(tmp_path, separator):
    content = f"before{separator}after"
    first = traces(span(attributes=[attribute("prompt", {"stringValue": content})]))
    second = logs({"body": {"stringValue": content}})
    data = (json.dumps(first, ensure_ascii=False) + "\n" + json.dumps(second, ensure_ascii=False) + "\n").encode()
    capture = decode_otlp_bytes(data)
    assert capture.spans[0]["attributes"]["prompt"] == capture.logs[0]["body"] == content
    source = tmp_path / "unicode.jsonl"
    source.write_bytes(data)
    assert looks_like_otlp_source(source)


def test_sniff_prefix_can_end_inside_utf8_code_point(tmp_path):
    limit = min(otlp_files.MAX_MEMBER_BYTES, 1024 * 1024) + 1
    # Keep the individual AnyValue under its byte limit, while the surrounding
    # request exceeds the sniff budget. Adjust alignment to split a two-byte é.
    for prefix in ("", "x"):
        payload = traces(span(attributes=[attribute("padding", {"stringValue": prefix + "é" * 524_200})]))
        data = json.dumps(payload, ensure_ascii=False).encode()
        try:
            data[:limit].decode("utf-8")
        except UnicodeDecodeError:
            break
    else:
        pytest.fail("fixture must split a code point at the sniff boundary")
    source = tmp_path / "large-utf8.json"
    source.write_bytes(data)
    assert looks_like_otlp_source(source)
    assert len(read_otlp_capture(source).spans) == 1


def test_bundle_symlink_swap_between_validation_and_open_is_rejected(tmp_path, monkeypatch):
    import os

    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("platform lacks descriptor-relative no-follow directory opens")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.jsonl").write_bytes(encoded(traces(span(name="outside"))))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    child = bundle / "child"
    child.mkdir()
    (child / "data.jsonl").write_bytes(encoded(traces(span())))
    write_manifest(bundle, traces_list=["child/data.jsonl"])
    original = os.open

    def swap(path, flags, *args, **kwargs):
        if path == "child":
            (child / "data.jsonl").unlink()
            child.rmdir()
            child.symlink_to(outside, target_is_directory=True)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    monkeypatch.setattr(os, "supports_dir_fd", {*os.supports_dir_fd, swap})
    with pytest.raises(LoadError) as error:
        read_otlp_capture(bundle)
    assert error.value.code == "unsafe_bundle"


@pytest.mark.parametrize("name", ["named-pipe.json", "named-pipe.zip"])
def test_special_files_are_not_opened_by_reader_or_format_sniff(tmp_path, monkeypatch, name):
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("platform has no named pipes")
    source = tmp_path / name
    os.mkfifo(source)
    original_open = os.open

    def guarded_open(path, *args, **kwargs):
        if os.fspath(path) == str(source):
            pytest.fail("special files must be refused before opening a descriptor")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    assert not looks_like_otlp_source(source)
    with pytest.raises(LoadError) as caught:
        read_otlp_capture(source)
    assert caught.value.code == "load_failed" and caught.value.path == str(source)
