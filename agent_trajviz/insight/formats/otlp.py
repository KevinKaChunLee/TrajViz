"""Decode saved OTLP JSON without an OpenTelemetry SDK or network access.

Transport decoding deliberately knows nothing about a particular agent. Bytes
AnyValues use a JSON-safe ``{"$otlp_bytes": base64}`` representation; the three
non-finite double spellings remain strings, as in OTLP's JSON encoding.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from ...errors import LoadError

MAX_RECORDS = 100_000
MAX_ATTRIBUTES = 4096
MAX_DEPTH = 32
MAX_TEXT_BYTES = 1024 * 1024
_INT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_HEX = re.compile(r"[0-9a-fA-F]+\Z")
_MISSING = object()


@dataclass
class OtlpCapture:
    """Decoded transport records plus lossless requests and content identity."""

    spans: list[dict[str, Any]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    sha256: str = ""


def _fail(message: str, path: str) -> LoadError:
    return LoadError(f"Invalid OTLP data in {path}: {message}", code="invalid_otlp", path=path)


def _get(value: dict, camel: str, default: Any = None) -> Any:
    """Accept both protobuf JSON spellings without changing producer fields."""
    if camel in value:
        return value[camel]
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", camel).lower()
    return value.get(snake, default)


def _mapping(value: Any, label: str, path: str) -> dict:
    if not isinstance(value, dict):
        raise _fail(f"{label} must be an object", path)
    return value


def _array(value: Any, label: str, path: str, limit: int = MAX_RECORDS) -> list:
    if not isinstance(value, list):
        raise _fail(f"{label} must be an array", path)
    if len(value) > limit:
        raise LoadError(f"OTLP {label} exceeds the {limit} item limit", code="size_limit", path=path)
    return value


def _text(value: Any, label: str, path: str) -> str:
    if not isinstance(value, str):
        raise _fail(f"{label} must be a string", path)
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as exc:
        raise _fail(f"{label} contains an invalid Unicode code point", path) from exc
    if size > MAX_TEXT_BYTES:
        raise LoadError(f"OTLP {label} exceeds the string size limit", code="size_limit", path=path)
    return value


def _integer(value: Any, label: str, path: str, *, signed: bool = False, bits: int = 64) -> int:
    if isinstance(value, bool) or not (
        isinstance(value, int) or isinstance(value, str) and _INT.fullmatch(value)
    ):
        raise _fail(f"{label} must be an integer or decimal integer string", path)
    try:
        result = int(value)
    except ValueError as exc:
        raise _fail(f"{label} is not a valid integer", path) from exc
    lower, upper = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)
    if not lower <= result <= upper:
        raise _fail(f"{label} is outside the {bits}-bit range", path)
    return result


def _identifier(value: Any, label: str, width: int, path: str, *, optional: bool = False) -> str:
    if optional and value in (None, ""):
        return ""
    if not isinstance(value, str) or len(value) != width or not _HEX.fullmatch(value) or not int(value, 16):
        raise _fail(f"{label} must be a nonzero {width}-character hexadecimal ID", path)
    return value.lower()


def _attributes(value: Any, path: str, depth: int = 0) -> dict[str, Any]:
    result = {}
    for item in _array(value, "attributes", path, MAX_ATTRIBUTES):
        item = _mapping(item, "attribute", path)
        key = _text(item.get("key"), "attribute key", path)
        if not key or key in result:
            raise _fail("attribute keys must be nonempty and unique", path)
        result[key] = _any_value(item.get("value", {}), path, depth + 1)
    return result


def _any_value(value: Any, path: str, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        raise LoadError("OTLP attribute nesting limit exceeded", code="size_limit", path=path)
    value = _mapping(value, "AnyValue", path)
    variants = ("stringValue", "boolValue", "intValue", "doubleValue", "arrayValue", "kvlistValue", "bytesValue")
    present = [(key, _get(value, key, _MISSING)) for key in variants]
    present = [(key, item) for key, item in present if item is not _MISSING]
    if not present and not value:
        return None
    if len(present) != 1:
        raise _fail("AnyValue must contain exactly one supported value", path)
    kind, item = present[0]
    if kind == "stringValue":
        return _text(item, kind, path)
    if kind == "boolValue":
        if not isinstance(item, bool):
            raise _fail("boolValue must be a boolean", path)
        return item
    if kind == "intValue":
        return _integer(item, kind, path, signed=True)
    if kind == "doubleValue":
        if isinstance(item, str) and item in ("NaN", "Infinity", "-Infinity"):
            return item
        if isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item):
            raise _fail("doubleValue must be a finite number or an OTLP non-finite spelling", path)
        return item
    if kind == "arrayValue":
        array = _mapping(item, kind, path)
        return [_any_value(entry, path, depth + 1) for entry in _array(array.get("values", []), kind, path)]
    if kind == "kvlistValue":
        return _attributes(_mapping(item, kind, path).get("values", []), path, depth + 1)
    encoded = _text(item, kind, path)
    try:
        base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise _fail("bytesValue must be base64", path) from exc
    return {"$otlp_bytes": encoded}


def _enum(value: Any, names: dict[str, int], label: str, path: str) -> int:
    if isinstance(value, str) and value in names:
        return names[value]
    result = _integer(value, label, path, bits=32)
    if result not in names.values():
        raise _fail(f"unknown {label} enum value", path)
    return result


def _scope(value: Any, path: str) -> dict:
    obj = _mapping(value, "scope", path)
    result = dict(obj)
    result["attributes"] = _attributes(obj.get("attributes", []), path)
    for key in ("name", "version"):
        if key in result:
            result[key] = _text(result[key], f"scope {key}", path)
    return result


def _span(obj: dict, common: dict, path: str, source: dict) -> dict:
    status = _mapping(obj.get("status", {}), "span status", path)
    events = []
    for event in _array(obj.get("events", []), "span events", path, 10_000):
        event = _mapping(event, "span event", path)
        events.append({
            "name": _text(event.get("name", ""), "event name", path),
            "time_unix_nano": _integer(_get(event, "timeUnixNano", 0), "event time", path),
            "attributes": _attributes(event.get("attributes", []), path),
            "dropped_attributes_count": _integer(_get(event, "droppedAttributesCount", 0), "dropped event attributes", path, bits=32),
            "raw": event,
        })
    links = []
    for link in _array(obj.get("links", []), "span links", path, 10_000):
        link = _mapping(link, "span link", path)
        links.append({
            "trace_id": _identifier(_get(link, "traceId"), "link traceId", 32, path),
            "span_id": _identifier(_get(link, "spanId"), "link spanId", 16, path),
            "attributes": _attributes(link.get("attributes", []), path), "raw": link,
        })
    return {
        **common,
        "trace_id": _identifier(_get(obj, "traceId"), "traceId", 32, path),
        "span_id": _identifier(_get(obj, "spanId"), "spanId", 16, path),
        "parent_span_id": _identifier(_get(obj, "parentSpanId"), "parentSpanId", 16, path, optional=True),
        "trace_state": _text(_get(obj, "traceState", ""), "trace state", path),
        "flags": _integer(obj.get("flags", 0), "span flags", path, bits=32),
        "name": _text(obj.get("name", ""), "span name", path),
        "kind": _enum(obj.get("kind", 0), {
            "SPAN_KIND_UNSPECIFIED": 0, "SPAN_KIND_INTERNAL": 1, "SPAN_KIND_SERVER": 2,
            "SPAN_KIND_CLIENT": 3, "SPAN_KIND_PRODUCER": 4, "SPAN_KIND_CONSUMER": 5,
        }, "span kind", path),
        "start_time_unix_nano": _integer(_get(obj, "startTimeUnixNano", 0), "span start time", path),
        "end_time_unix_nano": _integer(_get(obj, "endTimeUnixNano", 0), "span end time", path),
        "attributes": _attributes(obj.get("attributes", []), path),
        "dropped_attributes_count": _integer(_get(obj, "droppedAttributesCount", 0), "dropped span attributes", path, bits=32),
        "dropped_events_count": _integer(_get(obj, "droppedEventsCount", 0), "dropped span events", path, bits=32),
        "dropped_links_count": _integer(_get(obj, "droppedLinksCount", 0), "dropped span links", path, bits=32),
        "events": events, "links": links,
        "status": {
            "code": _enum(status.get("code", 0), {
                "STATUS_CODE_UNSET": 0, "STATUS_CODE_OK": 1, "STATUS_CODE_ERROR": 2,
            }, "status code", path),
            "message": _text(status.get("message", ""), "status message", path),
        },
        "raw": obj, "source": source,
    }


def _log(obj: dict, common: dict, path: str, source: dict) -> dict:
    return {
        **common,
        "time_unix_nano": _integer(_get(obj, "timeUnixNano", 0), "log time", path),
        "observed_time_unix_nano": _integer(_get(obj, "observedTimeUnixNano", 0), "observed log time", path),
        "trace_id": _identifier(_get(obj, "traceId"), "log traceId", 32, path, optional=True),
        "span_id": _identifier(_get(obj, "spanId"), "log spanId", 16, path, optional=True),
        "body": _any_value(obj.get("body", {}), path),
        "event_name": _text(_get(obj, "eventName", ""), "log event name", path),
        "attributes": _attributes(obj.get("attributes", []), path),
        "dropped_attributes_count": _integer(_get(obj, "droppedAttributesCount", 0), "dropped log attributes", path, bits=32),
        "flags": _integer(obj.get("flags", 0), "log flags", path, bits=32),
        "severity_number": _integer(_get(obj, "severityNumber", 0), "severity number", path, bits=32),
        "severity_text": _text(_get(obj, "severityText", ""), "severity text", path),
        "raw": obj, "source": source,
    }


def looks_like_otlp_payload(payload: Any) -> bool:
    """Recognize a transport envelope or explicit bundle, never a user directory."""
    return isinstance(payload, dict) and (
        any(key in payload for key in ("resourceSpans", "resourceLogs", "resource_spans", "resource_logs"))
        or payload.get("type") == "agent-trajviz-otlp-bundle"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _json(text: str) -> Any:
    def nonfinite(value: str) -> None:
        raise ValueError(f"unquoted non-finite number {value}")
    def number(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("number exceeds the finite float range")
        return parsed
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=nonfinite, parse_float=number)


def decode_otlp_bytes(data: bytes, path: str = "<memory>") -> OtlpCapture:
    """Decode one complete JSON request or Collector newline-delimited requests."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise _fail("expected UTF-8 JSON; binary/compressed OTLP is unsupported", path) from exc
    try:
        requests = [(1, _json(text))]
    except (ValueError, RecursionError):
        requests = []
        # JSON strings may contain raw Unicode line separators. Only CR/LF
        # delimit JSONL records; str.splitlines() also splits valid content.
        for line, part in enumerate(re.split(r"\r\n|[\r\n]", text), 1):
            if not part.strip():
                continue
            try:
                requests.append((line, _json(part)))
            except (ValueError, RecursionError) as exc:
                raise _fail(f"invalid JSON on line {line}: {exc}", path) from exc
            if len(requests) > MAX_RECORDS:
                raise LoadError("OTLP request count limit exceeded", code="size_limit", path=path) from None
    if not requests:
        raise _fail("the file is empty", path)
    capture = OtlpCapture(raw={"requests": [], "source_files": [path]}, sha256=hashlib.sha256(data).hexdigest())
    index = 0
    try:
        for line, request in requests:
            if not looks_like_otlp_payload(request) or request.get("type") == "agent-trajviz-otlp-bundle":
                raise LoadError(
                    f"Expected an OTLP trace/log request in {path} on line {line}; metrics, debug output, "
                    "binary framing and custom encodings are unsupported", code="unsupported_otlp", path=path,
                )
            capture.raw["requests"].append(request)
            for resource_key, scopes_key, records_key, decoder, output in (
                ("resourceSpans", "scopeSpans", "spans", _span, capture.spans),
                ("resourceLogs", "scopeLogs", "logRecords", _log, capture.logs),
            ):
                for resource in _array(_get(request, resource_key, []), resource_key, path):
                    resource = _mapping(resource, resource_key, path)
                    attrs = _attributes(_mapping(resource.get("resource", {}), "resource", path).get("attributes", []), path)
                    for scope in _array(_get(resource, scopes_key, []), scopes_key, path):
                        scope = _mapping(scope, scopes_key, path)
                        common = {
                            "resource": attrs, "scope": _scope(scope.get("scope", {}), path),
                            "resource_raw": resource.get("resource", {}), "scope_raw": scope.get("scope", {}),
                            "resource_schema_url": _text(_get(resource, "schemaUrl", ""), "resource schema URL", path),
                            "schema_url": _text(_get(scope, "schemaUrl", _get(resource, "schemaUrl", "")), "schema URL", path),
                        }
                        for obj in _array(_get(scope, records_key, []), records_key, path):
                            index += 1
                            if index > MAX_RECORDS:
                                raise LoadError("OTLP record limit exceeded", code="size_limit", path=path)
                            output.append(decoder(_mapping(obj, records_key, path), common, path, {
                                "path": path, "line": line, "index": index - 1,
                            }))
    except (RecursionError, OverflowError) as exc:
        raise _fail("data exceeds safe nesting or numeric limits", path) from exc
    return deduplicate_capture(capture)


def _record_key(record: dict) -> str:
    return json.dumps({key: value for key, value in record.items() if key not in ("source", "sources", "raw")},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def deduplicate_capture(capture: OtlpCapture) -> OtlpCapture:
    """Make repeated rotations and batch order deterministic; disclose conflicts."""
    spans: dict[tuple[str, str], tuple[str, dict]] = {}
    for span in capture.spans:
        identity = span["trace_id"], span["span_id"]
        key = _record_key(span)
        if identity in spans:
            old_key, old = spans[identity]
            sources = old.get("sources", [old["source"]]) + span.get("sources", [span["source"]])
            if old_key != key:
                capture.warnings.append(
                    f"Conflicting duplicate OTLP span {identity[0]}/{identity[1]}; "
                    "selected one canonical record without adding usage."
                )
            chosen = span if key < old_key else old
            chosen["sources"] = sorted(sources, key=lambda source: (source["path"], source["line"], source["index"]))
            spans[identity] = min(old_key, key), chosen
        else:
            spans[identity] = key, span
    capture.spans = sorted((pair[1] for pair in spans.values()), key=lambda span: (
        span["start_time_unix_nano"], span["trace_id"], span["span_id"],
    ))
    # Logs do not carry an intrinsic record ID. Compare their complete canonical
    # content including timestamps, rather than deduplicating event.sequence.
    logs: dict[str, dict] = {}
    for log in capture.logs:
        key = _record_key(log)
        if key in logs:
            previous = logs[key]
            previous["sources"] = previous.get("sources", [previous["source"]]) + log.get("sources", [log["source"]])
        else:
            logs[key] = log
    capture.logs = sorted(logs.values(), key=lambda log: (
        log["time_unix_nano"] or log["observed_time_unix_nano"], log["trace_id"], log["span_id"], _record_key(log),
    ))
    capture.warnings = sorted(set(capture.warnings))
    return capture
