"""Expected evidence limitations are explained; unexpected failures stay private."""
from unittest.mock import patch

import pytest

import agent_trajviz as tv
from agent_trajviz.converge.alignment import TranscriptAlignmentUnavailable
from agent_trajviz.insight.comparison import run_comparison
from agent_trajviz.insight.insight import build_ui


def _native():
    return tv.Trajectory.from_steps([{"index": 0, "role": "assistant", "parts": [], "tool_calls": []}])


def test_operation_capture_explains_alignment_limit_without_fabricating_scores(tmp_path):
    from tests.test_otlp_api import _session, _write

    capture = tv.load(_write(tmp_path, _session(1)))
    native = _native()
    result = run_comparison(capture.raw, native.raw, ref_steps=capture.steps, cmp_steps=native.steps)
    assert result["ok"] is False and result["unavailable"] == "transcript_alignment"
    assert "Transcript alignment unavailable" in result["report_html"]
    assert "scores have not been computed" in result["report_html"]
    assert "Run group" in result["report_html"]
    assert capture.view("token_usage").data


@pytest.mark.parametrize("exception", [ValueError("/private/hidden/session.json secret"),
                                       RuntimeError("/private/hidden/session.json secret")])
def test_unexpected_error_message_and_paths_remain_redacted(exception):
    native = _native()
    with patch("agent_trajviz.insight.comparison.build_comparison_report_from_steps", side_effect=exception):
        result = run_comparison(native.raw, native.raw, ref_steps=native.steps, cmp_steps=native.steps)
    assert not result["ok"]
    assert "unavailable" not in result
    assert "Comparison failed" in result["report_html"]
    assert "/private/hidden" not in result["report_html"] and "secret" not in result["report_html"]


def test_known_limitation_never_echoes_caller_supplied_exception_text():
    native = _native()
    with patch("agent_trajviz.insight.comparison.build_comparison_report_from_steps",
               side_effect=TranscriptAlignmentUnavailable("/private/hidden secret")):
        result = run_comparison(native.raw, native.raw, ref_steps=native.steps, cmp_steps=native.steps)
    assert result["unavailable"] == "transcript_alignment"
    assert "/private/hidden" not in result["report_html"] and "secret" not in result["report_html"]


def test_comparison_status_distinguishes_unavailable_alignment_from_failure(tmp_path):
    from tests.test_otlp_api import _session, _write

    source = _write(tmp_path, _session(1), _session(2))
    app = build_ui()
    try:
        callback = next(fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "on_run_comparison")
        result = callback(str(source), None, "", None, None, _native(), False, "session-2")
        assert "Transcript alignment unavailable" in result[3]
        assert "observed metrics" in result[3]
        assert "Comparison failed" not in result[3]
    finally:
        app.close()
