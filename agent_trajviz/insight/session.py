"""Gradio-free load pipeline: ingest a trajectory into a LoadedSession DTO."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Literal

from .analysis import NOISY_ROOT_CAUSE_FORMATS, Analysis
from .loaders import check_format_selection, detect_format, load_trajectory
from .parser import parse_steps

logger = logging.getLogger(__name__)

# Maximum steps to process — keeps rendering, metrics, and charts bounded.
MAX_STEPS = 2000

LoadErrorCode = Literal["not_found", "mismatch", "unknown", "read_error", "analysis_error"]

_NOISY_ROOT_CAUSE_FORMATS = NOISY_ROOT_CAUSE_FORMATS  # kept for importers of the old name


@dataclass(frozen=True)
class LoadError:
    """Failed ingest. Messages are plain text — the UI packer wraps HTML."""

    code: LoadErrorCode
    message: str
    selected: str | None = None
    detected: str | None = None


@dataclass
class LoadedSession:
    """Domain snapshot of a successfully loaded trajectory (no HTML / Plotly / Gradio)."""

    path: str
    raw: dict
    steps: list[dict]
    steps_total: int
    format: str
    message_rows: list[dict]
    metrics: dict
    verdicts: list[dict]
    agent_summaries: list[dict]
    step_analytics: list[dict]
    wall_clock: str
    tool_sequences: list[dict]
    failure_patterns: list[dict]
    file_interactions: list[dict]
    target_files: set[str]
    failure_chains: list
    chain_metrics: dict
    clusters: list
    performance_bottlenecks: list
    diagnostic_metrics: dict
    pressure_series: dict
    pressure_choices: list
    show_root_cause: bool
    plan_history: list
    plan_metrics: dict
    fruitless_streaks: list
    tool_selection: list
    edit_thrash: list
    repeated_searches: list
    # Computed on every load but read by no presenter, chart, report or UI
    # module — only by the two tests that construct a LoadedSession by keyword.
    # Kept rather than deleted: the field has no default, so dropping it is a
    # breaking change to every constructor site, and wiring it to Overview
    # Issues instead would add new antipattern issues to most runs, i.e. a
    # product decision about what the Overview claims, not a cleanup.
    phase_regressions: list
    premature_compactions: list
    truncated: bool = False


def _path_exists(file_path: str) -> bool:
    return os.path.isfile(file_path) or (
        os.path.isdir(file_path) and os.path.isfile(os.path.join(file_path, "session.jsonl"))
    )


def load_session(path: str, format_hint: str = "") -> LoadedSession | LoadError:
    """Load and analyze a trajectory file. Does not build HTML, Plotly, or Gradio updates."""
    if not path or not _path_exists(path):
        return LoadError(code="not_found", message="No file selected or file not found.")

    hint = format_hint or None
    raw = load_trajectory(path, format_hint=hint)
    if raw.get("_error_code") == "mismatch":
        return LoadError(
            code="mismatch",
            message=str(raw.get("_error") or "Format mismatch."),
            selected=raw.get("_selected") or format_hint or None,
            detected=raw.get("_detected") or None,
        )
    if "_error" in raw:
        return LoadError(code="read_error", message=str(raw["_error"]))

    detected = detect_format(raw)
    gate = check_format_selection(detected, format_hint)
    if gate == "unknown":
        return LoadError(
            code="unknown",
            message=(
                "Could not detect trajectory format. "
                "Select a format from the dropdown and try again."
            ),
        )
    if gate == "mismatch":
        return LoadError(
            code="mismatch",
            message=(
                f"Format mismatch: selected {format_hint} but file detected as {detected}."
            ),
            selected=format_hint or None,
            detected=detected,
        )

    # Same promise as the loader: a LoadedSession or a LoadError, never an
    # exception. The step model is type-normalised in parse_steps, so this
    # guard is the backstop for an analyser bug, not the routine path for
    # malformed input. The traceback goes to the log.
    try:
        return build_loaded_session(path, raw, detected=detected)
    except Exception as exc:  # noqa: BLE001 — see above
        logger.exception("could not analyse %s", path)
        return LoadError(
            code="analysis_error",
            message=f"The file loaded but could not be analysed ({type(exc).__name__}: {str(exc)[:160]}).",
        )


def build_loaded_session(path: str, raw: dict, *, detected: str | None = None) -> LoadedSession:
    """Analyze already-loaded *raw* into a ``LoadedSession`` (no file I/O)."""
    detected = detected if detected is not None else detect_format(raw)
    steps = parse_steps(raw)
    steps_total = len(steps)
    truncated = False
    if steps_total > MAX_STEPS:
        steps = steps[:MAX_STEPS]
        truncated = True

    # One orchestration for every consumer: see analysis.Analysis.
    a = Analysis(steps, raw, detected)

    return LoadedSession(
        path=path,
        raw=raw,
        steps=steps,
        steps_total=steps_total,
        format=detected,
        message_rows=a.message_rows,
        metrics=a.metrics,
        verdicts=a.verdicts,
        agent_summaries=a.agent_summaries,
        step_analytics=a.step_analytics,
        wall_clock=a.wall_clock,
        tool_sequences=a.tool_sequences,
        failure_patterns=a.failure_patterns,
        file_interactions=a.file_interactions,
        target_files=a.target_files,
        failure_chains=a.failure_chains,
        chain_metrics=a.chain_metrics,
        clusters=a.clusters,
        performance_bottlenecks=a.performance_bottlenecks,
        diagnostic_metrics=a.diagnostic_metrics,
        pressure_series=a.pressure_series,
        pressure_choices=a.pressure_choices,
        show_root_cause=a.show_root_cause,
        plan_history=a.plan_history,
        plan_metrics=a.plan_metrics,
        fruitless_streaks=a.fruitless_streaks,
        tool_selection=a.tool_selection,
        edit_thrash=a.edit_thrash,
        repeated_searches=a.repeated_searches,
        phase_regressions=a.phase_regressions,
        premature_compactions=a.premature_compactions,
        truncated=truncated,
    )
