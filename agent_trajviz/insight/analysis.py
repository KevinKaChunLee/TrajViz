"""Every analysis of one trajectory, computed on first use and cached.

This is the single orchestration of the analysis functions: which detector
runs on which steps, and which results feed which. The public
:class:`agent_trajviz.Trajectory`, the JSON views and the dashboard's
``LoadedSession`` all read from an :class:`Analysis`, so none of them can
compute a figure the others do not.

Nothing here builds HTML, figures or UI state.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from .analytics import compute_step_analytics
from .context_usage import (
    PRESSURE_ALL_AGENTS,
    context_pressure_series,
    detect_premature_compactions,
    pressure_agent_choices,
)
from .diagnostics import (
    annotate_clusters_with_agents,
    cluster_errors,
    compute_failure_chain_metrics,
    detect_failure_chains,
    detect_performance_bottlenecks,
    extract_file_interactions,
    identify_target_files,
    link_chains_to_agents,
)
from .metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_diagnostic_metrics,
    compute_health_verdict,
    compute_metrics,
    wall_clock_fmt,
)
from .patterns import (
    build_structural_phase_segments,
    compute_plan_metrics,
    detect_edit_thrash,
    detect_failure_patterns,
    detect_fruitless_streaks,
    detect_phase_anomalies,
    detect_repeated_searches,
    detect_tool_selection_antipatterns,
    detect_tool_sequences,
    extract_plan_history,
)

# Formats whose error text is too uniform for clustering to name a root cause.
NOISY_ROOT_CAUSE_FORMATS = frozenset({"opencode", "codearts", "codex"})


def hotspots(message_rows: list[dict], limit: int = 5) -> dict[str, list[dict]]:
    """The steps worth looking at first, from the per-step metrics rows.

    ``slowest`` and ``heaviest`` are the *limit* longest and most token-heavy
    steps. ``lowest_cache_read`` is the *limit* assistant steps with the
    smallest cache-read share -- empty when no step reports any cache read,
    because a ranking of zeros points at nothing. Rows whose cache share is
    unknown (a self-inconsistent record) are never ranked by it.
    """
    with_duration = [r for r in message_rows if r.get("duration") is not None]
    with_cache = [
        r for r in message_rows
        if r.get("role") == "assistant" and r["tokens_total"] > 0 and r["cache_ratio"] is not None
    ]
    return {
        "slowest": sorted(with_duration, key=lambda r: r["duration"], reverse=True)[:limit],
        "heaviest": sorted(message_rows, key=lambda r: r["tokens_total"], reverse=True)[:limit],
        "lowest_cache_read": (
            sorted(with_cache, key=lambda r: r["cache_ratio"])[:limit]
            if with_cache and any(r["cache_ratio"] > 0 for r in with_cache) else []
        ),
    }


class Analysis:
    """Lazily computed analyses of *steps* (parsed) from the export *raw*."""

    def __init__(self, steps: list[dict], raw: dict, fmt: str) -> None:
        self.steps = steps
        self.raw = raw
        self.format = fmt

    # -- per-step and session metrics ------------------------------------
    @cached_property
    def message_rows(self) -> list[dict]:
        return build_message_metrics(self.steps)

    @cached_property
    def metrics(self) -> dict:
        return compute_metrics(self.steps, self.raw, message_rows=self.message_rows)

    @cached_property
    def wall_clock(self) -> str:
        return wall_clock_fmt(self.metrics)[1]

    @cached_property
    def step_analytics(self) -> list[dict]:
        return compute_step_analytics(self.steps)

    @cached_property
    def verdicts(self) -> list[dict]:
        return compute_health_verdict(self.metrics, self.step_analytics if self.steps else [])

    @cached_property
    def agent_summaries(self) -> list[dict]:
        return compute_agent_summary(self.steps, self.raw)

    @cached_property
    def hotspots(self) -> dict[str, list[dict]]:
        return hotspots(self.message_rows)

    # -- diagnostics ------------------------------------------------------
    @cached_property
    def file_interactions(self) -> list[dict]:
        return extract_file_interactions(self.steps)

    @cached_property
    def target_files(self) -> set[str]:
        return identify_target_files(self.steps)

    @cached_property
    def failure_chains(self) -> list:
        return link_chains_to_agents(detect_failure_chains(self.steps), self.steps, self.agent_summaries)

    @cached_property
    def chain_metrics(self) -> dict:
        assistant_steps = sum(1 for s in self.steps if s.get("role") == "assistant")
        return compute_failure_chain_metrics(self.failure_chains, assistant_steps)

    @cached_property
    def clusters(self) -> list:
        return annotate_clusters_with_agents(cluster_errors(self.steps), self.steps, self.agent_summaries)

    @property
    def show_root_cause(self) -> bool:
        return self.format not in NOISY_ROOT_CAUSE_FORMATS

    @cached_property
    def performance_bottlenecks(self) -> list:
        return detect_performance_bottlenecks(self.steps, self.step_analytics)

    @cached_property
    def diagnostic_metrics(self) -> dict:
        traj = self.raw.get("trajectory") or self.raw.get("messages") or []
        return compute_diagnostic_metrics(
            self.steps,
            traj if isinstance(traj, list) else [],
            tool_fail=self.metrics.get("tool_fail"),
        )

    # -- context window ---------------------------------------------------
    @cached_property
    def pressure_series(self) -> dict:
        return context_pressure_series(self.steps, agent_key=PRESSURE_ALL_AGENTS, raw=self.raw)

    @cached_property
    def pressure_choices(self) -> list:
        return pressure_agent_choices(self.steps)

    @cached_property
    def premature_compactions(self) -> list:
        return detect_premature_compactions(self.steps, self.raw)

    # -- plan and phases --------------------------------------------------
    @cached_property
    def plan_history(self) -> list:
        return extract_plan_history(self.steps)

    @cached_property
    def plan_metrics(self) -> dict:
        return compute_plan_metrics(self.plan_history)

    @cached_property
    def structural_phases(self) -> list[dict]:
        return build_structural_phase_segments(self.steps)

    @cached_property
    def phase_regressions(self) -> list:
        return [
            a for a in detect_phase_anomalies(self.steps, self.structural_phases)
            if a.get("category") == "unintentional_drift"
        ]

    # -- behavioural patterns ---------------------------------------------
    @cached_property
    def tool_sequences(self) -> list[dict]:
        return detect_tool_sequences(self.steps)

    @cached_property
    def failure_patterns(self) -> list[dict]:
        return detect_failure_patterns(self.steps)

    @cached_property
    def fruitless_streaks(self) -> list:
        return detect_fruitless_streaks(self.steps)

    @cached_property
    def tool_selection(self) -> list:
        return detect_tool_selection_antipatterns(self.steps)

    @cached_property
    def edit_thrash(self) -> list:
        return detect_edit_thrash(self.steps)

    @cached_property
    def repeated_searches(self) -> list:
        return detect_repeated_searches(self.steps)

    # -- triage -------------------------------------------------------------
    @cached_property
    def issues(self) -> list:
        """Ranked issues (see :mod:`.issues`): errors, then waste, then latency."""
        from .issues import collect_overview_issues, rank_issues

        return rank_issues(collect_overview_issues(self))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        computed: list[Any] = [k for k in vars(self) if k not in ("steps", "raw", "format")]
        return f"Analysis(format={self.format!r}, steps={len(self.steps)}, computed={computed})"
