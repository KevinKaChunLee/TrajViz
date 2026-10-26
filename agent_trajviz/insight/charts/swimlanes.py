"""Step-axis agent timelines: swimlane, N-run, and tool outcomes."""

from __future__ import annotations

from collections import defaultdict

from ._layout import (
    _add_dummy_marker_legend,
    _apply_chart_layout,
    _apply_dark,
    _empty_figure,
    _truncate_chart_labels,
)
import plotly.graph_objects as go

from .. import chart_data
from ..palette import SESSION_COLORS, TOOL_OUTCOME_COLORS, USER_SWIMLANE_COLOR

USER_SWIMLANE_ID = chart_data.USER_LANE_ID
USER_SWIMLANE_LABEL = chart_data.USER_LANE_LABEL


def build_agent_swimlane_chart(steps: list[dict], dark: bool = False) -> go.Figure:
    """Horizontal swimlane chart showing each agent's active step ranges and token contribution.

    Drawn from :func:`chart_data.agent_swimlane`; see :func:`draw_agent_swimlane`.
    """
    return draw_agent_swimlane(chart_data.agent_swimlane(steps), dark=dark)


def draw_agent_swimlane(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.agent_swimlane` output.

    Human user prompts occupy a dedicated ``user`` lane (top). Main and
    sub-agent lanes sit below, even when only the parent agent is present.
    Task / system / compaction turns that some formats store as ``role=user``
    stay on the agent that owns them.
    """
    lanes = data["lanes"]
    if not lanes:
        fig = _empty_figure(180, "No agent activity recorded.")
        _apply_dark(fig, dark)
        return fig

    fig = go.Figure()
    y_labels: list[str] = []
    for lane in lanes:
        label = lane["label"]
        y_labels.append(label)
        if lane["color_index"] is None:
            hex_c = USER_SWIMLANE_COLOR
        else:
            hex_c = SESSION_COLORS[lane["color_index"] % len(SESSION_COLORS)]
        for seg in lane["segments"]:
            start, end, tok, tools = seg["start"], seg["end"], seg["tokens"], seg["tool_calls"]
            width = end - start + 1
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[width],
                    orientation="h",
                    base=start,
                    marker_color=hex_c,
                    name=label,
                    showlegend=False,
                    text=f"{width} steps, {tok:,} tok" if tok is not None else f"{width} steps",
                    textposition="inside",
                    # First step of this segment drives Workflow jump on click.
                    customdata=[start],
                    hovertext=(
                        f"{label}: steps {start}–{end}<br>"
                        + (f"{tok:,} tokens" if tok is not None else "Tokens unavailable")
                        + f", {tools} tool calls"
                    ),
                    hoverinfo="text",
                )
            )

    _apply_chart_layout(
        fig,
        "Agent Swimlane",
        xaxis="Step Index",
        height=max(160, 80 * max(1, len(lanes))),
        barmode="overlay",
        margin=dict(l=100, r=20, t=40, b=30),
    )
    fig.update_layout(
        yaxis=dict(categoryorder="array", categoryarray=y_labels, automargin=True),
        clickmode="event",
    )
    _apply_dark(fig, dark)
    return fig


def build_run_group_agent_timeline(
    runs: list[dict],
    *,
    dark: bool = False,
) -> go.Figure:
    """Compact N-run agent timeline: one horizontal lane per run.

    Drawn from :func:`chart_data.run_timelines`; see :func:`draw_run_timelines`.
    """
    return draw_run_timelines(chart_data.run_timelines(runs), dark=dark)


def draw_run_timelines(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.run_timelines` output.

    Each run dict needs ``label`` (or ``run_id``) and ``steps`` (parsed).
    Segments within a lane are colored by session / subagent / agent mode
    (see ``_timeline_context``); shared display names share colors across runs.
    """
    runs = data["runs"]
    if not runs:
        fig = _empty_figure(200, "No agent activity across runs.")
        _apply_dark(fig, dark)
        return fig
    palette_index = {bucket: i for i, bucket in enumerate(data["buckets"])}

    fig = go.Figure()
    legend_seen: set[str] = set()
    y_labels: list[str] = []

    # First loaded run at top
    for run in reversed(runs):
        label = run["label"]
        y_labels.append(label)
        for seg in run["segments"]:
            dlabel, seg_start, seg_end = seg["agent_label"], seg["start"], seg["end"]
            width = seg_end - seg_start + 1
            pal_i = palette_index.get(seg["bucket"], palette_index.get("main", 0))
            hex_c = SESSION_COLORS[pal_i % len(SESSION_COLORS)]
            show_leg = dlabel not in legend_seen
            if show_leg:
                legend_seen.add(dlabel)
            fig.add_trace(
                go.Bar(
                    y=[label],
                    x=[width],
                    orientation="h",
                    base=seg_start,
                    marker_color=hex_c,
                    name=dlabel,
                    legendgroup=dlabel,
                    showlegend=show_leg,
                    hovertext=(
                        f"<b>{label}</b><br>{dlabel}<br>"
                        f"steps {seg_start}–{seg_end} ({width})<br>"
                        + (f"{seg['tokens']:,} tokens" if seg["tokens"] is not None else "Tokens unavailable")
                        + f", {seg['tool_calls']} tool calls"
                    ),
                    hoverinfo="text",
                )
            )

    n = len(runs)
    _apply_chart_layout(
        fig,
        "Agent timeline (by run)",
        xaxis="Step Index",
        height=max(220, 56 * n + 100),
        barmode="overlay",
        margin=dict(l=120, r=20, t=90, b=40),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=y_labels,
            automargin=True,
        ),
    )
    _apply_dark(fig, dark)
    return fig


def build_tool_outcome_timeline(steps: list[dict], dark: bool = False) -> go.Figure:
    """Scatter of tool outcomes: by agent (color) + success/fail (shape) when multi-agent.

    Drawn from :func:`chart_data.tool_outcomes`; see :func:`draw_tool_outcomes`.
    """
    if not steps:
        fig = _empty_figure(340)
        _apply_dark(fig, dark)
        return fig
    return draw_tool_outcomes(chart_data.tool_outcomes(steps), dark=dark)


def draw_tool_outcomes(data: dict, dark: bool = False) -> go.Figure:
    """Draw :func:`chart_data.tool_outcomes` output.

    Failures always get a red marker border so they stay visible when fill is
    an agent color.
    """
    color_index = {a["id"]: i for i, a in enumerate(data["agents"])}
    labels = {a["id"]: a["label"] for a in data["agents"]}
    fail_border = TOOL_OUTCOME_COLORS["failure"]

    by_agent: dict[str, list[tuple[int, str, bool | None, str]]] = defaultdict(list)
    tool_names: set[str] = set()
    saw_ok = saw_fail = saw_unknown = False
    for call in data["calls"]:
        by_agent[call["agent"]].append((call["step"], call["tool"], call["ok"], call["hint"]))
        tool_names.add(call["tool"])
        if call["ok"] is True:
            saw_ok = True
        elif call["ok"] is False:
            saw_fail = True
        else:
            saw_unknown = True

    if not by_agent:
        fig = _empty_figure(340, "No tool calls recorded in this trajectory.")
        _apply_dark(fig, dark)
        return fig

    # Truncate the y labels as a set, not per call: two names agreeing on their
    # first 27 characters would otherwise land on one merged row.
    ordered = sorted(tool_names)
    label_of = dict(zip(ordered, _truncate_chart_labels(ordered), strict=True))
    by_agent = {
        agent: [(idx, label_of[name], ok, hint) for idx, name, ok, hint in calls]
        for agent, calls in by_agent.items()
    }

    has_agents = len(by_agent) > 1
    fig = go.Figure()

    if has_agents:
        for agent_id in sorted(by_agent.keys(), key=lambda a: color_index.get(a, 10**9)):
            group = by_agent[agent_id]
            color = SESSION_COLORS[color_index.get(agent_id, 0) % len(SESSION_COLORS)]
            label = labels.get(agent_id, agent_id or "main")
            oks = [p[2] for p in group]
            hovers = [
                ("Success" if p[2] is True else "Failure" if p[2] is False else "Outcome unavailable")
                + (f"<br>{p[3]}" if p[3] else "")
                for p in group
            ]
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in group],
                    y=[p[1] for p in group],
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=color,
                        size=9,
                        symbol=["circle" if ok is True else "x" if ok is False else "diamond-open" for ok in oks],
                        line=dict(
                            width=[2 if ok is False else 0 for ok in oks],
                            color=[
                                fail_border if ok is False else "rgba(0,0,0,0)"
                                for ok in oks
                            ],
                        ),
                    ),
                    # Step index in customdata drives Workflow jump on click.
                    customdata=[p[0] for p in group],
                    hovertext=hovers,
                    hovertemplate=(
                        f"{label}<br>Step %{{customdata}}<br>%{{y}}"
                        f"<br>%{{hovertext}}<extra></extra>"
                    ),
                )
            )
        shape_entries = []
        if saw_ok:
            shape_entries.append(
                ("Success (circle)", TOOL_OUTCOME_COLORS["success"], "circle")
            )
        if saw_fail:
            shape_entries.append(
                (
                    "Failure (x)",
                    TOOL_OUTCOME_COLORS["failure"],
                    "x",
                    dict(width=2, color=fail_border),
                )
            )
        if saw_unknown:
            shape_entries.append(("Outcome unavailable (diamond)", "#94a3b8", "diamond-open"))
        _add_dummy_marker_legend(fig, shape_entries, legendgroup="outcome-shape")
        title = "Tool Outcome Timeline by Agent"
    else:
        group = next(iter(by_agent.values()))
        for ok, name, color, symbol in (
            (True, "Success", TOOL_OUTCOME_COLORS["success"], "circle"),
            (False, "Failure", TOOL_OUTCOME_COLORS["failure"], "x"),
            (None, "Outcome unavailable", "#94a3b8", "diamond-open"),
        ):
            subset = [p for p in group if p[2] is ok]
            if not subset:
                continue
            marker = dict(color=color, size=9, symbol=symbol)
            if ok is False:
                marker["line"] = dict(width=2, color=fail_border)
            hovers = [
                name + (f"<br>{p[3]}" if p[3] else "") for p in subset
            ]
            fig.add_trace(
                go.Scatter(
                    x=[p[0] for p in subset],
                    y=[p[1] for p in subset],
                    mode="markers",
                    name=name,
                    marker=marker,
                    customdata=[p[0] for p in subset],
                    hovertext=hovers,
                    hovertemplate=(
                        "Step %{customdata}<br>%{y}<br>%{hovertext}<extra></extra>"
                    ),
                )
            )
        title = "Tool Outcome Timeline"

    _apply_chart_layout(
        fig,
        title,
        xaxis="Step",
        height=max(300, 30 * len(tool_names)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_layout(clickmode="event")
    _apply_dark(fig, dark)
    return fig
