"""The data behind every chart, as plain JSON-serializable dicts.

Each function computes exactly what one chart shows; the chart is drawn from
the returned dict and nothing else, so a consumer reading the JSON and a
reader looking at the figure see the same numbers. No figures, colours or
HTML here -- an agent's position in ``agents`` is all a renderer needs to
pick a colour.

Steps are addressed two ways: ``position`` (0-based order in the loaded
trajectory, the charts' x axis) and ``step`` (the step's own ``index``).
"""

from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict

from agent_trajviz.tool_vocab import parse_skill_name

from .metrics import (
    TOKEN_FIELDS,
    spawn_wait_seconds,
    step_duration_excluding_spawn,
    tool_call_stats_duration_ms,
)
from .parser import token_segments, usable_token_count
from .shell_cmd import tool_call_hint, tool_chart_name
from .step_errors import step_error_kind
from .timeline_agents import _legend_label, bind_timeline_agents
from .tool_failure import tool_call_failed
from .workflow_role import workflow_role

#: Lane id and label of the human user on the agent swimlane.
USER_LANE_ID = "__user__"
USER_LANE_LABEL = "user"


def _agent_list(color_map: dict[str, int], labels: dict[str, str]) -> list[dict]:
    """Agents in palette order: ``[{"id", "label"}]``; a renderer colours by position."""
    return [
        {"id": aid, "label": _legend_label(aid, labels)}
        for aid in sorted(color_map, key=lambda a: color_map[a])
    ]


def impossible_token_fields(tokens: dict) -> list[tuple[str, object]]:
    """``(field, reported value)`` for each token value that is not a count.

    The rule is :func:`metrics.sum_usable_tokens`' own, so the steps flagged
    here are exactly the steps the session totals call partial.
    """
    return [
        (field, tokens.get(field))
        for field in TOKEN_FIELDS
        if tokens.get(field) not in (None, 0) and usable_token_count(tokens.get(field)) is None
    ]


def detect_outliers(values: list[float], threshold: float = 2.0) -> list[tuple[int, float, str]]:
    """Return (index, value, label) for values exceeding *threshold* σ above the mean.

    Returns an empty list for fewer than 10 values or no outliers.
    """
    if len(values) < 10:
        return []
    clean = [v for v in values if v is not None and v > 0]
    if len(clean) < 5:
        return []
    mean = statistics.mean(clean)
    stdev = statistics.stdev(clean) if len(clean) > 1 else 0
    if stdev == 0:
        return []
    outliers = []
    for i, v in enumerate(values):
        if v is not None and v > 0 and (v - mean) > threshold * stdev:
            outliers.append((i, v, "spike"))
    return outliers


def token_usage(steps: list[dict], fmt: str | None = None) -> dict:
    """Per-step token breakdown in disjoint segments that add up to the total.

    A step whose export reports an impossible count (``invalid`` lists the
    fields and values) has no drawable breakdown: its segments are zero and
    only ``usable_total`` -- the reported total, when that is itself a count --
    describes it. ``has_breakdown`` is false when the format reports totals
    only.
    """
    rows = []
    for position, s in enumerate(steps):
        tok = s["tokens"]
        seg = token_segments(
            tok["total"], tok["input"], tok["output"], tok.get("reasoning", 0),
            tok["cache_read"], tok.get("cache_write", 0) or 0,
        )
        invalid = impossible_token_fields(tok)
        if invalid:
            seg.update(fresh=0, cache_read=0, cache_write=0, output=0, reasoning=0)
        rows.append({
            "position": position,
            "step": s.get("index", position),
            "fresh_input": seg["fresh"],
            "cache_read": seg["cache_read"],
            "cache_write": seg["cache_write"],
            "output": seg["output"],
            "reasoning": seg["reasoning"],
            "total": tok["total"],
            "usable_total": usable_token_count(tok["total"]),
            "invalid": [[field, value] for field, value in invalid],
        })
    has_breakdown = any(
        s["tokens"]["input"] > 0 or s["tokens"]["output"] > 0 or s["tokens"]["cache_read"] > 0 for s in steps
    )
    return {"format": fmt, "has_breakdown": has_breakdown, "steps": rows}


def step_durations(steps: list[dict]) -> dict:
    """Per-step duration excluding sub-agent spawn wait, with error kind and spikes.

    ``duration_s`` is ``None`` when the step has no duration; ``spawn_adjusted``
    marks steps whose wait on a child agent was subtracted; ``error_kind`` is
    ``"system"`` (scaffold tool), ``"tool"`` (agentic tool) or ``None``;
    ``outlier`` marks durations more than 2σ above the mean. ``average_s`` is
    the mean over steps that have a duration.
    """
    chart_durs = [step_duration_excluding_spawn(s) for s in steps]
    durations = [d if d is not None else 0 for d in chart_durs]
    real = [d for d in chart_durs if d is not None]
    outliers = {idx for idx, _, _ in detect_outliers(durations)}
    rows = []
    for i, (s, d) in enumerate(zip(steps, chart_durs, strict=True)):
        rows.append({
            "position": i,
            "step": int(s.get("index", i)),
            "duration_s": d,
            "spawn_adjusted": d is not None and spawn_wait_seconds(s) > 0,
            "error_kind": step_error_kind(s),
            "outlier": i in outliers,
        })
    return {"average_s": sum(real) / len(real) if real else 0, "steps": rows}


def tool_usage(steps: list[dict]) -> dict:
    """Tool calls per tool, split by agent. Tools are listed in first-use order."""
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    by_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = tool_chart_name(tc)
            by_agent[agent][name] += 1
            totals[name] += 1
    return {
        "agents": _agent_list(color_map, labels),
        "tools": [
            {"tool": name, "calls": n,
             "by_agent": {aid: by_agent[aid][name] for aid in by_agent if by_agent[aid].get(name)}}
            for name, n in totals.items()
        ],
    }


def tool_durations(steps: list[dict]) -> dict:
    """Timed tool calls per tool. Spawn tools are omitted (their time is the child's).

    Per-call ``seconds`` are summed per tool, so calls that ran in parallel
    each count; tools are listed in first-use order.
    """
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    calls: dict[str, list[dict]] = defaultdict(list)
    totals: dict[str, float] = defaultdict(float)
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        step_idx = int(s.get("index", i))
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            ms = tool_call_stats_duration_ms(tc)
            if ms is None:
                continue
            name = tool_chart_name(tc)
            secs = ms / 1000.0
            calls[name].append({"step": step_idx, "seconds": secs, "agent": agent})
            totals[name] += secs
    return {
        "agents": _agent_list(color_map, labels),
        "tools": [{"tool": name, "total_s": totals[name], "calls": calls[name]} for name in totals],
    }


def collect_skill_calls(steps: list[dict], agent_id_of) -> Counter[tuple[str, str]]:
    """Count ``(agent_id, skill_id)`` pairs from Skill-tool invocations."""
    counts: Counter[tuple[str, str]] = Counter()
    for step in steps:
        if not isinstance(step, dict):
            continue
        agent = agent_id_of(step)
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            skill = parse_skill_name(str(tc.get("tool_name") or ""), tc.get("input"))
            if skill:
                counts[(agent, skill)] += 1
    return counts


def skill_calls(steps: list[dict]) -> dict:
    """Which agent invoked which skill, and how often (first-occurrence order)."""
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    counts = collect_skill_calls(steps, agent_id_of)
    return {
        "agents": _agent_list(color_map, labels),
        "links": [{"agent": agent, "skill": skill, "calls": n} for (agent, skill), n in counts.items()],
    }


def tool_outcomes(steps: list[dict]) -> dict:
    """Every tool call with its step, agent and outcome (``ok``), in step order.

    ``ok`` uses the shared failure predicate, so it agrees with Tool Success,
    the issues and the failure patterns; ``hint`` is a short description of
    the call (a command, a path) or an empty string.
    """
    color_map, labels, agent_id_of = bind_timeline_agents(steps) if steps else ({}, {}, None)
    calls = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = agent_id_of(s)
        idx = int(s.get("index") if s.get("index") is not None else 0)
        for tc in s.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            calls.append({
                "step": idx,
                "tool": tool_chart_name(tc),
                "ok": not tool_call_failed(tc),
                "hint": tool_call_hint(tc),
                "agent": agent,
            })
    return {"agents": _agent_list(color_map, labels), "calls": calls}


def agent_tokens(agent_summaries: list[dict]) -> dict:
    """Token totals per agent, with the disjoint segments that add up to them.

    ``input_unusable_steps`` counts steps whose input was not a usable count
    and so is missing from the agent's totals.
    """
    agents = []
    for a in agent_summaries:
        seg = token_segments(
            a["total_tokens"], a["input_tokens"], a["output_tokens"], a["reasoning_tokens"],
            a["cache_read_tokens"], a.get("cache_write_tokens", 0) or 0,
        )
        agents.append({
            "agent_id": a.get("agent_id", ""),
            "label": a["label"],
            "total": a["total_tokens"],
            "input": a["input_tokens"],
            "output": a["output_tokens"],
            "reasoning": a["reasoning_tokens"],
            "cache_read": a["cache_read_tokens"],
            "cache_write": a.get("cache_write_tokens", 0) or 0,
            "input_unusable_steps": a.get("input_tokens_unusable_steps", 0) or 0,
            "segments": {k: seg[k] for k in ("fresh", "cache_read", "cache_write", "output", "reasoning")},
        })
    return {"agents": agents}


def agent_swimlane(steps: list[dict]) -> dict:
    """Each lane's contiguous step ranges, with their tokens and tool calls.

    Lanes run bottom to top: the main agent, sub-agents in first-appearance
    order, then the human user (``id`` ``"__user__"``). ``color_index`` is the
    agent's palette position (``None`` for the user lane).
    """
    color_map, label_map, agent_id_of = bind_timeline_agents(steps)
    runs: dict[str, list[list[int]]] = defaultdict(list)
    prev: dict[str, int] = {}
    for s in steps:
        if not isinstance(s, dict):
            continue
        agent = USER_LANE_ID if workflow_role(s) == "user" else agent_id_of(s)
        idx = int(s.get("index") if s.get("index") is not None else 0)
        tok = int((s.get("tokens") or {}).get("total") or 0)
        tools = int(s.get("tool_call_count") or 0)
        if agent not in prev:
            runs[agent].append([idx, idx, tok, tools])
        else:
            last = runs[agent][-1]
            if idx == last[1] + 1:
                runs[agent][-1] = [last[0], idx, last[2] + tok, last[3] + tools]
            else:
                runs[agent].append([idx, idx, tok, tools])
        prev[agent] = idx
    lane_ids = [aid for aid in sorted(color_map, key=lambda a: color_map[a]) if aid in runs]
    if USER_LANE_ID in runs:
        lane_ids.append(USER_LANE_ID)
    return {
        "lanes": [
            {
                "id": aid,
                "label": USER_LANE_LABEL if aid == USER_LANE_ID else _legend_label(aid, label_map),
                "color_index": None if aid == USER_LANE_ID else color_map.get(aid, 0),
                "segments": [
                    {"start": a, "end": b, "steps": b - a + 1, "tokens": tok, "tool_calls": tools}
                    for a, b, tok, tools in runs[aid]
                ],
            }
            for aid in lane_ids
        ]
    }


def file_interactions(
    interactions: list[dict],
    target_files: set[str] | None = None,
    steps: list[dict] | None = None,
) -> dict:
    """Every file read, write, search or skill load, by step and agent.

    ``target`` marks files the run edited (the likely task targets); ``files``
    lists paths in first-touch order; ``agents`` is the palette order of the
    timeline agents, and an interaction's ``agent`` is ``""`` for the main
    agent or when the steps are not supplied.
    """
    norm_targets = {os.path.normpath(t) if t else t for t in (target_files or set())}
    color_map: dict[str, int] = {"": 0}
    labels: dict[str, str] = {"": "main"}
    agent_id_of = None
    step_by_idx: dict = {}
    if steps:
        color_map, labels, agent_id_of = bind_timeline_agents(steps)
        step_by_idx = {s["index"]: s for s in steps if isinstance(s, dict) and "index" in s}

    def _agent(item: dict) -> str:
        if agent_id_of is None:
            return ""
        step = step_by_idx.get(item["step"])
        return agent_id_of(step) if step else ""

    files: list[str] = []
    seen: set[str] = set()
    rows = []
    for item in interactions:
        path = item["path"]
        if path not in seen:
            seen.add(path)
            files.append(path)
        rows.append({
            "step": item["step"],
            "path": path,
            "tool": item["tool"],
            "type": item["type"],
            "agent": _agent(item),
            "target": (os.path.normpath(path) if path else path) in norm_targets,
        })
    return {"agents": _agent_list(color_map, labels), "files": files, "interactions": rows}
