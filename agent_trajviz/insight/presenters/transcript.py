"""Chat-style full-conversation transcript presenter.

The Workflow tab is a card flow: every step shows a one-line preview, and the
full text / tool payloads live behind per-step detail panels whose tool output
is capped at 2,000 characters. This presenter renders the same parsed steps as
one scrollable chat transcript — full message text inline, reasoning and tool
calls as collapsed ``<details>`` blocks whose Input/Output are never truncated.

Every dynamic string goes through ``_esc`` (``html.escape(quote=True)``): a
trajectory is an untrusted document, and this HTML lands in the dashboard and
the exported report (see tests/test_render_injection.py).
"""

from __future__ import annotations

import html
import json
import math

from ..charts import bind_timeline_agents
from ..metrics import tool_call_duration_ms
from ..palette import AGENT_COLORS, AGENT_CSS_COLORS
from ..parser import usable_token_count

_ROLE_BADGE_STYLES = {
    "user": "background:var(--wf-border-user);color:white;",
    "assistant": "background:var(--wf-border-assistant);color:white;",
}
_DEFAULT_ROLE_BADGE_STYLE = "background:var(--wf-border-default);color:white;"

# Title-arg keys mirrored from the Workflow detail panel so a tool call with no
# explicit title shows the same hint in both views.
_TITLE_ARG_KEYS = ("command", "file_path", "pattern", "description", "prompt")

_EMPTY_MESSAGE_PLACEHOLDER = "(no content)"

TRANSCRIPT_CSS = """<style>
.tv-transcript {
    font-family: system-ui, -apple-system, 'Noto Sans SC', 'Noto Sans CJK SC',
        'PingFang SC', 'Microsoft YaHei', 'Hiragino Sans GB', sans-serif;
    max-width: 860px; margin: 0 auto;
}
.tv-stats {
    display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px;
    color: var(--ov-muted); padding: 2px 2px 10px;
}
.tv-stats strong { color: var(--ov-text); }
.tv-msg {
    border: 1px solid var(--ov-border); border-radius: 10px;
    background: var(--ov-card); padding: 10px 14px; margin-bottom: 10px;
}
.tv-msg-header { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; }
/* Shape is re-declared scoped so badges keep their look even before the
   Workflow tab's own CSS (WORKFLOW_CSS) has been rendered into the page. */
.tv-transcript .wf-badge {
    display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px;
    border-radius: 6px; margin-right: 6px; text-transform: uppercase;
    letter-spacing: 0.5px;
}
.tv-time { font-size: 11px; color: var(--ov-muted); margin-left: auto; white-space: nowrap; }
.tv-meta {
    font-size: 12px; color: var(--ov-muted); margin-top: 4px;
    display: flex; gap: 8px; flex-wrap: wrap;
}
.tv-text {
    white-space: pre-wrap; word-break: break-word; font-size: 13px;
    line-height: 1.55; margin-top: 8px; color: var(--ov-text);
}
.tv-details { margin-top: 8px; }
.tv-details-body pre { white-space: pre-wrap; word-break: break-word; }
.tv-io-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--ov-muted); margin: 8px 0 2px;
}
.tv-empty { color: var(--ov-muted); font-style: italic; font-size: 12px; margin-top: 6px; }
</style>"""


def _esc(value: object) -> str:
    """Escape one dynamic string for any HTML context (content or attribute)."""
    return html.escape(str(value), quote=True)


def _finite_ms(value: object) -> float | None:
    """Epoch-milliseconds as a finite float, or None when unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _earliest_created_ms(steps: list[dict]) -> float | None:
    values = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        ms = _finite_ms(step.get("time_created_ms"))
        if ms is not None:
            values.append(ms)
    return min(values) if values else None


def _relative_time_label(step: dict, base_ms: float | None) -> str:
    """Seconds since the earliest message, e.g. ``+12.3s``."""
    ms = _finite_ms(step.get("time_created_ms"))
    if base_ms is None or ms is None:
        return "\u2014"
    return f"+{(ms - base_ms) / 1000.0:.1f}s"


def _step_total_tokens(step: dict) -> int | None:
    tokens = step.get("tokens")
    if not isinstance(tokens, dict):
        return None
    return usable_token_count(tokens.get("total"))


def _stats_html(steps: list[dict]) -> str:
    """One summary line: message count, role split, sub-agent count, tokens."""
    total = sum(1 for s in steps if isinstance(s, dict))
    user = sum(1 for s in steps if isinstance(s, dict) and s.get("role") == "user")
    assistant = sum(1 for s in steps if isinstance(s, dict) and s.get("role") == "assistant")
    sub_agent = sum(1 for s in steps if isinstance(s, dict) and s.get("is_sub_agent"))
    tokens = sum(t for s in steps if isinstance(s, dict)
                 if (t := _step_total_tokens(s)) is not None)
    return (
        "<div class='tv-stats'>"
        f"<span><strong>{total}</strong> messages</span>"
        f"<span><strong>{user}</strong> user</span>"
        f"<span><strong>{assistant}</strong> assistant</span>"
        f"<span><strong>{sub_agent}</strong> sub-agent</span>"
        f"<span><strong>{tokens:,}</strong> tokens</span>"
        "</div>"
    )


def _text_part_html(part: dict) -> str:
    text = part.get("text") or ""
    if not text.strip():
        return ""
    # Full text, not the Workflow card's one-line preview. Markdown is not
    # rendered: plain escaped text with preserved line breaks. The label
    # matches the Input/Output markers so a bare paragraph reads as the
    # message text instead of unlabeled noise between reasoning and tools.
    return (
        "<div class='tv-io-label'>Text</div>"
        f"<div class='tv-text'>{_esc(text)}</div>"
    )


def reasoning_block_label(
    text: str,
    tokens: int | None,
    part_no: int = 1,
    part_count: int = 1,
) -> str:
    """Summary label for a reasoning block: real token metric when available.

    ``tokens`` is the step-level ``tokens.reasoning`` total. When a step has
    several reasoning parts the total cannot be split per part, so every
    label carries the total plus the part ordinal. Exports without token
    metrics fall back to a character count.
    """
    if tokens:
        if part_count > 1:
            return f"Reasoning ({tokens:,} tok · part {part_no}/{part_count})"
        return f"Reasoning ({tokens:,} tok)"
    return f"Reasoning ({len(text):,} chars)"


def _reasoning_part_html(
    part: dict,
    tokens: int | None = None,
    part_no: int = 1,
    part_count: int = 1,
) -> str:
    text = part.get("text") or ""
    if not text:
        return ""
    return (
        "<details class='dp-details tv-details'>"
        f"<summary>{reasoning_block_label(text, tokens, part_no, part_count)}</summary>"
        f"<div class='dp-details-body'><pre>{_esc(text)}</pre></div>"
        "</details>"
    )


def _tool_input_text(inp: object) -> str:
    if isinstance(inp, dict):
        try:
            return json.dumps(inp, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(inp)
    return str(inp)


def _tool_output_text(output: object) -> str:
    if isinstance(output, str):
        return output
    if (
        isinstance(output, list)
        and output
        and all(isinstance(b, dict) and b.get("type") == "text" for b in output)
    ):
        # unwrap content-block lists to readable text (same as the detail panel)
        return "\n".join(str(b.get("text", "")) for b in output)
    try:
        return json.dumps(output, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(output)


def _tool_error_text(error: object) -> str:
    if isinstance(error, str):
        return error
    try:
        return json.dumps(error, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(error)


def _tool_call_html(part: dict) -> str:
    """One collapsed tool call; Input/Output in full — no truncation.

    This is the differentiator vs the Workflow detail panel, which caps tool
    output at 2,000 characters.
    """
    tool_name = str(part.get("tool_name") or "?")
    title = str(part.get("title") or "")
    if not title:
        inp = part.get("input")
        if isinstance(inp, dict):
            for key in _TITLE_ARG_KEYS:
                value = inp.get(key, "")
                if value:
                    title = str(value)[:80]
                    break

    dur_ms = tool_call_duration_ms(part)
    dur_label = f" &mdash; {dur_ms / 1000.0:.2f}s" if dur_ms is not None else ""
    status = str(part.get("status") or "?")
    is_error = bool(part.get("error")) or status == "error"
    title_part = f" &middot; {_esc(title)}" if title else ""
    err_mark = " &middot; error" if is_error else ""

    body: list[str] = [
        f"<div class='tv-meta'><span>status: <code>{_esc(status)}</code></span></div>",
        f"<div class='tv-io-label'>Input</div>"
        f"<pre>{_esc(_tool_input_text(part.get('input', {})))}</pre>",
    ]
    output_text = _tool_output_text(part.get("output", ""))
    if output_text:
        body.append(
            f"<div class='tv-io-label'>Output</div><pre>{_esc(output_text)}</pre>"
        )
    error = part.get("error")
    if error:
        body.append(
            f"<div class='tv-io-label'>Error</div><pre>{_esc(_tool_error_text(error))}</pre>"
        )

    return (
        "<details class='dp-details tv-details'>"
        f"<summary><code>{_esc(tool_name)}</code>{title_part}{dur_label}{err_mark}</summary>"
        f"<div class='dp-details-body'>{''.join(body)}</div>"
        "</details>"
    )


def _patch_part_html(part: dict) -> str:
    diff = part.get("diff_content") or ""
    if not diff:
        return ""
    patch_hash = str(part.get("hash") or "")
    label = f"Patch {_esc(patch_hash[:12])}" if patch_hash else "Patch"
    return (
        "<details class='dp-details tv-details'>"
        f"<summary>{label}</summary>"
        f"<div class='dp-details-body'><pre>{_esc(diff)}</pre></div>"
        "</details>"
    )


def _compaction_part_html(part: dict) -> str:
    summary_text = part.get("summary") or ""
    if not summary_text:
        return ""
    return (
        "<details class='dp-details tv-details'>"
        "<summary>Compaction summary</summary>"
        f"<div class='dp-details-body'><pre>{_esc(summary_text)}</pre></div>"
        "</details>"
    )


def _parts_html(step: dict) -> str:
    """Render the conversation content of one step's parts in order."""
    parts = [p for p in (step.get("parts") or []) if isinstance(p, dict)]
    # Real token metric for the reasoning label; None falls back to chars.
    reasoning_total = (step.get("tokens") or {}).get("reasoning") or None
    reasoning_count = sum(
        1 for p in parts if p.get("type") == "reasoning" and p.get("text")
    )
    reasoning_no = 0
    chunks: list[str] = []
    for part in parts:
        ptype = part.get("type")
        if ptype == "text":
            chunk = _text_part_html(part)
        elif ptype == "reasoning":
            if part.get("text"):
                reasoning_no += 1
            chunk = _reasoning_part_html(
                part,
                tokens=reasoning_total,
                part_no=reasoning_no,
                part_count=reasoning_count or 1,
            )
        elif ptype == "tool_call":
            chunk = _tool_call_html(part)
        elif ptype == "patch":
            chunk = _patch_part_html(part)
        elif ptype == "compaction":
            chunk = _compaction_part_html(part)
        else:
            # step_start / step_finish / snapshot / unknown carry no
            # conversation content.
            chunk = ""
        if chunk:
            chunks.append(chunk)
    if not chunks:
        return f"<div class='tv-empty'>{_EMPTY_MESSAGE_PLACEHOLDER}</div>"
    return "".join(chunks)


def _message_html(
    step: dict,
    *,
    fallback_index: int,
    base_ms: float | None,
    color_map: dict[str, int],
    labels: dict[str, str],
    agent_id_of,
) -> str:
    raw_idx = step.get("index", fallback_index)
    idx = raw_idx if isinstance(raw_idx, int) and not isinstance(raw_idx, bool) else fallback_index

    role = str(step.get("role") or "?")
    role_style = _ROLE_BADGE_STYLES.get(role, _DEFAULT_ROLE_BADGE_STYLE)
    role_label = role.capitalize() if role else "?"

    # Agent identity and colors match the Workflow cards / swimlanes exactly.
    aid = agent_id_of(step)
    aidx = color_map.get(aid, 0)
    agent_hex = AGENT_COLORS[aidx % len(AGENT_COLORS)]
    agent_bg, agent_border = AGENT_CSS_COLORS[aidx % len(AGENT_CSS_COLORS)]
    agent_label = labels.get(aid) or (aid or "main")
    agent_badge = ""
    if len(color_map) > 1 or str(step.get("agent") or "").strip():
        agent_badge = (
            f'<span class="wf-badge" style="background:{agent_bg};color:{agent_border};'
            f'border:1px solid {agent_border};font-size:9px;">{_esc(agent_label)}</span>'
        )

    sub_agent_badge = ""
    sub_indent = ""
    if step.get("is_sub_agent"):
        sub_agent_badge = (
            '<span class="wf-badge" style="background:var(--wf-border-reasoning);'
            'color:white;font-size:10px;">sub-agent</span>'
        )
        sub_indent = "margin-left:24px;"

    meta_bits: list[str] = []
    duration = step.get("duration")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        meta_bits.append(f"<span>{duration}s</span>")
    total_tokens = _step_total_tokens(step)
    if total_tokens:
        meta_bits.append(f"<span>{total_tokens:,} tok</span>")
    meta = f"<div class='tv-meta'>{''.join(meta_bits)}</div>" if meta_bits else ""

    header = (
        "<div class='tv-msg-header'>"
        f'<span class="wf-badge" style="background:var(--ov-accent);color:white;">#{idx}</span>'
        f'<span class="wf-badge" style="{role_style}">{_esc(role_label)}</span>'
        f"{sub_agent_badge}{agent_badge}"
        f"<span class='tv-time'>{_esc(_relative_time_label(step, base_ms))}</span>"
        "</div>"
    )

    return (
        f"<div class='tv-msg' id='tv-msg-{idx}' "
        f"style='border-left:4px solid {agent_hex};{sub_indent}'>"
        f"{header}{meta}{_parts_html(step)}</div>"
    )


def build_step_parts_html(step: dict) -> str:
    """Public wrapper: the conversation content of one step, Workflow-embeddable.

    The Workflow expanded view embeds this below each card header; the
    standalone transcript tab used to render it inside its own layout.
    """
    return _parts_html(step)


def build_transcript_html(steps: list[dict]) -> str:
    """Render every parsed step as one chat-style, full-text transcript."""
    if not steps:
        return (
            "<div style='padding:2em;color:var(--ov-muted);text-align:center;'>"
            "No messages to display.</div>"
        )
    color_map, labels, agent_id_of = bind_timeline_agents(steps)
    base_ms = _earliest_created_ms(steps)
    blocks = [
        _message_html(
            step,
            fallback_index=i,
            base_ms=base_ms,
            color_map=color_map,
            labels=labels,
            agent_id_of=agent_id_of,
        )
        for i, step in enumerate(steps)
        if isinstance(step, dict)
    ]
    return (
        TRANSCRIPT_CSS
        + "<div class='tv-transcript'>"
        + _stats_html(steps)
        + "".join(blocks)
        + "</div>"
    )
