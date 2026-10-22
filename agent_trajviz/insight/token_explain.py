"""Per-step token breakdown explanations for the Token Usage chart.

Clicking a bar in the Summary section's Token Usage chart renders a
human-readable decomposition of that step's numbers: where the fresh input
came from (page-tail recomputation plus the previous step's tool results),
what the cache read covers, and what output/reasoning represent.

The decomposition mirrors the serving stack this project was validated
against (sglang radix cache + OpenAI-compatible usage). Numbers taken from
the usage report are marked 实际值 (exact); everything else is an
independent estimate marked ≈ — items are all listed, and their sum is
NOT forced to equal the real value. The estimation method itself is
rendered at the bottom of the panel (see ``_METHOD_DETAILS_HTML``).
"""

from __future__ import annotations

import html
from typing import Any

from .parser import infer_non_cache_input
from .shell_cmd import tool_call_hint
from .timeline_agents import timeline_agent_id_of

# sglang's radix cache matches prefixes in whole pages; 256 is the page size
# of the deployment these explanations were verified against.
RADIX_PAGE_TOKENS = 256

# Tool outputs are not billed separately by OpenAI-compatible APIs, so their
# token weight is estimated from text length. A single chars/token rate is
# off by ~2x between Chinese prose and code/ASCII, so the estimate mixes two
# rates by character class. Both were least-squares-fitted against a real
# Qwen tokenizer over a full session's tool outputs (aggregate error <5%);
# the project cannot ship per-model tokenizers, so this is as close as it
# gets without one.
CJK_CHARS_PER_TOKEN = 1.4
OTHER_CHARS_PER_TOKEN = 2.5

# Chat-template wrapper overhead, measured with a real tokenizer on the
# qwen3.5 template: consecutive tool results are merged into one user
# message (~5 tok header/footer + ~8 tok per response tag pair), a plain
# user message costs ~3 tok of role tags, and each tool call costs ~24 tok
# of function/parameter syntax (varies with the parameter count) plus ~3
# tok of assistant message header/footer.
TOOL_RESULT_WRAPPER_TOKENS = 8
TOOL_BATCH_WRAPPER_TOKENS = 5
USER_MSG_WRAPPER_TOKENS = 3
TOOL_CALL_SYNTAX_TOKENS = 24
ASSISTANT_MSG_WRAPPER_TOKENS = 3

_HINT_EN = "Click a bar to see that step's token breakdown."
_HINT_ZH = "点击上方 Token Usage 图表的柱子，查看该步 token 构成。"

# HTML collapses runs of plain spaces, so indentation must use &nbsp;.
# Level 1: top-level items (新增内容 / 页尾重算 / 首调构成项).
# Level 2: details nested under a level-1 item.
_IND1 = "&nbsp;&nbsp;"
_IND2 = "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"

_METHOD_DETAILS_HTML = (
    "<details class='tv-token-method' style='margin-top:6px;font-size:0.85em;"
    "color:var(--ov-muted);'>"
    "<summary style='cursor:pointer;user-select:none;'>估算方法（带 ≈ 的数值如何得出）</summary>"
    "<div style='padding:4px 0 2px 14px;line-height:1.7;'>"
    "导出文件只保存文本、不保存 token 数，因此带 ≈ 的项均为独立估算，"
    "逐项全部列出，但相加不必等于实际值：<br>"
    "1. 文本 → token：中文字符数 ÷ 1.4 + 其他字符数 ÷ 2.5 —— 参数用 Qwen tokenizer "
    "对真实轨迹的全部工具输出做最小二乘拟合，总体误差约 ±5%；<br>"
    f"2. 消息框架：每条工具结果 ≈ {TOOL_RESULT_WRAPPER_TOKENS} tok（模板标签对）"
    f"+ 每批工具结果 ≈ {TOOL_BATCH_WRAPPER_TOKENS} tok（消息头尾）；"
    f"每条用户消息 ≈ {USER_MSG_WRAPPER_TOKENS} tok（角色标记）；"
    f"每次工具调用语法 ≈ {TOOL_CALL_SYNTAX_TOKENS} tok（&lt;function&gt;/&lt;parameter&gt; "
    f"骨架，随参数个数变化）+ assistant 消息头尾 ≈ {ASSISTANT_MSG_WRAPPER_TOKENS} tok；<br>"
    f"3. 页大小 {RADIX_PAGE_TOKENS} token/页：sglang radix cache 的整页复用粒度，"
    "页尾重算 = 上一步总 token 对 256 取余；<br>"
    "标（实际值）的数字来自服务端 usage 上报，是精确的。"
    "</div></details>"
)


def build_token_step_panel_html(steps: list[dict]) -> str:
    """The full click-target panel: visible hint + one hidden div per step.

    The dashboard binds a client-side plotly_click handler (see
    ``tvBindTokenStepExplain`` in ``insight.py``) that reveals the
    ``[data-step]`` div matching the clicked bar, so no server round-trip
    is needed per click. The estimation-method note sits below the step
    divs so it stays visible whichever step is shown.
    """
    hint = (
        f"<div class='tv-token-hint'>{_esc(_HINT_EN)}<br>{_esc(_HINT_ZH)}</div>"
    )
    if not steps:
        return hint
    details = [
        f"<div class='tv-token-step-detail' data-step='{i}' style='display:none;'>"
        f"{build_token_step_explanation_html(steps, i)}"
        f"</div>"
        for i in range(len(steps))
    ]
    return hint + "".join(details) + _METHOD_DETAILS_HTML


def _est_tokens(text: Any) -> int:
    """Estimate the token count of *text* without the model's tokenizer."""
    if text is None:
        return 0
    s = text if isinstance(text, str) else str(text)
    if not s:
        return 0
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    other = len(s) - cjk
    return max(1, round(cjk / CJK_CHARS_PER_TOKEN + other / OTHER_CHARS_PER_TOKEN))


def _fmt(n: int) -> str:
    return f"{n:,}"


def _esc(s: Any) -> str:
    # quote=True: payloads must not survive even inside attribute context.
    return html.escape(str(s), quote=True)


def _tool_hint(tc: dict) -> str:
    """Short human hint for a tool call: name + first argument line."""
    name = tc.get("tool_name") or tc.get("name") or "?"
    hint = tool_call_hint(tc)
    if len(hint) > 48:
        hint = hint[:45] + "…"
    return f"{name}({hint})" if hint else str(name)


def _timeline_peer(steps: list[dict], step_index: int):
    """Same-conversation test for one step.

    With subagents the step list interleaves several conversations. A
    subagent step must not treat a root-agent step as its "previous step":
    they have different prefixes, so neither the radix-cache lineage nor the
    "new content since last call" narrative carries over. The conversation
    key is ``session_id`` — an agent-mode switch within one session (opencode
    plan/build) shares the context and must stay peers, so the timeline
    agent id (``session_id::agent``) would split it wrongly and is only the
    fallback when session ids are absent.
    """
    me = steps[step_index]
    me_sid = str(me.get("session_id") or "")
    if me_sid:

        def peer(step: dict) -> bool:
            return str(step.get("session_id") or "") == me_sid

        return peer

    agent_id = timeline_agent_id_of(steps)
    me_aid = agent_id(me)

    def peer(step: dict) -> bool:
        return agent_id(step) == me_aid

    return peer


def _prev_token_step(steps: list[dict], step_index: int) -> tuple[int, dict] | None:
    peer = _timeline_peer(steps, step_index)
    for i in range(step_index - 1, -1, -1):
        if not peer(steps[i]):
            continue
        if (steps[i].get("tokens") or {}).get("total", 0) > 0:
            return i, steps[i]
    return None


def _sum_note(est_total: int, actual: int, ind: str = _IND2) -> str:
    """Items are independent estimates; their sum need not match the actual."""
    if est_total == actual:
        return f"{ind}（估算合计 ≈ {_fmt(est_total)}，与实际一致）"
    return (
        f"{ind}（估算合计 ≈ {_fmt(est_total)}，实际 {_fmt(actual)}，"
        f"相差 {_fmt(abs(est_total - actual))} 属估算误差，各项独立、不必相等）"
    )


def _first_call_lines(steps: list[dict], step_index: int, fresh_input: int) -> list[str]:
    lines = [f"<b>{_fmt(fresh_input)}</b> — 首次 LLM 调用，全部为全新 prefill。可能构成："]
    est_total = 0
    peer = _timeline_peer(steps, step_index)
    for s in steps[:step_index]:
        if (s.get("role") or "") != "user" or not peer(s):
            continue
        est = _est_tokens(s.get("text_preview"))
        if est <= 0:
            continue
        est_total += est + USER_MSG_WRAPPER_TOKENS
        lines.append(
            f"{_IND1}- 用户消息 ≈ {_fmt(est)} tok + 包装 {_fmt(USER_MSG_WRAPPER_TOKENS)} tok（估算）"
        )
    rest = max(0, fresh_input - est_total)
    lines.append(
        f"{_IND1}- 系统提示 + 工具定义 ≈ {_fmt(rest)} tok：导出中不可见，由实际值减去可估算项推知"
        f"（opencode 的系统提示与工具定义通常占大头）"
    )
    return lines


def _fresh_input_lines(steps: list[dict], step_index: int, fresh_input: int) -> list[str]:
    prev_info = _prev_token_step(steps, step_index)
    if prev_info is None:
        return _first_call_lines(steps, step_index, fresh_input)

    prev_i, prev = prev_info
    prev_total = (prev.get("tokens") or {}).get("total", 0)
    tail = prev_total % RADIX_PAGE_TOKENS
    new_content = max(0, fresh_input - tail)
    lines = [
        f"<b>{_fmt(fresh_input)}</b> — 本步需重新计算（prefill）的输入，构成如下："
    ]

    # New content first: what actually entered the context for the first
    # time — the previous step's tool results, plus any user message sent
    # between the two calls. Every possible item is listed; each carries its
    # own estimate and the sum is not forced to match.
    lines.append(
        f"{_IND1}- 新增内容 {_fmt(new_content)}（实际值）：两步之间首次进入上下文的内容，逐项估算："
    )
    est_total = 0
    n_tools = 0
    for tc in prev.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        est = _est_tokens(tc.get("output"))
        est_total += est
        n_tools += 1
        lines.append(f"{_IND2}- {_esc(_tool_hint(tc))} 的执行结果 ≈ {_fmt(est)} tok（估算）")
    n_users = 0
    peer = _timeline_peer(steps, step_index)
    for s in steps[prev_i + 1 : step_index]:
        if (s.get("role") or "") != "user" or not peer(s):
            continue
        est = _est_tokens(s.get("text_preview"))
        est_total += est
        n_users += 1
        lines.append(f"{_IND2}- 用户新消息 ≈ {_fmt(est)} tok（估算）")
    framework = 0
    if n_tools:
        framework += TOOL_BATCH_WRAPPER_TOKENS + n_tools * TOOL_RESULT_WRAPPER_TOKENS
    if n_users:
        framework += n_users * USER_MSG_WRAPPER_TOKENS
    if framework:
        est_total += framework
        lines.append(
            f"{_IND2}- 消息框架 ≈ {_fmt(framework)} tok：角色标记与模板标签"
            f"（消息头尾 {_fmt(TOOL_BATCH_WRAPPER_TOKENS)} + 每条工具结果 {_fmt(TOOL_RESULT_WRAPPER_TOKENS)}"
            f" + 每条用户消息 {_fmt(USER_MSG_WRAPPER_TOKENS)}，估算）"
        )
    if n_tools or n_users:
        lines.append(_sum_note(est_total, new_content))
    else:
        lines.append(
            f"{_IND2}- 两步之间没有工具结果或新用户消息 — 新增内容可能来自消息框架、"
            f"系统层注入或上下文重组"
        )

    # Tail last: NOT new content. The radix cache reuses whole pages only, so
    # the previous sequence's last incomplete page is re-prefilled even though
    # the model already saw it last turn.
    if tail > 0:
        pages = prev_total // RADIX_PAGE_TOKENS
        lines.append(
            f"{_IND1}- 页尾重算 {_fmt(tail)}：非新内容 — 缓存按 {RADIX_PAGE_TOKENS} token/页 整页复用，"
            f"上一步 {_fmt(prev_total)} token 命中 {pages} 整页（{_fmt(pages * RADIX_PAGE_TOKENS)}），"
            f"末尾 {_fmt(tail)} 不满一页，只能重新计算"
        )
    return lines


def _cache_read_lines(steps: list[dict], step_index: int, cache_read: int) -> list[str]:
    if cache_read <= 0:
        return ["<b>0</b> — 无前缀缓存命中（冷启动 / 首次调用）"]
    prev_info = _prev_token_step(steps, step_index)
    if prev_info is None:
        return [f"<b>{_fmt(cache_read)}</b> — 前缀缓存命中"]
    prev_total = (prev_info[1].get("tokens") or {}).get("total", 0)
    pages = cache_read // RADIX_PAGE_TOKENS
    return [
        f"<b>{_fmt(cache_read)}</b> — 命中上一步结束时完整上下文 {_fmt(prev_total)} token 的整页部分"
        f"（{pages} 页 × {RADIX_PAGE_TOKENS}）：系统提示、历史消息与之前的工具结果基本全部命中，无需重新计算"
    ]


def _output_lines(step: dict, output: int) -> list[str]:
    if output <= 0:
        return ["<b>0</b> — 本步没有可见输出"]
    lines = [f"<b>{_fmt(output)}</b> — 模型可见输出（实际值，不含思考），逐项估算："]
    est_total = 0
    text = "".join(
        p.get("text") or ""
        for p in step.get("parts") or []
        if isinstance(p, dict) and p.get("type") == "text"
    )
    if text:
        est = _est_tokens(text)
        est_total += est
        lines.append(f"{_IND1}- 正文 {_fmt(len(text))} 字符 ≈ {_fmt(est)} tok（估算）")
    n_calls = 0
    for tc in step.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        est = _est_tokens(tc.get("input"))
        est_total += est
        n_calls += 1
        lines.append(f"{_IND1}- {_esc(_tool_hint(tc))} 参数 ≈ {_fmt(est)} tok（估算）")
    if n_calls:
        syntax = n_calls * TOOL_CALL_SYNTAX_TOKENS + ASSISTANT_MSG_WRAPPER_TOKENS
        est_total += syntax
        lines.append(
            f"{_IND1}- 调用语法标签 ≈ {_fmt(syntax)} tok：每次调用 {_fmt(TOOL_CALL_SYNTAX_TOKENS)}"
            f"（&lt;function&gt;/&lt;parameter&gt; 骨架，随参数个数变化）"
            f"+ 消息头尾 {_fmt(ASSISTANT_MSG_WRAPPER_TOKENS)}（估算）"
        )
    lines.append(_sum_note(est_total, output, ind=_IND1))
    return lines


def build_token_step_explanation_html(steps: list[dict], step_index: int) -> str:
    """Render the per-step token decomposition as HTML for the clicked bar."""
    if not steps or not (0 <= step_index < len(steps)):
        return "<div style='padding:0.6em;color:var(--ov-muted);'>未找到该步骤。</div>"

    step = steps[step_index]
    tokens = step.get("tokens") or {}
    total = tokens.get("total", 0)
    role = step.get("role") or "?"
    if total <= 0:
        return (
            f"<div style='padding:0.6em;color:var(--ov-muted);'>"
            f"第 {step_index} 步（{_esc(role)}）没有 token 记录 — 该类消息不产生 LLM 计费。</div>"
        )

    fresh_input = infer_non_cache_input(
        total_tokens=total,
        input_tokens=tokens.get("input", 0),
        output_tokens=tokens.get("output", 0),
        reasoning_tokens=tokens.get("reasoning", 0) or 0,
        cache_read_tokens=tokens.get("cache_read", 0),
    )
    cache_read = tokens.get("cache_read", 0)
    reasoning = tokens.get("reasoning", 0) or 0
    output = tokens.get("output", 0)
    prompt = fresh_input + cache_read
    completion = output + reasoning

    model = step.get("model_id") or ""
    header = f"第 {step_index} 步 · {_esc(role)}"
    if model:
        header += f" · {_esc(model)}"
    # Imported lazily: rendering pulls in pygments, and core analysis modules
    # import this file for the token estimator (they must stay dependency-free).
    from .rendering import _diag_jump_onclick

    jump = (
        f"<span class='insight-step-link' onclick=\"{_diag_jump_onclick(step_index)}\" "
        f"style='margin-left:10px;font-weight:400;'>在 Workflow 中查看该步 →</span>"
    )

    sections: list[tuple[str, list[str]]] = [
        ("Cache Read（缓存命中输入）", _cache_read_lines(steps, step_index, cache_read)),
        ("Fresh Input（新增输入）", _fresh_input_lines(steps, step_index, fresh_input)),
        (
            "Reasoning（思考）",
            [f"<b>{_fmt(reasoning)}</b> — 本步模型思考 token（think 块，含标签）"]
            if reasoning > 0
            else ["<b>0</b> — 本步无思考记录"],
        ),
        ("Output（可见输出）", _output_lines(step, output)),
    ]

    rows = []
    for title, lines in sections:
        body = "<br>".join(lines)
        rows.append(
            f"<tr><td style='padding:4px 10px 4px 0;white-space:nowrap;vertical-align:top;'>{title}</td>"
            f"<td style='padding:4px 0;'>{body}</td></tr>"
        )

    summary = (
        f"Total <b>{_fmt(total)}</b> = prompt {_fmt(prompt)}（{_fmt(fresh_input)} 新增 + {_fmt(cache_read)} 命中）"
        f" + completion {_fmt(completion)}（{_fmt(output)} 可见输出 + {_fmt(reasoning)} 思考）"
    )

    return (
        f"<div style='margin-top:0.4em;'>"
        f"<div style='font-weight:600;margin-bottom:4px;'>{header} — token 构成{jump}</div>"
        f"<table style='border-collapse:collapse;font-size:0.92em;'>{''.join(rows)}</table>"
        f"<div style='margin-top:6px;color:var(--body-text-color);'>{summary}</div>"
        f"</div>"
    )
