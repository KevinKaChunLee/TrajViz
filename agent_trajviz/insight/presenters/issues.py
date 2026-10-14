"""Overview Issues panel: render the ranked issues as foldable HTML cards.

The issues themselves are computed by :mod:`agent_trajviz.insight.issues`
(English titles/details for the public API). Dashboard chrome here is Chinese,
including the title / detail / why text shown on each card.
"""

from __future__ import annotations

import html
import re

from ..issues import (  # noqa: F401  (re-exported: UI and judge import them here)
    Confidence,
    IssueJudgment,
    IssueKind,
    OverviewIssue,
    collect_overview_issues,
    rank_issues,
)
from ..rendering import _step_link_chips
from ..session import LoadedSession

ISSUE_KIND_COLORS: dict[IssueKind, str] = {
    "error": "var(--ov-bad)",
    "antipattern": "var(--ov-warn)",
    "bottleneck": "var(--ov-accent)",
}

_ISSUE_KIND_LABELS_ZH: dict[IssueKind, str] = {
    "error": "错误",
    "antipattern": "行为问题",
    "bottleneck": "耗时过长",
}

_CONFIDENCE_ZH: dict[Confidence, str] = {"high": "高", "medium": "中", "low": "低"}


def _with_count(label_zh: str, title_en: str) -> str:
    """Keep the English ``(N×)`` suffix from ``issues._count_label`` titles."""
    m = re.search(r" \((\d+)×\)$", title_en)
    return label_zh + (m.group(0) if m else "")


def _localize_fruitless_detail(detail: str) -> str:
    m = re.match(r"(\d+) wasted steps — (.+)$", detail)
    if not m:
        return detail
    rest = m.group(2).replace(", ", "，")
    rest = re.sub(r"steps (\d+)-(\d+) \((\d+)\)", r"第\1–\2步（\3）", rest)
    rest = re.sub(r"\+(\d+) more \((\d+)\)", r"另有 \1 段（\2）", rest)
    return f"{m.group(1)} 步无效搜索 — {rest}"


def _localize_bottleneck_title(title: str) -> str:
    m = re.match(r"Idle/queue bottleneck before #(\d+) \((\d+(?:\.\d+)?)s gap\)$", title)
    if m:
        return f"空闲/排队瓶颈，发生在 #{m.group(1)}（{m.group(2)}秒空隙）"
    m = re.match(r"Tool bottleneck: (.+) \((\d+(?:\.\d+)?)s\)$", title)
    if m:
        return f"工具瓶颈：{m.group(1)}（{m.group(2)}秒）"
    m = re.match(r"Context/cache bottleneck at #(\d+) \((\d+(?:\.\d+)?)s\)$", title)
    if m:
        return f"上下文/缓存瓶颈，发生在 #{m.group(1)}（{m.group(2)}秒）"
    m = re.match(r"Inference bottleneck at #(\d+) \((\d+(?:\.\d+)?)s\)$", title)
    if m:
        return f"推理瓶颈，发生在 #{m.group(1)}（{m.group(2)}秒）"
    m = re.match(r"Performance bottleneck at #(\d+)$", title)
    if m:
        return f"性能瓶颈，发生在 #{m.group(1)}"
    return title


def _localize_issue_text(issue: OverviewIssue) -> tuple[str, str, str]:
    """Chinese title / detail / why for the dashboard card; English API unchanged."""
    sid = issue.source_id or ""
    title, detail, why = issue.title, issue.detail, issue.why

    if sid.startswith("antipattern:compaction_premature"):
        m = re.match(
            r"window occupancy at compaction time: "
            r"(.+)% of the assumed (.+)-token window$",
            detail,
        )
        if m:
            detail = f"压缩时窗口占用：假定 {m.group(2)} token 窗口的 {m.group(1)}%"
        elif detail.endswith("well below the window limit"):
            detail = "压缩时窗口占用：远低于窗口上限"
        return (
            _with_count("过早压缩上下文", title),
            detail,
            "上下文窗口在远未写满时就被压缩——通常是智能体主动触发的 compress/compact，"
            "而非运行环境强制回收。过早压缩会丢失上下文；请检查摘要是否保留了关键信息。",
        )

    if sid.startswith("antipattern:compaction_grew"):
        detail_zh = re.sub(r"step (\d+):", r"第\1步：", detail)
        detail_zh = re.sub(r", \+(\d+) more$", r"，另有 \1 次", detail_zh)
        return (
            _with_count("压缩后上下文反而变大", title),
            detail_zh,
            "压缩结束后窗口占用不小于压缩前——摘要本身比腾出的消息更占 token，"
            "这次压缩等于白做。",
        )

    if sid.startswith("antipattern:fruitless"):
        return (
            _with_count("无效搜索连续空转", title),
            _localize_fruitless_detail(detail),
            "连续三次及以上搜索均无匹配；持续空转通常说明搜错了位置或关键词。",
        )

    if sid.startswith("antipattern:bash_read"):
        return (
            _with_count("用 Bash 读文件", title),
            "这些步骤用了 sed/cat/head，而不是 Read 工具",
            "用 shell 读文件会绕过结构化的 Read：没有行号、缓存更差、上下文更大。",
        )

    if sid.startswith("antipattern:stalled_plan"):
        return (
            _with_count("计划项卡住未完成", title),
            detail,
            "待办长期停在进行中却未完成——常见于切换任务后没有收尾。",
        )

    if sid.startswith("antipattern:plan_resets"):
        return (
            _with_count("计划被整表重置", title),
            "待办列表内容被整体替换，新旧项几乎没有重叠",
            "运行中途整表重写计划，通常表示智能体放弃了原上下文，而不是就地完成或修订。",
        )

    if sid.startswith("antipattern:edit_thrash:"):
        m = re.match(r"Failed edit retries on (.+) \((\d+)×\)$", title)
        if m:
            title = f"对 {m.group(1)} 的失败编辑重试（{m.group(2)}×）"
        detail = re.sub(
            r"(\d+) failed write\(s\) in steps (\d+)–(\d+)",
            r"第\2–\3步中有 \1 次写入失败",
            detail,
        )
        return (
            title,
            detail,
            "同一文件在写入失败后反复重写——常见于路径错误、补丁对不上，或缺少前置检查。",
        )

    if sid.startswith("antipattern:repeated_search:"):
        m = re.match(r"Repeated empty search \((\d+)×\)$", title)
        if m:
            title = f"重复的空结果搜索（{m.group(1)}×）"
        return (
            title,
            detail,
            "同一搜索多次返回空结果（不一定连续）——查询词或路径很可能不对。",
        )

    if sid.startswith("antipattern:tool_errors"):
        return (
            _with_count("工具错误", title),
            "根据工具输出检测到（平台错误、权限、文件缺失等）",
            "失败的工具调用既费 token 又费轮次，且往往是环境问题而非智能体策略错误。",
        )

    if sid.startswith("fail:system:"):
        if title.startswith("Frequent system errors: "):
            title = f"频繁系统错误：{title[len('Frequent system errors: '):]}"
            why = (
                "单次运行中出现大量 Read/Grep/Edit 类失败——单次风险通常不高，"
                "但数量说明路径、权限或运行环境配置值得排查。"
            )
        elif title.startswith("System error: "):
            title = f"系统错误：{title[len('System error: '):]}"
            why = (
                "脚手架/工具链未命中（搜索、读取或写入）——通常风险较低；"
                "智能体往往无需改作者代码即可恢复。"
            )
        if detail == "system tool failure":
            detail = "系统工具失败"
        return title, detail, why

    if sid.startswith("fail:"):
        return title, detail, _localize_recovery_why(why)

    if sid.startswith("cascade:"):
        return (
            f"连续失败（{len(issue.steps)} 步）",
            re.sub(r"steps (\d+)–(\d+)", r"第\1–\2步", detail),
            "连续多个助手步骤失败且中间没有干净恢复；级联往往会把一个根因放大成反复折腾。",
        )

    if sid.startswith("bottleneck:"):
        title = _localize_bottleneck_title(title)
        detail = re.sub(r"^Step (\d+):", r"第\1步：", detail)
        detail = detail.replace(" executing tools", " 在执行工具")
        if why.startswith("Large pre-step idle"):
            why = "步骤前有很长空闲，通常是排队或限流——应降低并行压力或调整等待策略，而不是改提示词。"
        elif why.startswith("Most of this outlier step was spent in a tool"):
            why = "该异常耗时步骤主要花在工具上——优先换更轻的工具、缩小范围，或缓存结果避免重复重活。"
        elif why.startswith("Long inference with heavy tokens"):
            why = "推理耗时长且 token 重/缓存命中差——应精简上下文、提高缓存局部性，或避免每轮重载大文件。"
        elif why.startswith("Outlier model-side turn"):
            why = "模型侧耗时异常且工具等待不多——常见于过长推理或提示；收紧指令或拆分任务。"
        elif why.startswith("Session-relative duration outlier"):
            why = "相对本会话的耗时异常点，且有明确的主导工具、空闲/排队或上下文/推理原因。"
        return title, detail, why

    # Synthetic cards in tests (and recovery why without a fail: source id).
    return title, detail, _localize_recovery_why(why)


def _localize_recovery_why(why: str) -> str:
    if why.startswith("Typical recovery: "):
        return "常见恢复路径：" + why[len("Typical recovery: ") :]
    if why == "No recovery path found after this error cluster.":
        return "该错误簇之后未找到恢复路径。"
    return why


def _issue_card(issue: OverviewIssue, number: int) -> str:
    """Compact scan row: number + kind badge + title + step chips + optional LLM fix."""
    title_raw, detail_raw, why_raw = _localize_issue_text(issue)
    title = html.escape(title_raw)
    detail = html.escape(detail_raw)
    border = ISSUE_KIND_COLORS[issue.kind]
    steps_html = _step_link_chips(list(issue.steps), label="第{n}步", more="另有 {extra} 步")

    judgment_html = ""
    if issue.judgment is not None:
        j = issue.judgment
        also_html = ""
        if j.also:
            also_html = (
                f"<div style='font-size:11px;color:var(--ov-muted);margin-top:6px;"
                f"line-height:1.35;'>补充：{html.escape(j.also)}</div>"
            )
        judgment_html = (
            "<div style='margin-top:8px;font-size:12px;line-height:1.4;'>"
            "<span style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "color:var(--ov-muted);'>修改位置</span>"
            f"<div style='font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            f"font-size:12px;color:var(--ov-text);margin-top:2px;'>"
            f"{html.escape(j.where)}</div></div>"
            "<div style='margin-top:8px;padding:8px 10px;background:var(--ov-bg);"
            "border-radius:4px;'>"
            "<div style='font-size:10px;font-weight:600;letter-spacing:0.04em;"
            "color:var(--ov-muted);margin-bottom:4px;'>"
            f"修复建议 <span style='font-weight:500;letter-spacing:0;"
            f"color:var(--ov-muted);'>（置信度：{_CONFIDENCE_ZH[j.confidence]}）</span></div>"
            f"<div style='font-size:13px;line-height:1.4;color:var(--ov-text);'>"
            f"{html.escape(j.fix)}</div>"
            f"{also_html}</div>"
        )

    detail_html = (
        f"<span class='overview-issue-detail'>{detail}</span>" if detail_raw else ""
    )
    info_html = ""
    if why_raw:
        info_html = (
            f"<span class='overview-issue-info' title='{html.escape(why_raw)}' "
            f"aria-hidden='true'>ⓘ</span>"
        )
    return (
        f"<div class='overview-issue-card' style='border-left-color:{border};'>"
        f"<div class='overview-issue-head'>"
        f"<span class='overview-issue-index'>#{number}</span>"
        f"<span class='overview-issue-kind' style='color:{border};'>"
        f"{_ISSUE_KIND_LABELS_ZH[issue.kind]}</span>"
        f"<span class='overview-issue-title'>{title}</span>"
        f"{info_html}"
        f"{detail_html}"
        f"</div>"
        f"{steps_html}"
        f"{judgment_html}"
        f"</div>"
    )


def _issue_cards_html(issues: list[OverviewIssue]) -> str:
    """Render all issue cards (no preview cap), numbered from 1 in ranked order."""
    return "".join(
        _issue_card(issue, number) for number, issue in enumerate(issues, start=1)
    )


def render_overview_issues_html(
    issues: list[OverviewIssue],
    *,
    banner: str = "",
    progress: str = "",
) -> str:
    """Render a foldable Issues panel (healthy empty state when *issues* is empty).

    *progress* is a live status line shown inside the panel while the LLM judge runs.
    """
    banner_html = ""
    if banner:
        banner_html = (
            f"<div style='font-size:12px;color:var(--ov-muted);margin:0 0 8px;'>"
            f"{html.escape(banner)}</div>"
        )

    progress_html = ""
    if progress:
        # SMIL (not CSS) so prefers-reduced-motion / Gradio HTML swaps don't freeze it.
        progress_html = (
            "<div class='overview-issues-progress' role='status' aria-live='polite'>"
            "<span class='overview-issues-progress-label'>思考中…</span>"
            "<svg class='overview-issues-progress-spinner' width='14' height='14' "
            "viewBox='0 0 24 24' aria-hidden='true' "
            "style='flex-shrink:0;display:block'>"
            "<circle cx='12' cy='12' r='10' fill='none' "
            "stroke='currentColor' stroke-opacity='0.25' stroke-width='3'/>"
            "<path d='M12 2a10 10 0 0 1 10 10' fill='none' stroke='currentColor' "
            "stroke-width='3' stroke-linecap='round'>"
            "<animateTransform attributeName='transform' type='rotate' "
            "from='0 12 12' to='360 12 12' dur='0.7s' repeatCount='indefinite'/>"
            "</path></svg>"
            f"<span>{html.escape(progress)}</span>"
            "</div>"
        )

    if not issues:
        return (
            "<details class='overview-issues-panel' id='overview-issues'>"
            "<summary class='overview-issues-summary'>"
            "<span>问题</span>"
            "<span class='overview-issues-summary-meta'>未检测到问题</span>"
            "</summary>"
            "<div class='overview-issues-body'>"
            f"{progress_html}"
            f"{banner_html}"
            "<div style='padding:8px 0 4px;color:var(--ov-muted);text-align:center;font-size:13px;'>"
            "未检测到明显的工作流问题。"
            "</div></div></details>"
        )

    count = len(issues)
    judged = sum(1 for i in issues if i.judgment is not None)
    count_label = f"{count} 个问题"
    if progress:
        count_label += " · 正在生成修复建议…"
    elif judged:
        count_label += f" · 已为 {judged} 个问题生成修复建议"

    lead = (
        "各问题下方为 LLM 给出的修改位置和修复建议"
        if judged and not progress
        else "按严重程度排序"
    )
    hint = (
        f"<div class='overview-issues-hint'>{lead} · 点击步骤可跳转到 Workflow"
        " · 鼠标悬停 ⓘ 查看说明</div>"
    )
    return (
        "<details class='overview-issues-panel' id='overview-issues' open>"
        "<summary class='overview-issues-summary'>"
        "<span>问题</span>"
        f"<span class='overview-issues-summary-meta'>{html.escape(count_label)}</span>"
        "</summary>"
        "<div class='overview-issues-body'>"
        f"{progress_html}"
        f"{banner_html}"
        f"{hint}"
        + _issue_cards_html(issues)
        + "</div></details>"
    )


def build_overview_issues_html(
    session: LoadedSession,
    *,
    issues: list[OverviewIssue] | None = None,
    banner: str = "",
    progress: str = "",
) -> str:
    """Collect, rank, and render Overview Issues for *session*."""
    if issues is None:
        ranked = rank_issues(collect_overview_issues(session))
    else:
        ranked = list(issues)
    return render_overview_issues_html(ranked, banner=banner, progress=progress)
