"""Overview Issues panel: render the ranked issues as foldable HTML cards.

The issues themselves are computed by :mod:`agent_trajviz.insight.issues`
(English titles/details for the public API). Dashboard chrome here is Chinese.
"""

from __future__ import annotations

import html

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


def _issue_card(issue: OverviewIssue, number: int) -> str:
    """Compact scan row: number + kind badge + title + step chips + optional LLM fix."""
    title = html.escape(issue.title)
    detail = html.escape(issue.detail)
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
        f"<span class='overview-issue-detail'>{detail}</span>" if issue.detail else ""
    )
    why_attr = info_html = ""
    if issue.why:
        why_attr = f" title='{html.escape(issue.why)}'"
        info_html = f"<span class='overview-issue-info'{why_attr} aria-hidden='true'>ⓘ</span>"
    return (
        f"<div class='overview-issue-card' style='border-left-color:{border};'>"
        f"<div class='overview-issue-head'>"
        f"<span class='overview-issue-index'>#{number}</span>"
        f"<span class='overview-issue-kind' style='color:{border};'>"
        f"{_ISSUE_KIND_LABELS_ZH[issue.kind]}</span>"
        f"<span class='overview-issue-title'{why_attr}>{title}</span>"
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
            "<span>Issues</span>"
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
        "<span>Issues</span>"
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
