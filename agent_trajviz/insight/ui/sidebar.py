"""Analysis sidebar layout and chat callbacks."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..assistant import analyze_loaded_trajectory, answer_question
from ..llm_config import config_status_html
from ..presenters.issues import issue_analysis_question_from_payload
from .shared import SharedState


@dataclass
class SidebarRefs:
    analysis_sidebar: gr.Sidebar
    analysis_status: gr.HTML
    analysis_chatbot: gr.Chatbot
    analysis_input: gr.Textbox
    analysis_clear: gr.Button
    issue_ask_box: gr.Textbox
    issue_ask_btn: gr.Button


def _brief_text(brief: object) -> str:
    return brief if isinstance(brief, str) else ""


def layout() -> SidebarRefs:
    with gr.Sidebar(
        label="AI Trajectory Analysis",
        position="right",
        width="max(480px, 25vw)",
        open=False,
        elem_id="analysis-sidebar",
        elem_classes=["analysis-sidebar"],
    ) as analysis_sidebar:
        gr.HTML(
            "<div class='analysis-panel-title'>🤖 AI Trajectory Analysis</div>"
        )
        analysis_status = gr.HTML(config_status_html(loaded_steps=0))
        analysis_chatbot = gr.Chatbot(
            value=[],
            label="Analysis",
            show_label=False,
            height=380,
            resizable=True,
            layout="panel",
            placeholder=(
                "Open this panel after loading a trajectory to run analysis, "
                "then ask follow-up questions."
            ),
            buttons=["copy"],
            feedback_options=None,
            elem_classes=["analysis-chatbot"],
        )
        with gr.Row():
            analysis_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What is the biggest performance bottleneck?",
                scale=4,
                container=False,
                submit_btn=True,
                elem_id="analysis-input",
            )
        analysis_clear = gr.Button("Clear chat", size="sm", variant="secondary")
        # Gradio omits visible=False from the DOM; keep mounted and hide with CSS
        # (same bridge pattern as #wf-filter-hidden). Filled by Overview
        # "AI 深入分析"; the hidden button is the sole consumer.
        issue_ask_box = gr.Textbox(
            value="",
            visible=True,
            elem_id="tv-issue-ask",
            show_label=False,
            container=False,
        )
        issue_ask_btn = gr.Button(
            "Analyze issue",
            visible=True,
            elem_id="tv-issue-ask-btn",
        )
    return SidebarRefs(
        analysis_sidebar=analysis_sidebar,
        analysis_status=analysis_status,
        analysis_chatbot=analysis_chatbot,
        analysis_input=analysis_input,
        analysis_clear=analysis_clear,
        issue_ask_box=issue_ask_box,
        issue_ask_btn=issue_ask_btn,
    )


def _run_analysis(steps, brief):
    """First LLM pass using the brief packed from LoadedSession at load."""
    if not steps:
        return "", [], config_status_html(loaded_steps=0)
    packed = _brief_text(brief)
    brief, history = analyze_loaded_trajectory(steps, brief=packed)
    return brief, history, config_status_html(loaded_steps=len(steps))


def bind(refs: SidebarRefs, shared: SharedState, load_events) -> None:
    state_analysis_brief = shared.state_analysis_brief
    # Sidebar starts closed (open=False); expand/collapse keep this in sync so a
    # load while the panel is already open still kicks off analysis.
    state_sidebar_open = gr.State(False)

    def on_trajectory_loaded(steps, brief, sidebar_open):
        """Pack status on load; run LLM only if the analysis panel is open."""
        if not steps:
            return "", [], config_status_html(loaded_steps=0)
        packed = _brief_text(brief)
        status = config_status_html(loaded_steps=len(steps))
        if sidebar_open:
            return _run_analysis(steps, packed)
        return packed, [], status

    for _ev in load_events:
        _ev.then(
            fn=on_trajectory_loaded,
            inputs=[shared.state_steps, state_analysis_brief, state_sidebar_open],
            outputs=[state_analysis_brief, refs.analysis_chatbot, refs.analysis_status],
        )

    def on_sidebar_expand(steps, brief, history, pending_q):
        open_state = True
        packed = _brief_text(brief)
        n = len(steps) if steps else 0
        # Pending issue ask is handled by issue_ask_btn; suppress the automatic
        # whole-run first pass so the focused question is the first reply.
        if (pending_q or "").strip():
            return packed, history, config_status_html(loaded_steps=n), open_state
        if history:
            return packed, history, config_status_html(loaded_steps=n), open_state
        brief_out, history_out, status = _run_analysis(steps, packed)
        return brief_out, history_out, status, open_state

    def on_sidebar_collapse():
        return False

    refs.analysis_sidebar.expand(
        fn=on_sidebar_expand,
        inputs=[
            shared.state_steps,
            state_analysis_brief,
            refs.analysis_chatbot,
            refs.issue_ask_box,
        ],
        outputs=[
            state_analysis_brief,
            refs.analysis_chatbot,
            refs.analysis_status,
            state_sidebar_open,
        ],
    )
    refs.analysis_sidebar.collapse(
        fn=on_sidebar_collapse,
        outputs=[state_sidebar_open],
    )

    def on_analysis_ask(question, history, brief):
        return answer_question(question, history, brief), ""

    refs.analysis_input.submit(
        fn=on_analysis_ask,
        inputs=[refs.analysis_input, refs.analysis_chatbot, state_analysis_brief],
        outputs=[refs.analysis_chatbot, refs.analysis_input],
    )
    refs.analysis_clear.click(fn=lambda: [], outputs=[refs.analysis_chatbot])

    def on_issue_ask(pending_q, history, brief, steps, issue_rows):
        packed = _brief_text(brief)
        hist = list(history or [])
        status = config_status_html(loaded_steps=len(steps) if steps else 0)
        raw = (pending_q or "").strip()
        try:
            number = int(raw)
        except ValueError:
            number = 0
        rows = issue_rows if isinstance(issue_rows, list) else []
        question = issue_analysis_question_from_payload(rows, number)
        if question:
            hist = answer_question(question, hist, packed)
        return hist, packed, status, True, ""

    refs.issue_ask_btn.click(
        fn=on_issue_ask,
        inputs=[
            refs.issue_ask_box,
            refs.analysis_chatbot,
            state_analysis_brief,
            shared.state_steps,
            shared.state_overview_issues,
        ],
        outputs=[
            refs.analysis_chatbot,
            state_analysis_brief,
            refs.analysis_status,
            state_sidebar_open,
            refs.issue_ask_box,
        ],
    )
