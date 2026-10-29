"""Analysis sidebar layout and chat callbacks."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr

from ..assistant import analyze_loaded_trajectory, answer_question
from ..llm_config import AnalysisSettings, config_status_html, resolve_session_settings
from .shared import SharedState


@dataclass
class SidebarRefs:
    analysis_sidebar: gr.Sidebar
    analysis_status: gr.HTML
    analysis_chatbot: gr.Chatbot
    analysis_input: gr.Textbox
    analysis_clear: gr.Button


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
    return SidebarRefs(
        analysis_sidebar=analysis_sidebar,
        analysis_status=analysis_status,
        analysis_chatbot=analysis_chatbot,
        analysis_input=analysis_input,
        analysis_clear=analysis_clear,
    )


def _run_analysis(steps, brief, settings: AnalysisSettings | None = None):
    """First LLM pass using this session's settings and packed trajectory brief."""
    active = resolve_session_settings(settings)
    if not steps:
        return "", [], config_status_html(loaded_steps=0, config=active.config)
    packed = brief if isinstance(brief, str) else ""
    brief, history = analyze_loaded_trajectory(
        steps, brief=packed, config=active.config, language=active.language,
    )
    return brief, history, config_status_html(loaded_steps=len(steps), config=active.config)


def bind(refs: SidebarRefs, shared: SharedState, load_events) -> tuple:
    """Bind chat and return model event handles so settings can cancel stale work."""
    state_analysis_brief = shared.state_analysis_brief
    settings_inputs = [shared.state_settings] if shared.state_settings is not None else []
    model_events = []
    # Sidebar starts closed (open=False); expand/collapse keep this in sync so a
    # load while the panel is already open still kicks off analysis.
    state_sidebar_open = gr.State(False)

    def on_trajectory_loaded(steps, brief, sidebar_open, settings=None):
        """Pack status on load; run LLM only if the analysis panel is open."""
        active = resolve_session_settings(settings)
        if not steps:
            return "", [], config_status_html(loaded_steps=0, config=active.config)
        packed = brief if isinstance(brief, str) else ""
        status = config_status_html(loaded_steps=len(steps), config=active.config)
        if sidebar_open:
            return _run_analysis(steps, packed, active)
        return packed, [], status

    for _ev in load_events:
        model_events.append(_ev.then(
            fn=on_trajectory_loaded,
            inputs=[shared.state_steps, state_analysis_brief, state_sidebar_open, *settings_inputs],
            outputs=[state_analysis_brief, refs.analysis_chatbot, refs.analysis_status],
        ))

    def on_sidebar_expand(steps, brief, history, settings=None):
        active = resolve_session_settings(settings)
        open_state = True
        if history:
            # Already analyzed for this load (or user has chat); don't re-fire.
            n = len(steps) if steps else 0
            return (
                brief if isinstance(brief, str) else "",
                history,
                config_status_html(loaded_steps=n, config=active.config),
                open_state,
            )
        brief_out, history_out, status = _run_analysis(steps, brief, active)
        return brief_out, history_out, status, open_state

    def on_sidebar_collapse():
        return False

    model_events.append(refs.analysis_sidebar.expand(
        fn=on_sidebar_expand,
        inputs=[shared.state_steps, state_analysis_brief, refs.analysis_chatbot, *settings_inputs],
        outputs=[
            state_analysis_brief,
            refs.analysis_chatbot,
            refs.analysis_status,
            state_sidebar_open,
        ],
    ))
    refs.analysis_sidebar.collapse(
        fn=on_sidebar_collapse,
        outputs=[state_sidebar_open],
    )

    def on_analysis_ask(question, history, brief, settings=None):
        active = resolve_session_settings(settings)
        return answer_question(
            question, history, brief, config=active.config, language=active.language,
        ), ""

    model_events.append(refs.analysis_input.submit(
        fn=on_analysis_ask,
        inputs=[refs.analysis_input, refs.analysis_chatbot, state_analysis_brief, *settings_inputs],
        outputs=[refs.analysis_chatbot, refs.analysis_input],
    ))
    refs.analysis_clear.click(fn=lambda: [], outputs=[refs.analysis_chatbot])
    return tuple(model_events)
