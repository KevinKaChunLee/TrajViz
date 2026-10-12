"""Views tab: the views added with ``agent_trajviz.register_view``.

Built only when an application or an installed plugin registered views (the
default dashboard has no such tab). Each view is drawn from the loaded
trajectory's public view, the same way any application would draw it: its
registered figure when it has one, its JSON data otherwise.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import gradio as gr

from ..session import MAX_STEPS
from .shared import SharedState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViewsRefs:
    #: View id -> the component showing it, in catalog order.
    components: dict[str, Any]


def registered_view_ids() -> list[str]:
    """Ids of the views added with ``register_view``, in catalog order."""
    from ...views import _BUILT_IN_VIEWS, VIEW_CATALOG

    return [vid for vid in VIEW_CATALOG if vid not in _BUILT_IN_VIEWS]


def _has_figure(view_id: str) -> bool:
    from ...figures import FIGURE_VIEWS

    return view_id in FIGURE_VIEWS


def layout() -> ViewsRefs | None:
    """The tab, or ``None`` (and no tab) when no view is registered."""
    from ...views import VIEW_CATALOG

    ids = registered_view_ids()
    if not ids:
        return None
    components: dict[str, Any] = {}
    with gr.Tab("Views"):
        gr.Markdown("Views added by your application or an installed plugin, for the loaded trajectory.")
        for vid in ids:
            info = VIEW_CATALOG[vid]
            with gr.Accordion(info["title"], open=True):
                if info["description"]:
                    gr.Markdown(info["description"])
                components[vid] = gr.Plot(show_label=False) if _has_figure(vid) else gr.JSON(show_label=False)
    return ViewsRefs(components)


def fill_views(run: Any, dark: bool = False, view_ids: list[str] | None = None) -> list[Any]:
    """One value per registered view for *run* (``None`` before a load).

    The dashboard draws at most ``MAX_STEPS`` steps, and so do these views. A
    view whose code raises shows the error type instead of breaking the tab.
    """
    from ...figures import figure

    ids = registered_view_ids() if view_ids is None else view_ids
    if run is None or not run.steps:
        return [None] * len(ids)
    shown = dataclasses.replace(run, steps=run.steps[:MAX_STEPS]) if len(run.steps) > MAX_STEPS else run
    values: list[Any] = []
    for vid in ids:
        try:
            view = shown.view(vid)
            values.append(figure(view, dark=bool(dark)) if _has_figure(vid) else view.data)
        except Exception as exc:  # noqa: BLE001 -- one plugin's bug must not break the tab
            logger.exception("view %r failed", vid)
            values.append(None if _has_figure(vid) else {"error": f"{type(exc).__name__} (see the server log)"})
    return values


def bind(refs: ViewsRefs | None, shared: SharedState, load_events: tuple) -> None:
    """Fill the tab after every load."""
    if refs is None:
        return
    ids = list(refs.components)

    def on_load(run, dark=False):
        values = fill_views(run, dark, ids)
        # Gradio takes a lone output's return value as is: a list would be its value.
        return values[0] if len(values) == 1 else values

    for event in load_events:
        event.then(fn=on_load, inputs=[shared.state_run, shared.state_dark], outputs=list(refs.components.values()))
