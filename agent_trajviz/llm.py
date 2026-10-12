"""LLM-assisted analysis of a loaded run: ask questions, get suggested fixes.

Needs the ``llm`` extra (``pip install "agent-trajviz[llm]"``).

    >>> from agent_trajviz import llm
    >>> history = llm.ask(run, llm.first_question())       # the dashboard's first pass
    >>> history = llm.ask(run, "Why is step 12 slow?", history)
    >>> issues, errors = llm.suggest_fixes(run)             # issues with a "judgment"

Nothing leaves the machine until you call :func:`ask` or
:func:`suggest_fixes` with a configured endpoint (see :func:`config_from_env`),
and every request is redacted of credential-shaped text first. The model sees
a statistics brief (:func:`brief`) -- metrics, verdicts, bottlenecks,
failures, short previews -- not the raw trajectory.

Any client works: pass ``chat=`` a function ``(system, messages) -> reply``
(:data:`ChatFunction`) to use your own SDK, gateway or local model. It
receives the redacted prompt, so the guarantee above still holds, and no
endpoint configuration is needed. ``language=`` sets the reply language
(default: ``ANALYZE_LANGUAGE``, else English).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._extras import require

require("llm", "agent_trajviz.llm")

from .insight.assistant import (  # noqa: E402
    answer_question,
    auto_analysis_question,
    build_analysis_brief_from_session,
)
from .insight.issue_judge import JUDGE_ISSUE_CAP, judge_overview_issues  # noqa: E402
from .insight.llm_config import AnalysisLLMConfig, load_env_files, resolve_analysis_config  # noqa: E402
from ._redact import redact  # noqa: E402
from .views import _jsonable  # noqa: E402

if TYPE_CHECKING:
    from .api import Trajectory

__all__ = ["ChatFunction", "LLMConfig", "ask", "brief", "config_from_env", "first_question", "suggest_fixes"]

#: Endpoint, key, model and sampling settings for the LLM features.
LLMConfig = AnalysisLLMConfig

#: A client: ``(system prompt, messages) -> reply text``. Messages are
#: ``{"role": "user" | "assistant", "content": str}``.
ChatFunction = Callable[[str, list[dict[str, str]]], str]


def _redacted(chat: ChatFunction) -> Callable[[LLMConfig, str, list[dict[str, str]]], str]:
    """Adapt *chat* to the internal call shape, redacting everything it is sent."""

    def call(_config: LLMConfig, system: str, messages: list[dict[str, str]]) -> str:
        clean_system, _ = redact(system)
        clean = [{**m, "content": redact(m["content"])[0]} for m in messages]
        return chat(clean_system, clean)

    return call


def config_from_env(*, load_dotenv: bool = False) -> LLMConfig:
    """Settings from ``ANALYZE_*`` environment variables, with ``LABEL_*`` filling gaps.

    ``load_dotenv=True`` first reads ``.env`` in the working directory (only
    ``ANALYZE_*``/``LABEL_*`` keys, and never over a variable already set).
    ``config.ready`` says whether a request can be made; ``config.missing``
    names what is not set.
    """
    if load_dotenv:
        load_env_files()
    return resolve_analysis_config()


def brief(run: Trajectory) -> str:
    """The statistics brief the model reasons over, as plain text."""
    return build_analysis_brief_from_session(run._analysis)


def first_question(language: str | None = None) -> str:
    """The dashboard's first-pass question: bottlenecks, error patterns, root cause, fixes.

    *language* is the reply language (default: ``ANALYZE_LANGUAGE``).
    """
    return auto_analysis_question(language)


def ask(
    run: Trajectory,
    question: str,
    history: list[dict[str, Any]] | None = None,
    *,
    config: LLMConfig | None = None,
    chat: ChatFunction | None = None,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Ask a question about *run*; return *history* plus the question and the reply.

    Messages are ``{"role": "user" | "assistant", "content": str}``. A
    missing configuration or a failed request becomes an assistant message
    saying so, never an exception, so a chat UI can show it as it is. *chat*
    replaces the built-in HTTP client; *language* sets the reply language.
    """
    chat_fn = _redacted(chat) if chat is not None else None
    return answer_question(question, history, brief(run), config=config, chat_fn=chat_fn, language=language,
                           require_config=chat is None)


def suggest_fixes(
    run: Trajectory,
    *,
    config: LLMConfig | None = None,
    chat: ChatFunction | None = None,
    language: str | None = None,
    limit: int = JUDGE_ISSUE_CAP,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Ask the model where to change what for each of *run*'s first *limit* issues.

    Returns ``(issues, errors)``: the ranked issues (as in the ``issues``
    view), each with a ``judgment`` (``where``, ``fix``, ``also``,
    ``confidence``) when the model gave one, and the errors met on the way.
    *chat* replaces the built-in HTTP client; *language* sets the language of
    the suggestions.
    """
    a = run._analysis
    chat_fn = _redacted(chat) if chat is not None else None
    judged, errors = judge_overview_issues(a, list(a.issues), config=config, chat_fn=chat_fn, limit=limit,
                                           language=language, require_config=chat is None)
    return _jsonable(judged), list(errors)
