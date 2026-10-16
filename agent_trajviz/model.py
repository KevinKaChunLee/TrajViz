"""Typed views of the records a loaded trajectory is made of.

These are ``TypedDict`` descriptions of the plain dicts every loader produces,
so they cost nothing at runtime and work with code that already treats steps
as dicts. Every format fills the same keys; a value a format does not record is
empty (``""``, ``0``, ``None`` or ``[]``) rather than missing, except where noted.
"""

from __future__ import annotations

from typing import Any, TypedDict


class TokenUsage(TypedDict, total=False):
    """Token counts of one step, as the export reported them.

    Formats nest these differently (Codex counts cache reads inside ``input``;
    some models count ``reasoning`` inside ``output``). ``total`` is the step's
    reported total and every field is counted once in it.
    """

    total: int
    input: int
    output: int
    reasoning: int
    cache_read: int
    cache_write: int
    #: Explicit context-window occupancy, for formats that record it.
    context_window: int
    #: Largest single request of a step that merged several model responses.
    prompt_peak: int
    prompt_peak_cache_read: int


class ToolCall(TypedDict, total=False):
    """One tool invocation inside a step."""

    tool_name: str
    tool_id: str
    type: str
    title: str
    status: str
    input: dict[str, Any]
    output: str
    error: str | None
    error_type: str | None
    metadata: dict[str, Any]
    duration_ms: int | None
    time_start: int | None
    time_end: int | None
    time_created: int | None
    time_updated: int | None
    time_compacted: int | None
    message_id: str
    part_id: str
    session_id: str


class Step(TypedDict, total=False):
    """One normalized step (a user, assistant or tool-result message)."""

    index: int
    raw_index: int
    #: Source file (or archive member) that raw_index refers to, when recorded.
    source_path: str
    role: str
    text_preview: str
    parts: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    tool_call_count: int
    error_count: int
    tokens: TokenUsage
    has_reasoning: bool
    finish: str
    duration: float | None
    time_created_ms: int | None
    time_completed_ms: int | float | None
    model_id: str
    provider_id: str
    agent: str
    #: Whether this step belongs to a sub-agent.
    is_sub_agent: bool
    session_id: str
    parent_session_id: str
    #: Native Claude child identity; children can share their parent's session_id.
    parent_agent_id: str
    session_depth: int | None
    session_title: str
    message_id: str
    message_type: str
    id: str
    parent_id: str
    mode: str
    cwd: str
    root: str
    summary: bool | dict[str, Any]
    is_compaction_checkpoint: bool
    compaction_reason: str
