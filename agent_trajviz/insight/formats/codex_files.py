"""Bounded, explicit Codex rollout trees; never search a user's global store."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

from .parse import _parse_trajectory_text

MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_CHILDREN = 256
MAX_DEPTH = 16
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


@dataclass
class CodexSessionFiles:
    events: list
    children: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    child_hashes: dict[str, str] = field(default_factory=dict)


def session_meta(events: list) -> dict:
    """The owning header comes first; forked history may contain older headers."""
    for event in events:
        if isinstance(event, dict) and event.get("type") == "session_meta":
            payload = event.get("payload")
            return payload if isinstance(payload, dict) else {}
    return {}


def parent_thread_id(meta: dict) -> str:
    source = meta.get("source")
    source = source if isinstance(source, dict) else {}
    subagent = source.get("subagent")
    subagent = subagent if isinstance(subagent, dict) else {}
    spawned = subagent.get("thread_spawn")
    spawned = spawned if isinstance(spawned, dict) else {}
    direct, nested = meta.get("parent_thread_id"), spawned.get("parent_thread_id")
    # Disagreement is not evidence of an owning parent.
    if direct and nested and direct != nested:
        return ""
    value = direct or nested
    return value if isinstance(value, str) else ""


def agent_path(meta: dict) -> str:
    value = meta.get("agent_path")
    if isinstance(value, str) and value:
        return value
    source = meta.get("source")
    subagent = source.get("subagent") if isinstance(source, dict) else None
    spawned = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
    value = spawned.get("agent_path") if isinstance(spawned, dict) else None
    return value if isinstance(value, str) else ""


def own_codex_events(events: list) -> tuple[list, list[str]]:
    """Remove copied parent history using Codex's declared ordinal boundary."""
    meta = session_meta(events)
    boundary = meta.get("subagent_history_start_ordinal")
    if boundary is None:
        return events, []
    if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
        return [], ["Codex child has an invalid inherited-history boundary; skipped to avoid double counting."]
    first = next((event for event in events if isinstance(event, dict) and event.get("type") == "session_meta"), None)
    records = [event for event in events if isinstance(event, dict)
               and isinstance(event.get("ordinal"), int) and not isinstance(event.get("ordinal"), bool)
               and event["ordinal"] >= boundary and event.get("type") != "session_meta"]
    # An unnumbered body cannot be safely assigned before/after the boundary.
    unnumbered = any(isinstance(event, dict) and event.get("type") != "session_meta"
                     and (not isinstance(event.get("ordinal"), int) or isinstance(event.get("ordinal"), bool))
                     for event in events)
    warnings = ["Codex child has unnumbered records outside its history boundary; those records were skipped."] if unnumbered else []
    return ([first] if first else []) + records, warnings


def read_codex_events(path: Path, *, base: Path | None = None) -> tuple[list, bytes]:
    """Read a regular non-symlink rollout once and retain exactly those bytes."""
    if path.is_symlink():
        raise ValueError("symlinked rollout")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("rollout is not a regular file")
    if before.st_size > MAX_MEMBER_BYTES:
        raise ValueError("rollout size limit exceeded")
    directories = []
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
        if base is not None and os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
            relative = path.relative_to(base)
            current = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            directories.append(current)
            for part in relative.parts[:-1]:
                current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
                directories.append(current)
            descriptor = os.open(relative.name, flags, dir_fd=current)
        else:
            descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise ValueError("rollout changed while opening")
            data = handle.read(MAX_MEMBER_BYTES + 1)
    finally:
        for directory in reversed(directories):
            os.close(directory)
    if len(data) > MAX_MEMBER_BYTES:
        raise ValueError("rollout size limit exceeded")
    events, error = _parse_trajectory_text(data.decode("utf-8-sig"), str(path))
    if error or not isinstance(events, list) or not session_meta(events):
        raise ValueError(error or "not a native Codex rollout")
    return events, data


def _delegation_paths(events: list) -> set[str]:
    paths = set()
    calls = set()
    for event in events:
        payload = event.get("payload") if isinstance(event, dict) else None
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "function_call" and str(payload.get("name", "")).split(".")[-1] == "spawn_agent":
            calls.add(payload.get("call_id"))
        elif payload.get("type") == "function_call_output" and payload.get("call_id") in calls:
            output = payload.get("output")
            try:
                output = json.loads(output) if isinstance(output, str) else output
            except ValueError:
                continue
            if isinstance(output, dict):
                value = output.get("task_name") or output.get("agent_id") or output.get("thread_id")
                if isinstance(value, str) and value:
                    paths.add(value)
    return paths


def discover_codex_children(events: list, source_path: str) -> CodexSessionFiles:
    """Read only ``<selected stem>/subagents`` and verify header parent links.

    Recent Codex versions return canonical agent names rather than thread UUIDs
    from ``spawn_agent``. Such UUIDs are available to the explicit collector;
    loading a file never looks them up in global state or neighboring sessions.
    """
    bundle = CodexSessionFiles(events)
    source = Path(source_path)
    tree = source.with_suffix("") / "subagents"
    base = source.parent
    if any(path.is_symlink() for path in (tree, tree.parent)):
        bundle.warnings.append("Symlinked Codex subagent tree was skipped.")
        return bundle
    pending = {}
    total = 0
    inspected = 0
    directory_count = 0
    if tree.is_dir():
        for current, dirs, files in os.walk(tree, followlinks=False):
            directory_count += 1
            if directory_count > MAX_CHILDREN * 2:
                bundle.warnings.append("Codex child directory count limit reached.")
                break
            directory = Path(current)
            dirs[:] = sorted(name for name in dirs if not (directory / name).is_symlink())
            if len(directory.relative_to(tree).parts) >= MAX_DEPTH:
                dirs[:] = []
            for name in sorted(files):
                if not name.startswith("agent-") or not name.endswith(".jsonl"):
                    continue
                if inspected >= MAX_CHILDREN:
                    bundle.warnings.append(f"Only the first {MAX_CHILDREN} Codex child rollouts were inspected.")
                    break
                path = directory / name
                label = path.relative_to(base).as_posix()
                inspected += 1
                try:
                    # Charge invalid files too; damaged children cannot bypass
                    # either the number of inspections or the byte budget.
                    size = path.lstat().st_size
                    if total + size > MAX_TOTAL_BYTES:
                        raise ValueError("rollout tree size limit exceeded")
                    total += size
                    child, data = read_codex_events(path, base=base)
                    meta = session_meta(child)
                    identity = meta.get("id")
                    if not isinstance(identity, str) or not _SAFE_ID.fullmatch(identity):
                        raise ValueError("invalid child thread ID")
                    if path.stem != f"agent-{identity}":
                        raise ValueError("child filename disagrees with its header ID")
                    if identity in pending:
                        raise ValueError("duplicate child thread ID")
                    pending[identity] = {"events": child, "source_path": str(path),
                                         "parent_thread_id": parent_thread_id(meta),
                                         "sha256": hashlib.sha256(data).hexdigest()}
                except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                    bundle.warnings.append(f"Codex child {label} could not be read ({exc}); skipped.")
    root = session_meta(events).get("id")
    accepted = {root}
    depths = {root: 0}
    for _ in range(MAX_DEPTH):
        changed = False
        for identity, info in sorted(pending.items()):
            if identity in accepted or info["parent_thread_id"] not in accepted:
                continue
            depth = depths[info["parent_thread_id"]] + 1
            if depth >= MAX_DEPTH:
                continue
            bundle.children[identity] = info
            bundle.child_hashes[f"{info['parent_thread_id']}/{identity}"] = info["sha256"]
            accepted.add(identity)
            depths[identity] = depth
            changed = True
        if not changed:
            break
    for identity in sorted(set(pending) - accepted):
        bundle.warnings.append(f"Codex child {identity} has no verified parent in the selected tree; skipped.")
    expected = _delegation_paths(events)
    found = set(bundle.children)
    found.update(agent_path(session_meta(info["events"])) for info in bundle.children.values())
    if expected - found:
        bundle.warnings.append("Codex delegation was recorded but some child rollouts are missing; "
                               "collect the session with agent-trajviz-collect codex to include them.")
    return bundle
