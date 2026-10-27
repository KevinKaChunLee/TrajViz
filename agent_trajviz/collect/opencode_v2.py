"""Read-only collection of OpenCode 2 projected session records."""
from __future__ import annotations

import json
import sqlite3
from typing import Any


def _object(value: object) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    return value if isinstance(value, dict) else {}


def export_v2_session(conn: sqlite3.Connection, session_id: str, max_depth: int = 10) -> dict[str, Any]:
    """Collect the selected session and direct DB parent relationships only.

    The authentication, account and credential tables are never queried.
    Session_message.data is the native v2 record minus its id and type.
    """
    messages: list[dict] = []
    manifest: list[dict] = []
    warnings: list[str] = []
    visited: set[str] = set()

    def collect(identity: str, depth: int) -> dict:
        if depth >= max_depth:
            warnings.append(f"OpenCode collection depth limit reached at session {identity}.")
            return {}
        if identity in visited:
            return {}
        visited.add(identity)
        cursor = conn.execute("SELECT * FROM session_v2 WHERE id = ?", (identity,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Session not found: {identity}")
        fields = dict(zip((column[0] for column in cursor.description or ()), row, strict=True))
        info = {"id": identity, "projectID": fields.get("project_id"),
                "parentID": fields.get("parent_id"), "title": fields.get("title"),
                "version": fields.get("version"), "slug": fields.get("slug"),
                "location": {"directory": fields.get("directory")},
                "time": {"created": fields.get("time_created"), "updated": fields.get("time_updated")},
                "model": _object(fields.get("model")),
                "summary": {key: fields["summary_" + key] for key in ("additions", "deletions", "files")
                            if fields.get("summary_" + key) is not None}}
        manifest.append({"depth": depth, "info": info})
        rows = conn.execute("SELECT id, type, time_created, data FROM session_message "
                            "WHERE session_id = ? ORDER BY seq, id", (identity,)).fetchall()
        for message_id, kind, created, data in rows:
            try:
                record = json.loads(data) if isinstance(data, str) else data
            except (TypeError, ValueError):
                record = None
            if not isinstance(record, dict):
                warnings.append(f"OpenCode session {identity}: invalid message {message_id} skipped.")
                continue
            record = dict(record)
            record.update({"id": message_id, "type": kind, "sessionID": identity})
            time = record.get("time")
            if not isinstance(time, dict):
                time = {}
                record["time"] = time
            time.setdefault("created", created)
            record.update({"isSubAgent": depth > 0, "parentSessionID": fields.get("parent_id") or "",
                           "sessionDepth": depth, "sessionTitle": fields.get("title") or ""})
            messages.append(record)
        for (child,) in conn.execute("SELECT id FROM session_v2 WHERE parent_id = ? ORDER BY time_created, id",
                                     (identity,)).fetchall():
            collect(child, depth + 1)
        return info

    info = collect(session_id, 0)
    messages.sort(key=lambda item: (item.get("time", {}).get("created", 0), str(item.get("id", ""))))
    result: dict[str, Any] = {"info": info, "messages": messages, "session_manifest": manifest,
                              "statistics": {"subagent_sessions": max(0, len(manifest) - 1)},
                              "export_metadata": {"source_format": "opencode_v2", "schema_version": 2}}
    if warnings:
        result["_load_warnings"] = warnings
    return result
