"""Export exactly one Codex thread and its verified descendants, read-only.

The selected thread's explicit spawn edges are read from Codex's state database;
no session listing, title search or unrelated transcript scan is performed.
The output is a portable consolidated JSON trajectory, loadable without Codex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from .._fileio import ensure_output_does_not_overwrite, write_json_atomic
from ..insight.formats.codex import _convert_codex_to_internal
from ..insight.formats.codex_files import (
    MAX_CHILDREN,
    MAX_DEPTH,
    MAX_TOTAL_BYTES,
    parent_thread_id,
    read_codex_events,
    session_meta,
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _rollout_path(value: object, home: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Codex state has no rollout path for this thread")
    path = Path(value)
    if not path.is_absolute():
        path = home / path
    resolved = path.resolve()
    roots = [home.resolve() / "sessions", home.resolve() / "archived_sessions"]
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError("Codex rollout path escapes the selected Codex session store")
    relative = path.relative_to(home)
    if any(home.joinpath(*relative.parts[:index]).is_symlink() for index in range(1, len(relative.parts) + 1)):
        raise ValueError("Codex rollout path contains a symlink")
    return path


def collect_codex_session(session_id: str, *, codex_home: str | Path | None = None,
                          db: str | Path | None = None) -> tuple[dict, list[Path]]:
    """Return a portable conversion and its exact protected source paths."""
    if not isinstance(session_id, str) or not _SAFE_ID.fullmatch(session_id):
        raise ValueError("Choose an exact Codex thread ID")
    home = Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser().absolute()
    database = Path(db).expanduser().absolute() if db is not None else home / "state_5.sqlite"
    if not database.is_file() or database.is_symlink():
        raise ValueError("Codex state database was not found; pass --codex-home or --db explicitly")
    uri = database.resolve().as_uri() + "?mode=ro"
    warnings = []
    sources = [database, Path(str(database) + "-wal"), Path(str(database) + "-shm")]
    children = {}
    budget = [0]
    with closing(sqlite3.connect(uri, uri=True, timeout=10)) as connection:
        connection.execute("PRAGMA query_only=ON")

        def read(identity: str, parent: str = "") -> tuple[list, bytes, Path]:
            row = connection.execute("SELECT rollout_path FROM threads WHERE id = ?", (identity,)).fetchone()
            if not row:
                raise ValueError(f"Codex thread {identity} was not found")
            path = _rollout_path(row[0], home)
            # Charge malformed files as well, before parsing them. Otherwise
            # failed children could repeatedly bypass the total byte budget.
            size = path.lstat().st_size
            if budget[0] + size > MAX_TOTAL_BYTES:
                raise ValueError("Codex collection size limit exceeded")
            budget[0] += size
            events, data = read_codex_events(path)
            budget[0] += max(0, len(data) - size)
            if budget[0] > MAX_TOTAL_BYTES:
                raise ValueError("Codex collection size limit exceeded")
            meta = session_meta(events)
            if meta.get("id") != identity:
                raise ValueError("Codex rollout header disagrees with its database thread ID")
            if parent and parent_thread_id(meta) != parent:
                raise ValueError("Codex child header disagrees with its database spawn edge")
            sources.append(path)
            return events, data, path

        events, data, source = read(session_id)
        queue = [(session_id, 0)]
        seen = {session_id}
        while queue:
            parent, depth = queue.pop(0)
            rows = connection.execute(
                "SELECT child_thread_id FROM thread_spawn_edges WHERE parent_thread_id = ? "
                "ORDER BY child_thread_id LIMIT ?",
                (parent, MAX_CHILDREN + 1),
            ).fetchall()
            for row in rows:
                identity = row[0]
                if len(seen) > MAX_CHILDREN:
                    warnings.append("Codex collection child inspection limit reached.")
                    break
                if identity in seen:
                    warnings.append("Duplicate or cyclic Codex spawn edge was skipped.")
                    continue
                if not isinstance(identity, str) or not _SAFE_ID.fullmatch(identity):
                    warnings.append("Codex state contains an invalid child thread ID; skipped.")
                    continue
                seen.add(identity)
                if depth + 1 >= MAX_DEPTH or len(children) >= MAX_CHILDREN:
                    warnings.append("Codex collection depth or child-count limit reached; remaining descendants skipped.")
                    continue
                try:
                    child_events, child_data, path = read(identity, parent)
                    sha = hashlib.sha256(child_data).hexdigest()
                    children[identity] = {"events": child_events, "source_path": str(path),
                                          "parent_thread_id": parent, "sha256": sha}
                    queue.append((identity, depth + 1))
                except (OSError, ValueError, UnicodeError, RecursionError) as exc:
                    warnings.append(f"Codex child {identity} could not be collected ({exc}); skipped.")
    result = _convert_codex_to_internal(events, source_path=str(source),
                                        child_event_lists=children, child_warnings=warnings)
    parent_sha = hashlib.sha256(data).hexdigest()
    result["_codex_source_files"] = [
        {"thread_id": session_id, "sha256": parent_sha, "source_path": str(source)},
        *[{"thread_id": identity, "parent_thread_id": info["parent_thread_id"],
           "sha256": info["sha256"], "source_path": info["source_path"]}
          for identity, info in sorted(children.items())],
    ]
    result["_codex_collection"] = {"version": 1, "session_id": session_id,
                                   "child_count": len(children), "source_bytes": budget[0]}
    return result, sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-trajviz-collect codex", description=__doc__)
    parser.add_argument("session_id", help="Exact Codex thread ID (no implicit last-session selection)")
    parser.add_argument("-o", "--output", help="Consolidated JSON output, or - for stdout")
    parser.add_argument("--codex-home", type=Path, help="Codex state directory (default: CODEX_HOME or ~/.codex)")
    parser.add_argument("--db", type=Path, help="Explicit read-only Codex state database")
    args = parser.parse_args(argv)
    output = args.output or f"codex-{args.session_id}.json"
    try:
        result, sources = collect_codex_session(args.session_id, codex_home=args.codex_home, db=args.db)
        ensure_output_does_not_overwrite(output, sources, allow_stdout_dash=True)
        if output == "-":
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        else:
            write_json_atomic(output, result)
        print(f"Collected Codex session with {result['_codex_collection']['child_count']} child thread(s); "
              f"{len(result.get('_load_warnings', []))} warning(s).", file=sys.stderr)
    except (OSError, ValueError, sqlite3.Error, UnicodeError, RecursionError) as exc:
        print(f"agent-trajviz-collect codex: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
