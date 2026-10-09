"""``agent-trajviz-collect <tool> [args...]``: dispatch to one collection tool."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

TOOLS = {
    "cursor": ("agent_trajviz.collect.cursor", "Cursor agent chat (transcripts + Composer DB)"),
    "opencode": ("agent_trajviz.collect.opencode", "OpenCode session and its children (opencode.db)"),
    "codearts": ("agent_trajviz.collect.codearts", "CodeArts session (opencode.db or a session folder)"),
}


def _usage() -> str:
    lines = ["usage: agent-trajviz-collect <tool> [args...]", "", "tools:"]
    lines += [f"  {name:<9} {summary}" for name, (_, summary) in TOOLS.items()]
    lines += ["", "Run 'agent-trajviz-collect <tool> --help' for a tool's options."]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_usage(), file=sys.stdout if args else sys.stderr)
        return 0 if args else 2
    name, rest = args[0], args[1:]
    if name not in TOOLS:
        print(f"agent-trajviz-collect: unknown tool {name!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(TOOLS[name][0])
    result = module.main(rest)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
