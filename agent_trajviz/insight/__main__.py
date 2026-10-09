"""``agent-trajviz``: the Insight dashboard, or a standalone HTML report with ``--report``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .. import __version__, _serve
from .._cli import configure_logging
from .._extras import require_or_exit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-trajviz", description="TrajViz dashboard and HTML report")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _serve.add_server_arguments(parser, default_port=7860)
    parser.add_argument(
        "--report",
        metavar="TRAJECTORY",
        help="Write a standalone HTML report for TRAJECTORY and exit (no dashboard)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="HTML report output file or directory (default: a temp file; prints the path)",
    )
    parser.add_argument(
        "--dark",
        action="store_true",
        help="Render the HTML report with dark charts and tokens (with --report)",
    )
    parser.add_argument(
        "--inline-plotly",
        action="store_true",
        help="Embed plotly.js in the report so it works offline (with --report; adds several MB)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if args.report:
        require_or_exit("charts", "The HTML report", "agent-trajviz")
        from .report import ReportError, report_from_path

        try:
            path = report_from_path(args.report, args.output, dark=args.dark, inline_plotly=args.inline_plotly)
        except ReportError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(path)
        return 0

    require_or_exit("app", "The dashboard", "agent-trajviz")
    try:
        launch = _serve.launch_kwargs(args)
    except ValueError as exc:
        parser.error(str(exc))
    from .insight import LAUNCH_PRESENTATION, build_ui

    # A reachable dashboard must not let its visitors point it at paths on
    # this machine (the Attribution tab's reference-data root).
    app = build_ui(allow_server_paths=not _serve.is_exposed(args))
    # build_ui() returns a deliberately unstyled app; presentation is applied
    # at launch. Spread the shared dict so an embedder cannot drift from it.
    app.launch(**launch, **LAUNCH_PRESENTATION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
