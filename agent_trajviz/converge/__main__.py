"""Launch the Converge dashboard: ``python -m agent_trajviz.converge``.

The batch/report command line is a separate entry point (``cli.py``). This
module launches the app, as ``agent_trajviz.insight.__main__`` does for the
Insight dashboard.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .. import __version__, _serve
from .._cli import configure_logging
from .._extras import require_or_exit

PROG = "python -m agent_trajviz.converge"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description="Converge dashboard: compare agent trajectories")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _serve.add_server_arguments(parser, default_port=7861)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.cwd(),
        metavar="DIR",
        help="Directory batch-manifest paths resolve against and must stay inside "
        "(default: the current directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if not args.data_root.is_dir():
        parser.error(f"--data-root {args.data_root} is not a directory")
    require_or_exit("app", "The Converge dashboard", PROG)
    try:
        launch = _serve.launch_kwargs(args)
    except ValueError as exc:
        parser.error(str(exc))
    from .app import LAUNCH_PRESENTATION, build_ui

    app = build_ui(data_root=args.data_root.resolve())
    app.launch(**launch, **LAUNCH_PRESENTATION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
