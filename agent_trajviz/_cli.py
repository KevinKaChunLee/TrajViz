"""Helpers shared by the command-line entry points (not part of the public API)."""

from __future__ import annotations

import logging
import sys

_MARK = "_agent_trajviz_cli_handler"


def configure_logging(level: int = logging.WARNING, fmt: str = "%(levelname)s %(name)s: %(message)s") -> None:
    """Send this package's log records at *level* and above to stderr.

    Library code only logs; an entry point decides where the records go. Calling
    this twice keeps one handler, so tests that call ``main()`` repeatedly do not
    multiply output.
    """
    logger = logging.getLogger("agent_trajviz")
    handler = next((h for h in logger.handlers if getattr(h, _MARK, False)), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _MARK, True)
        logger.addHandler(handler)
    handler.setFormatter(logging.Formatter(fmt))
    handler.setLevel(level)
    if logger.level == logging.NOTSET or logger.level > level:
        logger.setLevel(level)
