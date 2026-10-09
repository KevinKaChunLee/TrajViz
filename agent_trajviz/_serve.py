"""Launch options shared by the Insight and Converge dashboards.

Both dashboards bind to ``127.0.0.1`` by default. A ``--share`` link or a
non-loopback ``--host`` makes one reachable by anyone who can reach the port:
they can upload files, spend the LLM key configured on this machine, and --
where an app accepts server-side paths -- make it read files here. So an
exposed launch warns unless a login is set, a login comes from the
environment rather than the command line (where ``ps`` and shell history
would show the password), uploads are size-capped, and :func:`is_exposed`
lets an app switch its server-path inputs off.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os

logger = logging.getLogger(__name__)

AUTH_ENV = "TRAJVIZ_AUTH"
DEFAULT_MAX_FILE_SIZE = "500mb"


def add_server_arguments(parser: argparse.ArgumentParser, *, default_port: int) -> None:
    """Add ``--host``, ``--port``, ``--share`` and ``--max-file-size`` to *parser*."""
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: %(default)s). 0.0.0.0 accepts connections from any "
        f"machine; set {AUTH_ENV}=user:password to require a login.",
    )
    parser.add_argument("--port", type=int, default=default_port, help="Server port (default: %(default)s)")
    parser.add_argument(
        "--share",
        action="store_true",
        help=f"Create a public Gradio share link (set {AUTH_ENV} to require a login)",
    )
    parser.add_argument(
        "--max-file-size",
        default=DEFAULT_MAX_FILE_SIZE,
        metavar="SIZE",
        help="Largest accepted upload, e.g. 200mb or 2gb (default: %(default)s)",
    )


def is_loopback(host: str) -> bool:
    """Whether *host* only accepts connections from this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A hostname or an empty string: it may resolve to any interface.
        return False


def is_exposed(args: argparse.Namespace) -> bool:
    """Whether a launch with *args* is reachable from other machines."""
    return bool(args.share) or not is_loopback(args.host)


def auth_from_env() -> tuple[str, str] | None:
    """The ``(user, password)`` login from ``TRAJVIZ_AUTH``, or ``None`` when unset.

    Raises :class:`ValueError` for a value that is not ``user:password``; a
    malformed login must not quietly launch an app without one.
    """
    raw = os.environ.get(AUTH_ENV, "")
    if not raw:
        return None
    user, sep, password = raw.partition(":")
    if not sep or not user or not password:
        raise ValueError(f"{AUTH_ENV} must have the form user:password")
    return user, password


def launch_kwargs(args: argparse.Namespace) -> dict:
    """``Blocks.launch`` keyword arguments for *args*, warning on an open exposure.

    Raises :class:`ValueError` when ``TRAJVIZ_AUTH`` is malformed.
    """
    auth = auth_from_env()
    if is_exposed(args) and auth is None:
        where = "through a public share link" if args.share else f"on {args.host}:{args.port}"
        logger.warning(
            "The dashboard will be reachable %s without a login: anyone who can reach it can "
            "upload files and use the LLM key configured here. Set %s=user:password to require one.",
            where,
            AUTH_ENV,
        )
    return {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "auth": auth,
        "max_file_size": args.max_file_size,
    }
