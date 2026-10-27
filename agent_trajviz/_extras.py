"""Optional features and the extra that installs each, with a clear install hint.

The core (loading, parsing, metrics, context usage, patterns, comparison and the
collection tools) needs only the standard library. Everything else is an extra:

* ``charts`` — plotly and pygments: figures and the static HTML report;
* ``llm``    — requests: the AI analysis sidebar, the Issues judge, the labeler;
* ``app``    — gradio plus both of the above: the Insight and Converge dashboards;
* ``zstd``   — zstandard on Python 3.11–3.13: compressed native session logs
  (Python 3.14+ supplies ``compression.zstd`` in its standard library).
"""

from __future__ import annotations

import importlib.util

from .errors import MissingExtraError

DISTRIBUTION = "agent-trajviz"

EXTRAS: dict[str, tuple[str, ...]] = {
    "charts": ("plotly", "pygments"),
    "llm": ("requests",),
    "app": ("gradio", "plotly", "pygments", "requests"),
}


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # a broken or shadowed installation counts as absent
        return False


def missing(extra: str) -> list[str]:
    """Import names of *extra* that cannot be found in this environment."""
    return [name for name in EXTRAS[extra] if not _importable(name)]


def require(extra: str, feature: str) -> None:
    """Raise :class:`MissingExtraError` unless every package of *extra* is importable."""
    gaps = missing(extra)
    if gaps:
        raise MissingExtraError(extra, gaps, feature, DISTRIBUTION)


def require_or_exit(extra: str, feature: str, prog: str) -> None:
    """Command-line form of :func:`require`: print the hint and exit with status 2."""
    import sys

    try:
        require(extra, feature)
    except MissingExtraError as exc:
        print(f"{prog}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
