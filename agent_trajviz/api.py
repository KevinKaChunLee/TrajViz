"""The public API: load a trajectory, analyze it, compare two, export the results.

Everything here needs only the standard library except :func:`write_report`,
which needs the ``charts`` extra. Names not exported from ``agent_trajviz``
(the ``insight`` and ``converge`` subpackages included) are internal and may
change between minor versions.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from ._extras import require
from .errors import LoadError, ReportError
from .insight import loaders
from .insight.analysis import Analysis
from .insight.context_usage import context_pressure_series, context_pressure_stats
from .insight.parser import parse_steps
from .model import Step

if TYPE_CHECKING:
    from .views import View

#: Supported formats: format id (as in :attr:`Trajectory.format`) -> display name.
FORMATS: Mapping[str, str] = MappingProxyType(dict(loaders.FORMAT_LABELS))

#: Version of the layout :meth:`Trajectory.to_dict` produces.
SCHEMA_VERSION = 1

#: The ``type`` field that marks a :meth:`Trajectory.to_dict` document.
TRAJECTORY_TYPE = "agent-trajviz-trajectory"

PathLike = str | os.PathLike[str]


def _plain(steps: list[Step]) -> list[dict[str, Any]]:
    """The same list, typed as the internal modules see it (steps are plain dicts)."""
    return cast("list[dict[str, Any]]", steps)


@dataclass(frozen=True, eq=False)
class Trajectory:
    """One loaded agent run: its detected format and its normalized steps.

    Create one with :func:`load`. Treat ``steps`` as read-only: :attr:`metrics`,
    :attr:`health` and :attr:`agents` are computed from them once and cached.
    """

    #: Format id, a key of :data:`FORMATS` (for example ``"codex"``).
    format: str
    steps: list[Step]
    #: The file it was loaded from.
    source: str | None = None
    #: SHA-256 of the source bytes (for a merged multi-file export, of all of them).
    sha256: str | None = None
    #: What the loader had to skip or could not read, in plain words.
    warnings: tuple[str, ...] = ()
    #: True when the export recorded no token usage and the counts are estimates.
    tokens_estimated: bool = False
    #: The converted export the steps came from (format-specific; for advanced use).
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def format_label(self) -> str:
        """Display name of :attr:`format`, for example ``"Codex CLI"``."""
        return FORMATS.get(self.format, self.format)

    @cached_property
    def _analysis(self) -> Analysis:
        # The one orchestration of every analysis; views and properties share it.
        return Analysis(_plain(self.steps), self.raw, self.format)

    @property
    def metrics(self) -> dict[str, Any]:
        """Session metrics: tokens, timing, tool use, finish reasons, throughput."""
        return self._analysis.metrics

    @property
    def health(self) -> list[dict[str, Any]]:
        """Health verdicts, one per check, each with a status and an explanation."""
        return self._analysis.verdicts

    @property
    def agents(self) -> list[dict[str, Any]]:
        """Per-agent summary rows (the main agent and each sub-agent)."""
        return self._analysis.agent_summaries

    def context_pressure(self, window_limit: int | None = None) -> dict[str, Any]:
        """Context-window occupancy per agent and its summary statistics.

        Returns ``{"stats": ..., "series": ...}``. *window_limit* overrides the
        window the export declares or the model implies; without either, the
        stats say the window is assumed or unknown rather than guess silently.
        """
        series = context_pressure_series(_plain(self.steps), raw=self.raw, window_limit=window_limit)
        return {"stats": context_pressure_stats(series), "series": series}

    def views(self, *, only: Iterable[str] | None = None) -> dict[str, View]:
        """Every chart and table as a JSON-serializable :class:`View`, keyed by id.

        *only* restricts the result to those ids (see :data:`VIEW_CATALOG`);
        an unknown id raises :class:`KeyError`. Results are computed once per
        trajectory and shared between views.
        """
        from .views import build_views

        return build_views(self, self._analysis, only)

    def view(self, view_id: str, **options: Any) -> View:
        """One view by id (see :data:`VIEW_CATALOG`).

        Some views take options, listed under ``"options"`` in the catalog --
        for example ``run.view("context_pressure", agent=key, window_limit=200_000)``.
        """
        from .views import build_view

        return build_view(self, self._analysis, view_id, options)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """A JSON-serializable copy: format, provenance, warnings and steps.

        ``include_raw=True`` adds the converted export (:attr:`raw`), which
        some analyses read (session metadata, timing, sub-agent links); the
        result then reloads with :meth:`from_dict` or :func:`load` to
        identical views.
        """
        from .views import _jsonable

        data: dict[str, Any] = {
            "type": TRAJECTORY_TYPE,
            "schema_version": SCHEMA_VERSION,
            "format": self.format,
            "format_label": self.format_label,
            "source": self.source,
            "sha256": self.sha256,
            "warnings": list(self.warnings),
            "tokens_estimated": self.tokens_estimated,
            "steps": self.steps,
        }
        if include_raw:
            data["raw"] = self.raw
        return cast("dict[str, Any]", _jsonable(data))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trajectory:
        """Rebuild a trajectory from :meth:`to_dict` output (parsed JSON).

        Raises :class:`ValueError` for a document of another layout or
        version. Without ``raw`` (``to_dict()`` without ``include_raw``) the
        steps are restored but analyses that read the export's metadata see
        none.
        """
        if not isinstance(data, Mapping) or not isinstance(data.get("steps"), list):
            raise ValueError("not an agent-trajviz trajectory: no steps")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema_version {version!r} (expected {SCHEMA_VERSION})")
        fmt = data.get("format")
        if not isinstance(fmt, str) or not fmt:
            raise ValueError("not an agent-trajviz trajectory: no format")
        raw = data.get("raw")
        return cls(
            format=fmt,
            steps=cast("list[Step]", list(data["steps"])),
            source=data.get("source"),
            sha256=data.get("sha256"),
            warnings=tuple(str(w) for w in data.get("warnings") or ()),
            tokens_estimated=bool(data.get("tokens_estimated")),
            raw=dict(raw) if isinstance(raw, Mapping) else {},
        )


def load(source: PathLike, *, format: str | None = None) -> Trajectory:
    """Load a trajectory file of any supported format (see :data:`FORMATS`).

    The format is detected from the content. *format* (a key of
    :data:`FORMATS`) requires that format instead: a file detected as another
    format raises :class:`LoadError` with code ``"mismatch"``, and a file
    whose format cannot be detected is read as *format*. Raises
    :class:`LoadError` when the file cannot be read, is not a supported
    format, or holds no steps; :class:`ValueError` for an unknown *format*.
    """
    if format is not None and format not in FORMATS:
        raise ValueError(f"unknown format {format!r}; expected one of: {', '.join(FORMATS)}")
    path = os.fspath(source)
    exported = _read_exported(path)
    if exported is not None:
        return _load_exported(exported, path, format)
    raw = loaders.load_trajectory(path, format_hint=format)
    if raw.get("_error"):
        raise LoadError(str(raw["_error"]), code=raw.get("_error_code") or "load_failed", path=path)
    if raw.get("type") == TRAJECTORY_TYPE:
        # An export written with sorted keys: no format claims it, so the
        # loader hands the document back as parsed.
        return _load_exported(raw, path, format)
    if format is not None and loaders.check_format_selection(loaders.detect_format(raw), format) == "mismatch":
        raise LoadError(
            f"the file is {loaders.detect_format(raw)}, not {format}", code="mismatch", path=path,
        )
    steps = parse_steps(raw)
    fmt = raw.get("_source_format") or loaders.detect_format(raw)
    if not steps:
        raise LoadError(f"no trajectory steps found (detected format: {fmt})", code="no_steps", path=path)
    return Trajectory(
        format=fmt,
        steps=cast("list[Step]", steps),
        source=path,
        sha256=raw.get("_source_sha256"),
        warnings=tuple(str(w) for w in raw.get("_load_warnings") or ()),
        tokens_estimated=loaders.tokens_are_estimates(raw),
        raw=raw,
    )


def _load_exported(data: Mapping[str, Any], path: str, format: str | None) -> Trajectory:
    """A :meth:`Trajectory.to_dict` document read back from *path*."""
    try:
        run = Trajectory.from_dict(data)
    except ValueError as exc:
        raise LoadError(str(exc), code="unsupported_schema", path=path) from exc
    if format is not None and run.format != format:
        raise LoadError(f"the file is {run.format}, not {format}", code="mismatch", path=path)
    return run


def _read_exported(path: str) -> dict[str, Any] | None:
    """The parsed document when *path* is a :meth:`Trajectory.to_dict` export, else ``None``.

    Only a JSON object whose ``type`` is :data:`TRAJECTORY_TYPE` qualifies; a
    cheap prefix check keeps every other file on the format loaders' path.
    """
    import json

    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
            if TRAJECTORY_TYPE.encode() not in head:
                return None
            fh.seek(0)
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("type") == TRAJECTORY_TYPE else None


def compare(reference: Trajectory | PathLike, compared: Trajectory | PathLike, *,
            fuzzy_commands: bool = False, anchor_patch: PathLike | None = None) -> dict[str, Any]:
    """Align two runs of the same task and report where they diverge.

    Accepts loaded trajectories or file paths. The report includes the step
    alignment, divergence points, outcome and a confidence label; see the
    ``agent-trajviz-converge`` command for the same report from the shell.
    *anchor_patch* is a unified diff of the intended change (for example the
    task's reference patch): the comparison is then grounded on the files it
    touches, and the report gains an anchor analysis. A patch with no
    recognizable ``a/``/``b/`` headers leaves the comparison un-anchored.
    """
    from .converge.alignment import _parse_anchor_files, build_comparison_report_from_steps

    anchor_files = None
    if anchor_patch is not None:
        patch = os.fspath(anchor_patch)
        if not os.path.isfile(patch):
            raise FileNotFoundError(f"anchor patch not found: {patch}")
        anchor_files = _parse_anchor_files(patch)
    ref = reference if isinstance(reference, Trajectory) else load(reference)
    cmp = compared if isinstance(compared, Trajectory) else load(compared)
    return build_comparison_report_from_steps(
        ref.raw, cmp.raw, _plain(ref.steps), _plain(cmp.steps),
        fuzzy_commands=fuzzy_commands, anchor_files=anchor_files,
        ref_path=ref.source or "", cmp_path=cmp.source or "",
    )


def write_report(source: Trajectory | PathLike, destination: PathLike | None = None, *,
                 dark: bool = False, inline_plotly: bool = False) -> Path:
    """Write the standalone HTML report for a trajectory and return its path.

    *destination* is a file or a directory (default: a temporary file).
    ``inline_plotly`` embeds plotly.js so the report works offline. Needs the
    ``charts`` extra; raises :class:`ReportError` if the trajectory cannot be read.
    """
    require("charts", "write_report()")
    if isinstance(source, Trajectory):
        if not source.source:
            raise ReportError("this Trajectory was not loaded from a file")
        path = source.source
    else:
        path = os.fspath(source)
    from .insight.report import report_from_path

    dest = os.fspath(destination) if destination is not None else None
    return Path(report_from_path(path, dest, dark=dark, inline_plotly=inline_plotly))
