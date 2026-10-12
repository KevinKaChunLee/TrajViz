"""Exceptions raised by the public API. Catch :class:`TrajVizError` for all of them."""

from __future__ import annotations


class TrajVizError(Exception):
    """Base class for every error agent_trajviz raises on purpose."""


class LoadError(TrajVizError):
    """A file could not be read as a trajectory of any supported format.

    ``code`` is a short machine-readable reason: ``"unknown_format"`` (no
    format recognizes the file), ``"mismatch"`` (the file is not the format
    asked for; ``detected`` names what it is), ``"no_steps"``,
    ``"parse_failed"`` (a registered format's parser raised),
    ``"unsupported_schema"`` (an exported trajectory of another version),
    ``"invalid_labels"``, or ``"load_failed"`` and the loader's own codes for
    a file that cannot be read. ``path`` is the file that was read.
    """

    def __init__(self, message: str, *, code: str | None = None, path: str | None = None,
                 detected: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        #: The format the file was detected as, when known (set for ``"mismatch"``).
        self.detected = detected


class PluginError(TrajVizError):
    """An installed plugin's hook failed to load or raised (see :func:`agent_trajviz.load_plugins`)."""


class ReportError(TrajVizError, ValueError):
    """A trajectory could not be turned into an HTML report."""


class MissingExtraError(TrajVizError, ImportError):
    """An optional feature was used without the extra that provides it."""

    def __init__(self, extra: str, missing: list[str], feature: str, distribution: str = "agent-trajviz") -> None:
        self.extra = extra
        self.missing = missing
        super().__init__(
            f"{feature} needs the '{extra}' extra (not installed: {', '.join(missing)}). "
            f'Install it with: pip install "{distribution}[{extra}]"'
        )
