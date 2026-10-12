"""Exceptions raised by the public API. Catch :class:`TrajVizError` for all of them."""

from __future__ import annotations


class TrajVizError(Exception):
    """Base class for every error agent_trajviz raises on purpose."""


class LoadError(TrajVizError):
    """A file could not be read as a trajectory of any supported format.

    ``code`` is a short machine-readable reason (for example ``"no_steps"`` or
    the loader's own error code) and ``path`` the file that was read.
    """

    def __init__(self, message: str, *, code: str | None = None, path: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


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
