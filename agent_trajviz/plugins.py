"""Formats of your own, and the plugins installed packages declare.

A format TrajViz does not read becomes loadable by registering how to parse
it into steps (see :meth:`~agent_trajviz.Trajectory.from_steps` for the step
layout):

    >>> import agent_trajviz as tv
    >>> tv.register_format(tv.Format(
    ...     key="myagent", label="My Agent",
    ...     detect=lambda doc: isinstance(doc, dict) and doc.get("agent") == "myagent",
    ...     parse=lambda doc: [{"role": t["role"], "text_preview": t["text"]} for t in doc["turns"]],
    ... ))
    >>> tv.load("run.json").format_label
    'My Agent'

A package can do the same for everyone who installs it by declaring a hook
in the ``agent_trajviz.plugins`` entry-point group; :func:`load_plugins` runs
the hooks, and the ``agent-trajviz`` command runs them on start::

    [project.entry-points."agent_trajviz.plugins"]
    myagent = "myagent_trajviz:register"    # a function taking no arguments
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import PluginError
from .insight.loaders import FORMAT_LABELS

if TYPE_CHECKING:
    from importlib.metadata import EntryPoints

#: The entry-point group :func:`load_plugins` reads.
PLUGIN_GROUP = "agent_trajviz.plugins"

_KEY = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_BUILT_IN = frozenset(FORMAT_LABELS)
# Format id -> display name for every format; agent_trajviz.FORMATS is a
# read-only view of it, so a registered format appears there at once.
_LABELS: dict[str, str] = dict(FORMAT_LABELS)
_CUSTOM: dict[str, Format] = {}
_LOADED: list[str] = []


@dataclass(frozen=True)
class Format:
    """A trajectory format of your own: how to recognize a file and parse it into steps.

    Both callables receive the file's parsed JSON: the document, or for JSON
    Lines the list of records. ``parse`` returns the steps, mappings as
    :meth:`~agent_trajviz.Trajectory.from_steps` takes them. ``detect`` returns
    True for a file of this format; without it the format is used only when
    :func:`~agent_trajviz.load` is asked for it by ``format=key``. Detection
    runs only for files no built-in format recognizes, in registration order.
    """

    #: Format id, as in :attr:`Trajectory.format` and ``load(..., format=key)``:
    #: lowercase letters, digits, ``_``, ``.`` and ``-``.
    key: str
    #: Display name, as in :attr:`Trajectory.format_label`.
    label: str
    parse: Callable[[Any], Iterable[Mapping[str, Any]]]
    detect: Callable[[Any], bool] | None = None


def register_format(fmt: Format, *, replace: bool = False) -> None:
    """Make :func:`~agent_trajviz.load` read *fmt*, and list it in :data:`~agent_trajviz.FORMATS`.

    Raises :class:`ValueError` for a key that is malformed or built in, or
    already registered (pass ``replace=True`` to replace your own).
    """
    if not isinstance(fmt, Format):
        raise TypeError(f"expected an agent_trajviz.Format, got {type(fmt).__name__}")
    if not isinstance(fmt.key, str) or not _KEY.fullmatch(fmt.key) or fmt.key == "unknown":
        raise ValueError(f"invalid format key {fmt.key!r}: use lowercase letters, digits, '_', '.' and '-'")
    if fmt.key in _BUILT_IN:
        raise ValueError(f"{fmt.key!r} is a built-in format")
    if fmt.key in _CUSTOM and not replace:
        raise ValueError(f"format {fmt.key!r} is already registered; pass replace=True to replace it")
    _CUSTOM[fmt.key] = fmt
    _LABELS[fmt.key] = fmt.label


def registered_format(key: str | None) -> Format | None:
    """The registered (not built-in) format *key*, or ``None``."""
    return _CUSTOM.get(key) if key else None


def registered_formats() -> list[Format]:
    """The registered formats, in registration order."""
    return list(_CUSTOM.values())


def entry_points(*, group: str) -> EntryPoints:
    """The installed entry points of *group* (``importlib.metadata`` is imported only when needed)."""
    from importlib.metadata import entry_points as installed

    return installed(group=group)


def load_plugins() -> list[str]:
    """Run the hook of every installed plugin, once; returns the names of all loaded so far.

    A plugin is an entry point in the ``agent_trajviz.plugins`` group naming a
    function that takes no arguments (it typically calls
    :func:`register_format` or :func:`~agent_trajviz.register_view`). Raises
    :class:`~agent_trajviz.PluginError` naming a plugin whose hook fails.
    """
    for ep in entry_points(group=PLUGIN_GROUP):
        if ep.name in _LOADED:
            continue
        try:
            ep.load()()
        except Exception as exc:
            raise PluginError(f"plugin {ep.name!r} ({ep.value}) failed: {type(exc).__name__}: {exc}") from exc
        _LOADED.append(ep.name)
    return list(_LOADED)
