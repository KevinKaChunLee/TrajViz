"""agent_trajviz (TrajViz): offline analytics and visualization for LLM agent trajectories.

    >>> import agent_trajviz as tv
    >>> run = tv.load("trajectory.json")
    >>> run.format_label, len(run.steps), run.metrics["tokens"]["total"]

The names below are the public API. The ``insight`` (dashboard) and
``converge`` (comparison) subpackages are internal and may change.
"""

from .api import FORMATS, SCHEMA_VERSION, Trajectory, compare, load, write_report
from .errors import LoadError, MissingExtraError, ReportError, TrajVizError
from .model import Step, TokenUsage, ToolCall

__version__ = "1.0.0"

__all__ = [
    "FORMATS",
    "SCHEMA_VERSION",
    "LoadError",
    "MissingExtraError",
    "ReportError",
    "Step",
    "TokenUsage",
    "ToolCall",
    "Trajectory",
    "TrajVizError",
    "compare",
    "load",
    "write_report",
]
