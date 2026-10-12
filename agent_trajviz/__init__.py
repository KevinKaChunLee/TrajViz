"""agent_trajviz (TrajViz): parse and analyze LLM agent trajectories.

    >>> import agent_trajviz as tv
    >>> run = tv.load("trajectory.json")
    >>> run.format_label, len(run.steps), run.metrics["tokens"]["total"]
    >>> run.view("tool_usage").data           # any chart or table, as JSON data
    >>> tv.export_json(run, "out/")             # all of them, one file each

The names below are the public API; the core needs only the standard
library. Optional modules: :mod:`agent_trajviz.figures` (Plotly figures,
``charts`` extra) and :mod:`agent_trajviz.dashboard` (Gradio, ``app`` extra).
The ``insight`` and ``converge`` subpackages are internal and may change.
"""

import logging

from .api import FORMATS, SCHEMA_VERSION, Trajectory, compare, load, write_report
from .errors import LoadError, MissingExtraError, ReportError, TrajVizError
from .model import Step, TokenUsage, ToolCall
from .views import (
    VIEW_CATALOG,
    VIEWS_SCHEMA_VERSION,
    View,
    compare_runs,
    comparison_views,
    export_json,
    label_comparison_views,
    label_views,
    run_group_views,
)

__version__ = "0.1.0"

# A library never configures logging for its host application.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FORMATS",
    "SCHEMA_VERSION",
    "VIEWS_SCHEMA_VERSION",
    "VIEW_CATALOG",
    "LoadError",
    "MissingExtraError",
    "ReportError",
    "Step",
    "TokenUsage",
    "ToolCall",
    "Trajectory",
    "TrajVizError",
    "View",
    "compare",
    "compare_runs",
    "comparison_views",
    "export_json",
    "label_comparison_views",
    "label_views",
    "load",
    "run_group_views",
    "write_report",
]
