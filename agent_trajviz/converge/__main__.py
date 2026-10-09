"""Launch the Converge Gradio app via `python -m agent_trajviz.converge`.

The batch/report CLI is a separate entry point: the ``trajectory-converge``
console script, or ``python -m agent_trajviz.converge.cli``. This module intentionally
launches the app (same precedent as ``agent_trajviz.insight.__main__``), so do not
repoint it at the CLI — that would break every documented app invocation.
"""

from .._cli import configure_logging
from .._extras import require_or_exit


def main():
    configure_logging()
    require_or_exit("app", "The Converge dashboard", "agent_trajviz.converge")
    from .app import LAUNCH_PRESENTATION, build_ui

    app = build_ui()
    app.launch(**LAUNCH_PRESENTATION)


if __name__ == "__main__":
    main()
