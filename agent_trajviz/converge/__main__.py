"""Launch the Converge Gradio app via `python -m agent_trajviz.converge`.

The batch/report CLI is a separate entry point: the ``trajectory-converge``
console script, or ``python -m agent_trajviz.converge.cli``. This module intentionally
launches the app (same precedent as ``agent_trajviz.insight.__main__``), so do not
repoint it at the CLI — that would break every documented app invocation.
"""

from .app import LAUNCH_PRESENTATION, build_ui


def main():
    app = build_ui()
    app.launch(**LAUNCH_PRESENTATION)


if __name__ == "__main__":
    main()
