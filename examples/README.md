# Examples

Both apps use only the public `agent_trajviz` API. Install the package with
the `app` extra first (`pip install "agent-trajviz[app]"`, or `uv sync --extra
app` in a checkout).

| Example | What it shows |
|---|---|
| [`trajviz_app/`](trajviz_app/) | The TrajViz dashboard rebuilt on the public API: KPI strip, Overview sections (summary, tools, agents, context utilization, diagnostics, deep dive, labels), Patterns, Attribution, Comparison (run group and pairwise), Workflow, Raw Data and the AI sidebar. `python examples/trajviz_app/app.py` → http://127.0.0.1:7870 |
| [`views_app.py`](views_app.py) | A small app of your own: summary, health and issues, any chart or table from the views, a JSON download of every view, and a comparison. `python examples/views_app.py` → http://127.0.0.1:7871 |
| [`otlp/`](otlp/) | Local Collector configuration and manifest for recording Claude Code OTLP traces and companion logs; see the README's saved-capture guide. |

The API they use:

- `agent_trajviz.load()` and the `Trajectory` views — every chart's and
  table's data as JSON (`run.views()`, `run.view(id, **options)`);
- `agent_trajviz.figures.figure(view)` — the dashboard's Plotly figure for a
  chart view;
- `agent_trajviz.compare()`, `compare_runs()`, `label_views()` and their
  `*_views()` — comparisons and step labels;
- `agent_trajviz.llm` — the AI analysis and suggested fixes;
- `agent_trajviz.attribution` — DECAF failure attribution, when DECAF is
  installed.

`tests/test_replica_app.py` and `tests/test_example_app.py` launch each app
on a local port and drive it over HTTP with Gradio's client.
