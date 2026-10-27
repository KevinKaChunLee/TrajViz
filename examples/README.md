# Examples

Both apps use the public `agent_trajviz` API. Install `agent-trajviz[app]`, or
run `uv sync --extra app` in a checkout.

| Example | Purpose | Run |
|---|---|---|
| [`trajviz_app/`](trajviz_app/) | A dashboard built from the public API | `python examples/trajviz_app/app.py` → http://127.0.0.1:7870 |
| [`views_app.py`](views_app.py) | Browse views, export JSON and compare trajectories | `python examples/views_app.py` → http://127.0.0.1:7871 |

The examples combine `load()`, trajectory views, Plotly figures, comparisons
and optional LLM analysis. Tests in `tests/test_replica_app.py` and
`tests/test_example_app.py` exercise both apps over HTTP.
