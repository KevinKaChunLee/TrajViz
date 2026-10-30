# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, through **Report a vulnerability**
on the repository's **Security** tab, rather than in a public issue. Include
the version (`agent-trajviz --version`), what an attacker controls (a
trajectory file, a manifest, network access to a dashboard, ...), and the
smallest input that shows the problem. You should get a first response
within a week.

Only the latest release receives security fixes.

## Security model

TrajViz analyzes agent trajectories, which routinely contain source code,
absolute paths, shell commands and credentials that an agent happened to
print. It is built to keep them on your machine.

- **Files are untrusted input.** Every loader treats a trajectory as
  hostile: malformed, oversized or adversarial files must fail with a
  `LoadError` (or a clear message on the command line), never execute
  anything, and never write outside the output you asked for. Content is
  HTML-escaped wherever it is rendered, including the standalone report.
- **Core parsing and metrics run locally.** They make no network calls.
  Gradio's usage telemetry is disabled. A `--share` link uses Gradio's tunnel
  service; standalone reports load plotly.js from its CDN unless built with
  `--inline-plotly`.
- **Model assistance uses the configured endpoint.** Once a connection is
  ready, opening the AI panel for a loaded trajectory can request analysis;
  loading a new trajectory while that panel is open can do the same.
  Submitting a question, pressing **Suggest fixes**, or running
  `agent-trajviz-label` also requests model assistance. Applying Settings
  sends no model request. These features send a statistics brief and relevant
  step text to the configured provider. Before a model request leaves,
  credential-shaped text in it -- provider API keys, tokens, private keys,
  passwords in URLs and authorization headers, values of secret-named
  assignments -- is replaced with `[REDACTED]`. That is a safety net for
  credentials an agent happened to print, not a guarantee: review what you
  send to a third-party endpoint.
- **Dashboards are local by default.** Both bind to `127.0.0.1`. Exposing
  one (`--host 0.0.0.0` or `--share`) lets anyone who can reach it upload
  files and spend the LLM key configured on your machine. Set
  `TRAJVIZ_AUTH=user:password` first; TrajViz warns at startup if you do not.
  An exposed Insight dashboard drops its server-path inputs, Converge
  resolves manifest paths inside `--data-root` only, uploads are capped by
  `--max-file-size` (default 500 MB), and error messages shown in the
  browser never carry a traceback.
- **API keys are not remembered in the browser.** Keys can come from
  environment variables, a local `.env` (only `ANALYZE_*` and `LABEL_*` keys
  are read from it), or dashboard Settings. Keys entered in Settings stay in
  the current server-side session; optional remembered browser preferences
  exclude API keys. Settings do not write keys to `.env`. Keys are never
  logged or included in errors shown to a user.
