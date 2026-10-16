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
- **Nothing leaves the machine unless you configure an endpoint.** The core
  library, the HTML report and the dashboards make no network calls: Gradio's
  usage telemetry is switched off, and a `--share` link reaches Gradio's
  tunnel service only because you asked for one. The LLM features (the AI
  analysis sidebar, the Issues judge, `agent-trajviz-label`) are off until
  you configure an endpoint. In the dashboard, the Issues judge runs after
  each load (and again if you press the button); the AI sidebar runs when
  you open it. They send a statistics brief and the step text needed for
  the question to the endpoint you configured, and nowhere else. Before any
  request leaves, credential-shaped text in it -- provider API keys, tokens,
  private keys, passwords in URLs and authorization headers, values of
  secret-named assignments -- is replaced with `[REDACTED]`. That is a
  safety net for credentials an agent happened to print, not a guarantee:
  review what you send to a third-party endpoint. Standalone reports load
  plotly.js from its CDN unless built with `--inline-plotly`.
- **Dashboards are local by default.** Both bind to `127.0.0.1`. Exposing
  one (`--host 0.0.0.0` or `--share`) lets anyone who can reach it upload
  files and spend the LLM key configured on your machine. Set
  `TRAJVIZ_AUTH=user:password` first; TrajViz warns at startup if you do not.
  An exposed Insight dashboard drops its server-path inputs, Converge
  resolves manifest paths inside `--data-root` only, uploads are capped by
  `--max-file-size` (default 500 MB), and error messages shown in the
  browser never carry a traceback.
- **Secrets stay in the environment.** API keys are read from environment
  variables or a local `.env` (only `ANALYZE_*` and `LABEL_*` keys are read
  from it), are never logged, and never appear in an error shown to a user.
