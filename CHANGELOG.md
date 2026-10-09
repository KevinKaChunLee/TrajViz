# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/): before 1.0, a minor version may
change the public API (the names exported from `agent_trajviz`).

## [0.1.0] - Unreleased

First release as the `agent-trajviz` package.

### Added

- A public library API: `load()` returns a `Trajectory` (format, normalized
  steps, provenance, load warnings) with cached `metrics`, `health` and
  `agents`, plus `context_pressure()` and a versioned, JSON-serializable
  `to_dict()`; `compare()` aligns two runs; `write_report()` writes the
  standalone HTML report. Steps are described by the `Step`, `ToolCall` and
  `TokenUsage` typed records; errors derive from `TrajVizError` (`LoadError`,
  `ReportError`, `MissingExtraError`). The package ships `py.typed`.
- Optional extras: `charts` (figures, HTML report), `llm` (AI sidebar, Issues
  judge, labeler), `app` (both dashboards) and `all`. The core needs only the
  standard library, and a command whose extra is missing names it.
- Console commands: `agent-trajviz` (dashboard, or `--report` for HTML),
  `agent-trajviz-converge`, `agent-trajviz-collect cursor|opencode|codearts`
  and `agent-trajviz-label`; each takes `--version`. The Converge dashboard
  is `python -m agent_trajviz.converge`.
- Supported formats: Claude Code, Cursor, OpenCode, CodeArts, ICode, Codex CLI,
  Pi and DeepSeek Harness.
- Dashboard server options shared by both apps: `--host`, `--port`, `--share`
  and `--max-file-size` (default 500 MB), and a login from
  `TRAJVIZ_AUTH=user:password`. The Converge dashboard takes `--data-root`.
- `SECURITY.md`: how to report a vulnerability, and the security model.
- Python 3.14 support.

### Changed

- The import package is `agent_trajviz` (was `trajviz`) and the dashboard
  command is `agent-trajviz` (was `trajviz`). Every command carries the
  `agent-trajviz` prefix; the Converge command line is
  `agent-trajviz-converge` (was `trajectory-converge`).
- The collection and labeling tools moved from `scripts/` into the package.
- The step labeler is one command, `agent-trajviz-label`: `--assistant-only`
  replaces the separate compatibility command, the default output is
  `<input>_labeled.json`, and progress goes through `logging`. Module and
  file names no longer carry version labels (`step_labeler_v2` is
  `agent_trajviz.labeling.labeler`); the CodeArts tool writes
  `<session>_trajectory.json`.
- Every command's entry point takes `argv` and returns its exit status: 0
  success, 1 a failure caused by the input, 2 a usage error. Input failures
  print one sentence; a genuine bug keeps its traceback.
- Library code logs through `logging` (the package root has a
  `NullHandler`) instead of printing tracebacks to stderr; the commands
  attach a stderr handler.
- Agent rows (`Trajectory.agents`, the Agents cards) report
  `output_tokens_per_sec` -- output tokens over generation time, the same
  measurement as the Overview's "gen tok/s" -- with its timed-step coverage,
  instead of all tokens over wall-clock time. Cards count "assistant steps".
- Converge: the batch aggregate's `success_count`/`failure_count` are
  `compared_count`/`error_count` (they counted comparisons, not successful
  runs) and per-task status is `compared`/`error`; the pairwise summary says
  whether each run "stopped normally" rather than "Success"/"Failure".
- Converge before/after significance is a pure-Python Wilcoxon signed-rank
  test with Holm correction across metrics (`paired_significance`, was
  `test_significance`), so a result no longer depends on whether SciPy is
  installed.
- The Converge dashboard listens on port 7861 by default, beside Insight's
  7860.
- Dependency lower bounds are the oldest releases the suite passes with:
  plotly 5.15, pygments 2.13, requests 2.32.4, pytest 8.0, hypothesis 6.100.

### Removed

- `requirements.txt`; `pyproject.toml` is the single source of dependencies.
- The `tokens_per_second` field of agent rows (see `output_tokens_per_sec`).

### Fixed

- A run with no tool calls reported a tool success rate of 0% ("every call
  failed"); the rate is now `None` and shown as n/a.
- The per-step token chart stacked the remaining fields of a step whose
  export reported an impossible count (a negative input), drawing it taller
  than its own total. Such steps are drawn as their reported total in a
  separate, labelled series, and the title counts them.
- The Converge dashboard resolved a manifest's relative paths against the
  upload's temporary directory, so they could never be found.
- A cleared Token Rate box in the Converge dashboard raised a `TypeError`.
- `agent-trajviz-collect opencode` printed usage examples without its
  subcommand.
- Help and setup text that described a source checkout to installed users,
  or claimed cache-read anomalies were left out of the token totals.

### Security

- An exposed dashboard (`--share`, or a non-loopback `--host`) warns unless
  `TRAJVIZ_AUTH` sets a login, and hides inputs that name paths on the
  serving machine (the Attribution tab's reference-data root), ignoring such
  a value even when a client sends one.
- Converge manifest paths must stay inside `--data-root`, `..` and symlinks
  included.
- Errors shown in the Converge dashboard carry no traceback and no server
  paths.
- The dashboards no longer send Gradio's usage telemetry.
