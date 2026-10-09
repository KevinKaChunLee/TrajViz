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
  `trajectory-converge`, `agent-trajviz-collect cursor|opencode|codearts` and
  `agent-trajviz-label`; each takes `--version`.
- Supported formats: Claude Code, Cursor, OpenCode, CodeArts, ICode, Codex CLI,
  Pi and DeepSeek Harness.

### Changed

- The import package is `agent_trajviz` (was `trajviz`) and the dashboard
  command is `agent-trajviz` (was `trajviz`).
- The collection and labeling tools moved from `scripts/` into the package.
- The step labeler is one command, `agent-trajviz-label`: `--assistant-only`
  replaces the separate compatibility command, the default output is
  `<input>_labeled.json`, and progress goes through `logging`. Module and
  file names no longer carry version labels (`step_labeler_v2` is
  `agent_trajviz.labeling.labeler`); the CodeArts tool writes
  `<session>_trajectory.json`.
- `requirements.txt` is gone; `pyproject.toml` is the single source of
  dependencies.
