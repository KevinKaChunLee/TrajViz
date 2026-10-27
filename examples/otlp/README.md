# Saved OTLP import end-to-end audit

Audit date: 2026-10-27. Branch: `codex/otlp-import`, starting at `08e0e61`.
The scope is importing saved Claude Code OTLP files into TrajViz. Native Claude
JSONL remains supported. No live receiver or telemetry dependency is added to
TrajViz.

## Coverage matrix

| Layer | Cases exercised | Expected behavior |
| --- | --- | --- |
| Real producer | Claude Code 2.1.310 → official Collector Contrib 0.163.2 file exporters | Import the actual emitted JSONL; compare independently with native JSONL |
| Containers | JSON, JSONL, directory and explicit manifest, nested ZIP; stored, Deflate, BZIP2 and LZMA members | Equivalent selected runs and stable relocation identity |
| Multi-session graph | Interleaved sessions, child-before-parent batches, replayed records, reused request/tool IDs across agent lanes | Explicit selection, no cross-session or cross-lane joins, no double-counted usage |
| Evidence | Observed, partial, absent, redacted and truncated content; invalid token counts; API failures | Null/unavailable values stay distinguishable from measured zero; errors remain visible |
| Timing | Model requests, execution vs permission waits, zero and fractional milliseconds, overlapping tool intervals | Exact duration evidence survives metrics, charts and normalized exports |
| API and CLI | Discovery, automatic/forced format, late transport keys beyond 1 MiB, empty selectors, mismatches | Consistent selection and actionable errors; no traceback or polluted stdout |
| Views and exports | Every one of the 36 JSON views and 12 chart kinds; full and lightweight normalized exports; offline HTML | Shipped schemas validate; reloads preserve values, coverage and session identity |
| Dashboard | Upload, session changes, clear, malformed archive, native reload, report download | No stale metrics, selection or report; limited evidence is visible |
| Bounded input | Corrupt compressed members, malformed/truncated JSON, binary/framed inputs, unsupported profiles, member/total limits, traversal, symlinks and FIFOs | Refuse unsupported or unsafe input with public LoadError |
| Installed package | Clean Python 3.11.15 environment containing only agent-trajviz | Core loading/analysis/JSON export need no third-party dependencies; chart/report extras fail gracefully |

## Live producer evidence

The supplied `otelcol.yaml` validated and ran unchanged on loopback. The official
Collector archive was checked against its published SHA-256 before execution.
Claude telemetry was directed only to that local Collector. The Collector was
stopped gracefully before final file validation.

| Recording | Model requests | Tools | Agent lanes | Tokens | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Default content redaction, linter implementation | 7 | 9 | 1 | 380,366 | Counts and each token component match native JSONL; task hit its 600-second limit, so task completion is not claimed |
| Content enabled, completed independent code review | 4 | 5 | 2 | 67,902 | Native parent plus reviewer match exactly; child usage is counted once |

The completed review invoked one reviewer subagent and waited for its completed
notification. Operation-based OTLP has five steps while the native transcript
has six; this is expected and is not transcript alignment. Even content-enabled
telemetry omitted some response/tool content, which remains partial rather than
being inferred. Neither recording establishes capture completeness.

An intermediate invocation with an incorrectly delimited wrapper was stopped
and excluded from producer validation. The successful bounded review used an
explicit argument delimiter, closed stdin and read-only tools.

Raw captures and native files remain local. The checked-in fixtures replace
identities, token counts and content with controlled values and remove personal
resource attributes. They preserve producer structure, graph relationships,
missing fields and precise timing:

- `../../tests/fixtures/otlp_claude_2_1_310/`: redacted recording excerpt.
- `../../tests/fixtures/otlp_claude_2_1_310_content/`: completed parent/reviewer graph.

## Defects corrected by this audit

- Normalize corrupt ZIP decompression errors and route conflicting manifests to
  the bundle validator.
- Preserve explicit format checks and session selection for large envelopes;
  reject empty CLI selectors consistently.
- Resolve logs and tools only with unambiguous agent/request identity; support
  client request IDs when server request IDs are absent.
- Prefer valid span token evidence across aliases, with valid log fallback for
  invalid counts; retain explicit input rewrites and redaction evidence.
- Surface model failures without fabricating failed tool calls.
- Preserve observed zero, fractional timing and exact overlapping execution
  intervals through public metrics and both export modes.
- Describe partial token charts and unavailable tool timing accurately.
- Show limited evidence in Issues, unavailable usage in workflow cards, and
  escaped model error details in workflow detail and Raw.
- Discover large multi-session envelopes in the dashboard and clear prior
  capture state and report exports across load transitions.

The original mapper fails 14 of the first 23 new mapping cases; the fixed mapper
passes them. Fractional/zero timing regressions also reproduced before their
fix. These checks exercise public output, rather than only internal helpers.

## Reproduce

From this worktree, install the locked development environment and run:

```bash
uv sync --locked --extra dev
uv run --locked python -m pytest tests/test_otlp*.py -q
uv run --locked python -m pytest tests -q -rs
uv run --locked ruff check agent_trajviz tests examples
```

The subprocess, bundle, mapping, UI and recorded-producer suites are in
`tests/test_otlp_e2e.py`, `test_otlp_bundle_e2e.py`,
`test_otlp_mapping_e2e.py`, `test_otlp_ui_e2e.py` and
`test_otlp_recorded_latest.py`.

The separate installed-wheel audit exercised twelve selected captures
(observed/partial × directory/manifest/ZIP × two sessions), validating **432
exported views**, **24 full/lightweight all-view roundtrips**, native subagents,
ten clean CLI errors and six expected missing-extra errors. It ran outside the
checkout and confirmed exactly one installed distribution.

Live browser checks confirmed selection of each session, unavailable token and
outcome displays, clear/error/native recovery, the corrected Issues heading and
a selected-session HTML download. The downloaded report contains the capture
warning and unavailable checks and does not claim that no errors were detected.

## Final verification

- Full suite: **1,745 passed, 892 subtests passed, 46 skipped**. Skips require
  optional dependencies or external/reference fixtures; no OTLP test was skipped.
- Ruff: passed across package, tests and examples.
- Strict mypy: passed for all six public API modules.
- Locked dependency resolution: passed offline.
- Wheel and source distribution: built offline; both passed `twine check`.
- Independent final code review: no outstanding actionable findings.

These results describe the isolated branch. The audit does not publish a package
or merge changes into main.

## Limits

This validates Claude Code's current beta span profile on macOS arm64, with
Python 3.13.9 development checks and Python 3.11.15 installed-core checks. It does
not claim all operating systems or future producer versions. Other OTLP producer
profiles, metrics-only/logs-only captures and binary/framed Collector files
remain unsupported. Native transcript comparison and failure attribution remain
unavailable for operation-based OTLP inputs.
