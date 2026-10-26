# Native coding-agent compatibility audit

Test date: 2026-10-26. Each agent ran the repository task in [task.md](task.md)
in its own copy of the same TrajViz commit. The task required a batch audit
command, isolated input errors, atomic JSON/CSV output, tests, documentation,
and review when the agent supported it. Accounts and model choices differed;
the evidence below checks collection and parsing.

| Agent | Tested release | Primary capture |
|---|---|---|
| Claude Code | 2.1.310 | Saved session JSONL and associated subagent files |
| Cursor Agent CLI | 2026.10.22-27dcf5f | Native stream JSONL and collected transcript family |
| OpenCode | 1.19.4 | Native session export and read-only database collection |
| Codex CLI | 0.164.0 | Native rollout family and consolidated collector JSON |
| Pi | 1.4.0 | Native v3 session JSONL |
| DeepSeek Harness | 0.2.1-rc.2 (npm latest) | Native v4 compressed session files |

## Verification procedure

1. Update the agent from its official distribution before starting a new session.
2. Use an isolated public repository copy and a unique session ID/title. Keep
   saved authentication outside the workspace and capture only the new run.
3. Run the task to completion and execute independent probes for mixed native
   and OTLP inputs, missing inputs, CSV quoting, stable order, repeated output,
   duplicate IDs and unsupported manifest versions.
4. Load the captured source through `agent_trajviz.load`, check tools, usage,
   model IDs, timing and child relationships against the native records.
5. Validate all 36 view documents against their JSON schemas, export the JSON
   views, render a standalone HTML report, and check full trajectory round trips.
   Lightweight exports preserve steps; use `include_raw=True` to preserve all
   session metadata used by analyses.
6. Open the capture in the dashboard and inspect Summary, Tools, Agents, Context
   Utilization and the detailed workflow. Check that incomplete evidence is
   visible instead of being presented as a complete measurement.

Raw runs stay local because agent session headers and tool output can contain
machine paths and account metadata. Small regression fixtures retain only the
schema details needed to reproduce a parsing defect.

## Findings

- Cursor IDE JSONL must split only at JSONL record boundaries. Unicode line
  separators inside JSON strings are valid content.
- Cursor CLI streams use a separate protocol with thinking deltas, typed tool
  start/completion events and complete-message flushes. Transcript mirrors can
  supply child activity while omitting outputs, clocks and billed token usage.
- Peer Cursor reviewer logs must remain peer children; scanning the enclosing
  subagent folder from each child creates false cycles and depths.
- Codex reviewer rollouts include inherited parent records. Trim the declared
  history prefix before aggregating usage and tool calls. Verify parent IDs and
  collect exact spawn edges to make portable parent/child captures.
- Pi reasoning tokens are part of output tokens. Request timing and tool timing
  describe separate intervals; the next message timestamp can include idle time.
- DeepSeek Harness uses numbered native generations and concatenated Zstandard
  frames. Select the newest supported generation and follow explicit child
  catalog references, retaining hashes of the bytes actually read.

## Captured activity

| Agent | Steps | Agent lanes | Tool calls | Independent acceptance |
|---|---:|---:|---:|---:|
| Claude Code | 29 | 2 | 85 | 11/11 |
| Cursor native transcript family | 49 | 4 | 171 | 11/11 |
| OpenCode | 71 | 2 | 107 | 11/11 |
| Codex CLI | 16 | 2 | 17 | 11/11 |
| Pi | 26 | 1 | 40 | 11/11 |
| DeepSeek Harness | 133 | 3 | 140 | 11/11 |

Every capture passed 36 view schemas, 12 figure renders, JSON export, full-raw
round trips and standalone HTML generation. [results.json](results.json) keeps
the counts, source hashes and coverage warnings. Cursor's supplementary stream
has 26 steps and 76 tools; child summaries are present, with detailed activity
available in its transcript family. Cursor token totals are estimates.

Claude's parent completed, while its reviewer log contains an interruption;
that marker remains visible as an incomplete-capture warning. OpenCode passed
acceptance before its follow-up was stopped; the unfinished response is warned.
The remaining four parent runs completed normally.

The visual walkthrough also caught a contradicted assumed context limit and
tool clocks being subtracted from measured model intervals. Unknown context
limits now have null percentages, and model intervals remain intact.
