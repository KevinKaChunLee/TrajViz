# Codex 0.164 rollout shape

Minimal synthetic records follow the native rollout structure observed in an isolated coding task and its reviewer on Codex CLI 0.164.0. IDs, paths, text, counts and timestamps are sanitized examples. The child owns the first session header, carries an inherited parent header, and declares `subagent_history_start_ordinal`.

Expected consumption is 13 parent tokens plus 9 child tokens, with one spawn and one child test call. Copied parent records and the parallel `token_usage_record` representation must not add usage or tools again. Model identity comes from each thread's `turn_context`.
