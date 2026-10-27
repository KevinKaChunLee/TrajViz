# Saved Claude OTLP fixture

These are synthetic, sanitized standard OTLP JSON requests, not an exported personal session. They follow the documented Claude Code span/event profile and intentionally put children before parents across batches. Nanosecond precision is retained. The two traces belong to one session.

The capture contains four model requests totaling 57 tokens, three logical tools, 2 seconds of Bash execution plus 2 seconds of permission waiting, and a nested child agent. The API log repeats the first span's usage; its counts must not be added again. The explicit request/tool IDs attach each tool to its originating request.
