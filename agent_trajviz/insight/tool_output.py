"""Readable projections of tool results; native payloads remain unchanged."""
from __future__ import annotations

import json
import re


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError):
        return str(value)


def _block_text(text: str) -> tuple[str, bool]:
    """Unwrap a known shell result, rather than interpreting its stdout."""
    try:
        result = json.loads(text)
    except (ValueError, RecursionError):
        result = None
    if (isinstance(result, dict) and "output" in result
            and any(key in result for key in ("chunk_id", "wall_time_seconds", "original_token_count"))):
        output = result["output"]
        body = output if isinstance(output, str) else _json_text(output)
        code = result.get("exit_code")
        if isinstance(code, int) and not isinstance(code, bool):
            return "\n".join(part for part in (body.strip(), f"Process exited with code {code}") if part), code != 0
        # A running result has no final code; keep all its recorded fields.
        return text, False
    failed = bool(re.match(r"(?i)(?:Script failed\b|Error executing code:|failed to spawn code-mode host\b)", text))
    lines = text.splitlines()
    if lines and re.fullmatch(r"Script (?:completed|failed)", lines[0]):
        lines = lines[1:]
        while lines and (not lines[0].strip() or lines[0].startswith("Wall time ") or lines[0] == "Output:"):
            lines.pop(0)
        return "\n".join(lines), failed
    return text, failed


def tool_output_text(output: object, *, errors_only: bool = False) -> str:
    """Show text-block results without their transport representation.

    Code-mode tools can return a list of typed text blocks containing shell
    result envelopes. Summaries prefer failed commands so an earlier successful
    command cannot hide the diagnostic. This projection never changes the
    stored output, status, or exit code.
    """
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    if not isinstance(output, list):
        return _json_text(output)
    segments: list[tuple[str, bool]] = []
    for block in output:
        if (isinstance(block, dict) and block.get("type") in ("input_text", "output_text", "text")
                and isinstance(block.get("text"), str)):
            text, failed = _block_text(block["text"])
            failed |= block.get("isError") is True
            if text.strip():
                segments.append((text, failed))
        else:
            segments.append((_json_text(block), False))
    selected = [text for text, failed in segments if failed] if errors_only else []
    if not selected:
        selected = [text for text, _ in segments]
    return "\n\n".join(selected) if selected else _json_text(output)
