"""The label taxonomy and the conversation used to apply it.

``taxonomy.md`` (bundled) lists the phases (``### phase``) and the actions under
each (``- `action`: description``). This module parses it, builds the system
prompt and the per-step message, and validates a model's answer against it.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The taxonomy shipped with the package.
DEFAULT_TAXONOMY = Path(__file__).with_name("taxonomy.md")

UNKNOWN_LABEL = {"phase": "unknown", "action": "unknown"}


def load_taxonomy(taxonomy_path: str | Path) -> tuple[dict[str, list[str]], str]:
    """Parse a taxonomy file into ``({phase: [action, ...]}, version)``.

    ``version`` comes from a ``(vN)`` marker in the file and is ``"unknown"``
    without one.
    """
    raw_text = Path(taxonomy_path).read_text(encoding="utf-8")

    mapping: dict[str, list[str]] = {}
    current_phase = None
    for line in raw_text.splitlines():
        line = line.strip()
        # Phase header: ### understand, ### plan, etc.
        if line.startswith("### ") and not line.startswith("#### "):
            phase = line[4:].strip().lower()
            if phase and not any(c in phase for c in (" ", "(")):
                current_phase = phase
                mapping.setdefault(current_phase, [])
        # A level-1/2 heading (`# Title`, `## Notes`) CLOSES the current phase.
        # Without this, every later bullet keeps the last `### <phase>`, and
        # `## Notes` is the last heading in taxonomy.md, so a Notes bullet in
        # the file's own `- `name`: ...` style would silently become a valid
        # action of the last phase. `#### ` is deliberately excluded: those
        # sub-subsections refine the phase they sit under.
        elif re.match(r"#{1,2} ", line):
            current_phase = None
        # Action: - `action_name`: description
        elif line.startswith("- `") and current_phase:
            match = re.match(r"- `(\w+)`", line)
            if match:
                mapping[current_phase].append(match.group(1))

    version = "unknown"
    ver_match = re.search(r"\(v(\d+)\)", raw_text)
    if ver_match:
        version = f"v{ver_match.group(1)}"
    return mapping, version


def build_valid_sets(mapping: dict[str, list[str]]) -> tuple[set[str], set[str], dict[str, str]]:
    """``(valid_phases, valid_actions, action_to_phase)`` for a parsed taxonomy."""
    valid_phases = set(mapping)
    valid_actions: set[str] = set()
    action_to_phase: dict[str, str] = {}
    for phase, actions in mapping.items():
        for action in actions:
            valid_actions.add(action)
            action_to_phase[action] = phase
    return valid_phases, valid_actions, action_to_phase


def build_system_prompt(taxonomy_text: str) -> str:
    """The instruction that tells the model how to label one step."""
    return f"""You are a trajectory step classifier. Your task is to label each agent step with a phase tag and an action tag from the taxonomy below.

## Taxonomy

{taxonomy_text}

## Instructions

For each step provided, determine:
1. **phase**: The coarse workflow stage (e.g., understand, plan, implement, debug, validate, report)
2. **action**: The specific behavior within that phase (e.g., code_reading, implement_runtime_logic)

Choose the dominant intent of the step. If a step could fit multiple actions, pick the primary one.

Respond with ONLY a JSON object (no markdown fences, no explanation):
{{"phase": "<phase>", "action": "<action>"}}"""


def build_step_message(step: dict[str, Any], max_chars: int = 8000) -> str:
    """Describe one step for classification: metadata, tool calls, text, reasoning."""
    parts = [f"Step #{step.get('index', '?')}"]
    dur = step.get("duration")
    if dur is not None:
        parts.append(f"Duration: {dur}s")
    tok = step.get("tokens", {})
    if tok.get("total"):
        parts.append(f"Tokens: {tok['total']:,}")
    finish = step.get("finish", "")
    if finish:
        parts.append(f"Finish reason: {finish}")
    agent = step.get("agent", "")
    if agent:
        parts.append(f"Agent: {agent}")
    if step.get("is_sub_agent"):
        parts.append("Context: This is a sub-agent step")

    # Tool calls, with output or error as evidence of what happened.
    tool_calls = step.get("tool_calls", [])
    if tool_calls:
        tool_lines = []
        for tc in tool_calls:
            name = tc.get("tool_name", "?")
            status = tc.get("status", "?")
            inp = tc.get("arguments") or tc.get("input", {})
            if isinstance(inp, dict):
                inp_summary = ", ".join(f"{k}={repr(v)[:80]}" for k, v in list(inp.items())[:5])
            else:
                inp_summary = str(inp)[:200]
            line = f"  - {name} ({status}): {inp_summary}"
            short_result = tc.get("short_result", "")
            output = tc.get("output", "") or tc.get("result", "")
            error = tc.get("error", "")
            if error:
                line += f"\n    ERROR: {str(error)[:200]}"
            elif short_result:
                line += f"\n    Result: {str(short_result)[:200]}"
            elif output:
                out_str = str(output)
                if len(out_str) > 300:
                    out_str = out_str[:300] + "..."
                line += f"\n    Output: {out_str}"
            tool_lines.append(line)
        parts.append("Tool calls:\n" + "\n".join(tool_lines))

    # Sub-agent steps: the executor, its calls and its token use in one block.
    if agent and tool_calls:
        triple_lines = [f"Executor: {agent}"]
        for tc in tool_calls:
            fn = tc.get("tool_name", "?")
            args = tc.get("arguments", {})
            if isinstance(args, dict):
                args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in list(args.items())[:5])
            else:
                args_str = str(args)[:200]
            triple_lines.append(f"ToolCall: {fn}({args_str})")
            sr = tc.get("short_result", "")
            if sr:
                triple_lines.append(f"Result: {sr}")
        tok_total = step.get("tokens", {}).get("total", 0)
        tok_in = step.get("tokens", {}).get("input", 0)
        tok_out = step.get("tokens", {}).get("output", 0)
        if tok_total:
            triple_lines.append(f"Tokens: {tok_total:,} (in={tok_in:,}, out={tok_out:,})")
        parts.append("Step triple:\n" + "\n".join(triple_lines))

    preview = step.get("text_preview", "")
    if preview:
        parts.append(f"Text:\n{preview}")

    reasoning_blocks = [
        p["text"] for p in step.get("parts", []) if p.get("type") == "reasoning" and p.get("text")
    ]
    if reasoning_blocks:
        parts.append("Reasoning:\n" + "\n---\n".join(reasoning_blocks))

    content = "\n".join(parts)
    if len(content) > max_chars:
        # Keep the first 75% and the last 25%, so both the early context and the
        # final outcome (e.g. a tool error at the end) reach the model.
        separator = "\n\n[... truncated ...]\n\n"
        budget = max_chars - len(separator)
        if budget <= 0:
            content = content[:max_chars]
        else:
            head_chars = (budget * 3) // 4
            content = content[:head_chars] + separator + content[-(budget - head_chars):]
    return content


def parse_label_response(
    text: str,
    valid_phases: set[str],
    valid_actions: set[str],
    action_to_phase: dict[str, str],
) -> dict[str, str]:
    """Read the model's answer as a validated ``{"phase", "action"}`` label.

    A valid action fixes its phase; anything outside the taxonomy becomes
    ``unknown`` and is logged.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^}]+\}", text)
        if not match:
            return dict(UNKNOWN_LABEL)
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError:
            return dict(UNKNOWN_LABEL)
    if not isinstance(result, dict):
        return dict(UNKNOWN_LABEL)

    phase = str(result.get("phase") or "unknown").strip().lower()
    action = str(result.get("action") or "unknown").strip().lower()
    if action in valid_actions:
        phase = action_to_phase.get(action, phase)
    elif action != "unknown":
        logger.warning("unknown action %r in model answer", action)
        action = "unknown"
    if phase not in valid_phases and phase != "unknown":
        logger.warning("unknown phase %r in model answer", phase)
        phase = "unknown"
    return {"phase": phase, "action": action}
