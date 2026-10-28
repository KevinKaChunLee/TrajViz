"""Shared helpers for LLM HTTP requests (env JSON headers, auth merge)."""

from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)


def _clean_env_scalar(raw: str) -> str:
    value = raw.strip()
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def env_json_dict(name: str) -> dict[str, str]:
    """Parse a JSON object env var into string headers (empty on missing/invalid)."""
    raw = os.getenv(name)
    if raw is None:
        return {}
    value = _clean_env_scalar(raw)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except ValueError:
        logger.warning("invalid JSON in %s; ignoring", name)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("%s must be a JSON object; ignoring", name)
        return {}
    return {
        str(key): str(val)
        for key, val in parsed.items()
        if isinstance(val, (str, int, float, bool))
    }


def merge_llm_headers(
    extra: dict[str, str] | None = None,
    *,
    auth: dict[str, str],
) -> dict[str, str]:
    """Content-Type, optional gateway extras, then *auth* (auth wins on conflict)."""
    headers = {"Content-Type": "application/json"}
    if extra:
        headers.update(extra)
    headers.update(auth)
    return headers
