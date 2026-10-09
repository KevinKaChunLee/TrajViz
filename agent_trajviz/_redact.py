"""Redact credential-shaped text before it leaves the machine.

Trajectories routinely contain credentials an agent printed or typed: a key
in a ``curl`` command, a token in an ``export`` line, a private key it read
from disk. Every LLM request TrajViz sends passes its text through
:func:`redact` first, which replaces such substrings with a placeholder.

It is a safety net, not a guarantee. It recognizes common provider key
formats, PEM private keys, credentials in URLs and authorization headers, and
``NAME=value`` assignments whose name says secret -- and for those it only
takes values that look generated (a letter and a digit, 12+ characters), so
``max_tokens: 4096`` or ``token = response.token`` pass through untouched.
"""

from __future__ import annotations

import re

PLACEHOLDER = "[REDACTED]"

# Whole-match patterns: the match is the secret.
_TOKEN_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bsk-(?:ant-|proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}",  # OpenAI, Anthropic
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",  # AWS access key id
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",  # GitHub
        r"\bgithub_pat_[A-Za-z0-9_]{22,}\b",
        r"\bglpat-[A-Za-z0-9_-]{20,}\b",  # GitLab
        r"\bxox[abposr]-[A-Za-z0-9-]{10,}",  # Slack
        r"\bAIza[0-9A-Za-z_-]{35}\b",  # Google API key
        r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b",  # Stripe secret / restricted
        r"\bhf_[A-Za-z0-9]{30,}\b",  # Hugging Face
        r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",  # JWT
    )
)

_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----|\Z)",
    re.DOTALL,
)

# Prefix-keeping patterns: group 1 is kept, group 2 is the secret. Every
# repeat that precedes a literal is bounded, so a long run of identifier
# characters (minified code, a hex dump) cannot make a pattern quadratic.
_URL_CREDENTIALS = re.compile(r"(?i)\b([a-z][a-z0-9+.-]{0,20}://[^\s/:@]{1,100}:)([^\s/@]{1,200})(?=@)")
_AUTH_HEADER = re.compile(r"(?i)\b(authorization\s{0,5}[:=]\s{0,5}[\"']?(?:bearer|basic|token)\s{1,5})([A-Za-z0-9._~+/=-]{8,})")
_SECRET_NAME = (
    r"[A-Za-z0-9_.-]{0,40}"
    r"(?:api[_-]?key|secret|token|passw(?:or)?d|private[_-]?key|access[_-]?key|credentials?)"
    r"[A-Za-z0-9_.-]{0,40}"
)
_ASSIGNMENT = re.compile(
    rf"(?i)\b({_SECRET_NAME}[\"']?\s{{0,5}}[:=]\s{{0,5}}[\"']?)([A-Za-z0-9_\-./+=~]{{12,}})"
)
# One linear scan decides whether the name-based patterns can match at all.
_SECRET_WORD = re.compile(r"(?i)api[_-]?key|secret|token|passw|private[_-]?key|access[_-]?key|credential")
_CLI_FLAG = re.compile(rf"(?i)(--{_SECRET_NAME}(?:=|\s{{1,5}})[\"']?)([A-Za-z0-9_\-./+=~]{{12,}})")


def _looks_generated(value: str) -> bool:
    return any(c.isdigit() for c in value) and any(c.isalpha() for c in value)


def redact(text: str) -> tuple[str, int]:
    """Return *text* with credential-shaped substrings replaced, and how many."""
    count = 0

    def _whole(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return PLACEHOLDER

    def _keep_prefix(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return match.group(1) + PLACEHOLDER

    def _keep_prefix_if_generated(match: re.Match[str]) -> str:
        # A name that says "secret" is not enough: `token = response.token`.
        if not _looks_generated(match.group(2)):
            return match.group(0)
        return _keep_prefix(match)

    def _private_key(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "[REDACTED PRIVATE KEY]"

    text = _PRIVATE_KEY.sub(_private_key, text)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(_whole, text)
    for pattern in (_URL_CREDENTIALS, _AUTH_HEADER):
        text = pattern.sub(_keep_prefix, text)
    if _SECRET_WORD.search(text):
        for pattern in (_ASSIGNMENT, _CLI_FLAG):
            text = pattern.sub(_keep_prefix_if_generated, text)
    return text, count
