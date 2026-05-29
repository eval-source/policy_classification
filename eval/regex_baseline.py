"""Deterministic regex baseline for IT secrets.

This is the cheap, zero-hallucination signal the brief asks for. It can ONLY catch the
literal-secret subcategories (secret_credential, infra_config). It is blind to access
changes, security policy, vuln/incident, and PII-by-description — so on its own it will
have high precision but low recall across the full IT policy. That contrast is exactly
the point: it shows where you need a model.
"""
import re

# Each pattern is intentionally specific to avoid matching placeholders.
PATTERNS = {
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "aws_secret_key": re.compile(r"\baws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{40}\b", re.I),
    "github_token": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "slack_token": re.compile(r"\bxoxb-\d+-\d+-[A-Za-z0-9]{20,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "conn_string_pw": re.compile(r"\b(?:postgres|postgresql|mysql|redis|mongodb)://[^\s:@/]+:[^\s:@/]+@", re.I),
    "generic_hex_key": re.compile(r"\b[0-9a-f]{32,}\b"),
    "ssn": re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
}

# Strings that signal a placeholder, used to suppress false hits on tutorials.
PLACEHOLDER_HINTS = re.compile(
    r"(your[_-]?(api[_-]?)?key|xxxx+|<[^>]+>|placeholder|\*\*\*|YOUR_|HOST:5432/DB|USER:PASSWORD)",
    re.I,
)


def predict(text: str):
    """Return (label, matched_pattern_names)."""
    matches = [name for name, pat in PATTERNS.items() if pat.search(text)]
    # A long hex string inside an obvious placeholder block is suppressed.
    if matches == ["generic_hex_key"] and PLACEHOLDER_HINTS.search(text):
        matches = []
    return (1 if matches else 0), matches
