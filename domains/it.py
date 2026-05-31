"""IT domain spec (sensitive technical / security / access content)."""
import re
from domains._base import DomainSpec

POLICY = """\
POLICY — Information Technology (sensitive technical, security, and access content).

A text TRIGGERS this policy if it contains or discloses ANY of:
  - credentials or secrets: API keys, passwords, tokens, connection strings
  - access-control grants or changes (who can access what)
  - a security policy or security control statement
  - infrastructure configuration that contains sensitive data
  - vulnerability or security-incident details
  - handling of PII (personal identifiable information)

A text DOES NOT trigger if it is:
  - a routine helpdesk ticket (e.g. "my monitor is broken")
  - a public tutorial or code snippet that contains NO real secrets (placeholders are fine)
  - a bare mention of a tool or vendor name
  - general device/support chatter (e.g. "my laptop is slow")"""

FEWSHOT = [
    ("Here's the deploy bot's GitHub token: ghp_9fK2mNqRsT4vWxYz7aBcDeFgHiJkLmNoPq01", 1),
    ("Set GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (placeholder — fill your own)", 0),
    ("Done — granted Priya admin/sudo on the prod cluster and added her to the secrets group.", 1),
    ("Priya is asking whether she could get admin on prod — should we approve? No change made yet.", 0),
    ("INCIDENT: we were breached overnight, attacker exfiltrated the prod customer DB. IR is live.", 1),
    ("Reuters: Globex disclosed a data breach this week affecting 2M users. Glad it wasn't us.", 0),
    ("Customer record: Jane Doe, DOB 1987-03-12, SSN 412-55-9087, 14 Elm St, card 4242 4242 4242 4242.", 1),
    ("thanks again John, really appreciate the help — talk Tuesday!", 0),
    ("api key (base64, decode before use): QUtJQVRFU1RLRVkxMjM0NTY3ODkw", 1),
    ("anyone else's VPN flaking out today? third time this week ugh", 0),
]

POS_KIND = {
    "secret_credential": "a real credential or secret",
    "access_control": "an enacted access grant or change",
    "security_policy": "an internal security policy or control",
    "infra_config": "infrastructure configuration with sensitive values",
    "vuln_incident": "security-incident or vulnerability details affecting us",
    "pii_handling": "sensitive personal data (PII)",
}
POS_EXTRA = {
    "obfuscation": " The secret is only obscured (encoded/spaced), which still counts as disclosure.",
    "intent_only": " It discloses this even without pasting a literal value.",
    "casual": " Although written casually, it still discloses this.",
}
NEG_WHY = {
    "counterfactual": ("it only references the topic without a real disclosure — e.g. a "
                       "placeholder/example value, a request not yet enacted, third-party news, "
                       "or an invalidated credential"),
    "near_boundary": ("it is general security news/commentary, public boilerplate, a vendor "
                      "claim, or an abstract question — not our sensitive content"),
    "casual": "it is casual chatter or a public tip that discloses nothing sensitive",
    "core": "it is routine, non-sensitive content",
}
NEG_WHY_SUBCAT = {
    "pii_handling": ("it only mentions low-sensitivity identifiers (a name or email) in ordinary "
                     "text, not sensitive PII like SSNs or financial data"),
}

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
    "password_kv": re.compile(
        r"\bpass(?:word|wd|phrase)?['\"]?\s*[:=]\s*['\"]?"
        r"(?=[^\s'\"]*[0-9])(?=[^\s'\"]*[!@#$%^&*\]\[)(\-_=+/])[^\s'\"]{7,}", re.I),
}
PLACEHOLDER_HINTS = re.compile(
    r"(your[_-]?(api[_-]?)?key|xxxx+|<[^>]+>|placeholder|\*\*\*|YOUR_|HOST:5432/DB|USER:PASSWORD)", re.I)

SPEC = DomainSpec(
    name="it", policy=POLICY, fewshot=FEWSHOT,
    pos_kind=POS_KIND, pos_extra=POS_EXTRA, neg_why=NEG_WHY, neg_why_subcat=NEG_WHY_SUBCAT,
    regex_patterns=PATTERNS, placeholder_hints=PLACEHOLDER_HINTS,
    suppress_in_placeholder=("generic_hex_key", "password_kv"),
    hedge="the text sounds IT-related",
    default_pos_kind="sensitive IT content",
    default_neg_why="it does not disclose sensitive IT content",
)
