"""Classification prompts. v0 = zero-shot, non-CoT (direct label)."""

IT_POLICY = """\
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


def build_messages(text: str, policy: str = IT_POLICY, cot: bool = False):
    """Return chat messages for the classification task.

    cot=False: model emits the label directly (fast baseline).
    cot=True : model reasons briefly, then emits the label on the last line.
    """
    if cot:
        instruction = (
            "Think step by step in 1-3 short sentences about whether the text discloses "
            "any in-scope content, then on a NEW final line output exactly one word: "
            "TRIGGER or PASS."
        )
    else:
        instruction = "Answer with exactly one word: TRIGGER or PASS."

    user = f"{policy}\n\nTEXT:\n\"\"\"\n{text}\n\"\"\"\n\nDoes the text trigger the policy? {instruction}"
    return [
        {"role": "system", "content": "You are a precise content-policy classifier."},
        {"role": "user", "content": user},
    ]


def parse_label(output: str):
    """Map a model completion to {1,0,None}. None = unparseable."""
    if not output:
        return None
    text = output.strip()
    # For CoT, the decision is on the last non-empty line.
    last = ""
    for line in reversed(text.splitlines()):
        if line.strip():
            last = line.strip()
            break
    candidates = [last.lower(), text.lower()]
    for c in candidates:
        has_trigger = "trigger" in c
        has_pass = "pass" in c
        # guard against "does not trigger" phrasing
        negated = any(p in c for p in ["not trigger", "no trigger", "doesn't trigger", "does not trigger"])
        if has_trigger and not negated:
            return 1
        if has_pass and not has_trigger:
            return 0
        if negated:
            return 0
    return None
