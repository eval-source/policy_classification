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


# Hand-authored few-shot examples covering the boundaries teachers fail (esp. counterfactuals).
# Not drawn from the test set. Used to lift a zero-shot teacher's discrimination in-context.
FEWSHOT_EXAMPLES = [
    ("Here's the deploy bot's GitHub token: ghp_9fK2mNqRsT4vWxYz7aBcDeFgHiJkLmNoPq01", 1),  # real secret
    ("Set GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (placeholder — fill your own)", 0),  # placeholder
    ("Done — granted Priya admin/sudo on the prod cluster and added her to the secrets group.", 1),  # enacted grant
    ("Priya is asking whether she could get admin on prod — should we approve? No change made yet.", 0),  # request, not enacted
    ("INCIDENT: we were breached overnight, attacker exfiltrated the prod customer DB. IR is live.", 1),  # our incident
    ("Reuters: Globex disclosed a data breach this week affecting 2M users. Glad it wasn't us.", 0),  # third-party news
    ("Customer record: Jane Doe, DOB 1987-03-12, SSN 412-55-9087, 14 Elm St, card 4242 4242 4242 4242.", 1),  # PII record
    ("thanks again John, really appreciate the help — talk Tuesday!", 0),  # casual name mention
    ("api key (base64, decode before use): QUtJQVRFU1RLRVkxMjM0NTY3ODkw", 1),  # obfuscated secret
    ("anyone else's VPN flaking out today? third time this week ugh", 0),  # support chatter
]


def build_messages_fewshot(text: str, policy: str = IT_POLICY):
    """Few-shot prompt: policy in system, then labeled example turns, then the query."""
    system = (
        "You are a precise content-policy classifier.\n\n" + policy
        + "\n\nFor each TEXT, answer with exactly one word: TRIGGER or PASS."
    )
    msgs = [{"role": "system", "content": system}]
    for ex_text, ex_label in FEWSHOT_EXAMPLES:
        msgs.append({"role": "user", "content": f"TEXT:\n\"\"\"\n{ex_text}\n\"\"\""})
        msgs.append({"role": "assistant", "content": "TRIGGER" if ex_label == 1 else "PASS"})
    msgs.append({"role": "user", "content": f"TEXT:\n\"\"\"\n{text}\n\"\"\""})
    return msgs


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
