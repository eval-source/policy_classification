"""
Domain-agnostic core for policy classification.

A DomainSpec carries everything that differs between policy domains (IT / Legal / Marketing):
the policy text, few-shot examples, CoT-rationale vocabulary, and the regex signal. The shared
prompt-building, label-parsing, CoT-rationale, and regex logic live here so the eval harness,
trainers, and label audit work for any domain via `domains.get(name)`.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

OPENERS = ["Reasoning:", "Let me check:", "Assessing the text:"]


def parse_label(output: str):
    """Map a model completion to {1,0,None}. None = unparseable. (Domain-independent.)"""
    if not output:
        return None
    text = output.strip()
    last = ""
    for line in reversed(text.splitlines()):  # for CoT, decision is the last non-empty line
        if line.strip():
            last = line.strip()
            break
    for c in (last.lower(), text.lower()):
        has_trigger = "trigger" in c
        has_pass = "pass" in c
        negated = any(p in c for p in ["not trigger", "no trigger", "doesn't trigger", "does not trigger"])
        if has_trigger and not negated:
            return 1
        if has_pass and not has_trigger:
            return 0
        if negated:
            return 0
    return None


@dataclass
class DomainSpec:
    name: str
    policy: str                                  # the policy definition (in/out of scope)
    fewshot: list                                # [(text, label), ...] boundary examples
    pos_kind: dict                               # subcategory -> "discloses <X>" phrase (CoT)
    neg_why: dict                                # hardening   -> "but it <...>" phrase (CoT)
    pos_extra: dict = field(default_factory=dict)        # hardening -> extra positive clause
    neg_why_subcat: dict = field(default_factory=dict)   # subcategory -> override neg reason
    regex_patterns: dict = field(default_factory=dict)   # name -> compiled pattern
    placeholder_hints: "re.Pattern | None" = None
    suppress_in_placeholder: tuple = ()                  # pattern names to drop in placeholder blocks
    hedge: str = "the text looks related to the policy"  # CoT negative lead-in
    default_pos_kind: str = "in-scope sensitive content"
    default_neg_why: str = "it does not disclose in-scope content"
    system_prompt: str = "You are a precise content-policy classifier."

    # ---- prompting ----
    def build_messages(self, text: str, cot: bool = False):
        instr = ("Think step by step in 1-3 short sentences about whether the text is in scope, "
                 "then on a NEW final line output exactly one word: TRIGGER or PASS.") if cot \
            else "Answer with exactly one word: TRIGGER or PASS."
        user = f'{self.policy}\n\nTEXT:\n"""\n{text}\n"""\n\nDoes the text trigger the policy? {instr}'
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    def build_messages_fewshot(self, text: str):
        system = (f"{self.system_prompt}\n\n{self.policy}"
                  "\n\nFor each TEXT, answer with exactly one word: TRIGGER or PASS.")
        msgs = [{"role": "system", "content": system}]
        for ex_text, ex_label in self.fewshot:
            msgs.append({"role": "user", "content": f'TEXT:\n"""\n{ex_text}\n"""'})
            msgs.append({"role": "assistant", "content": "TRIGGER" if ex_label == 1 else "PASS"})
        msgs.append({"role": "user", "content": f'TEXT:\n"""\n{text}\n"""'})
        return msgs

    @staticmethod
    def parse_label(output: str):
        return parse_label(output)

    # ---- regex baseline ----
    def regex_predict(self, text: str):
        matches = [n for n, p in self.regex_patterns.items() if p.search(text)]
        if self.placeholder_hints and self.placeholder_hints.search(text):
            matches = [m for m in matches if m not in self.suppress_in_placeholder]
        return (1 if matches else 0), matches

    # ---- CoT rationale (for thinking/CoT SFT data) ----
    def cot_rationale(self, r: dict) -> str:
        opener = OPENERS[hash(r["id"]) % len(OPENERS)]
        if r["label"] == 1:
            kind = self.pos_kind.get(r["subcategory"], self.default_pos_kind)
            extra = self.pos_extra.get(r.get("hardening", "core"), "")
            return f"{opener} the text discloses {kind}.{extra} That is in scope.\nTRIGGER"
        why = self.neg_why_subcat.get(r["subcategory"]) \
            or self.neg_why.get(r.get("hardening", "core"), self.default_neg_why)
        return f"{opener} {self.hedge}, but {why}.\nPASS"
