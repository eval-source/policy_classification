"""
Build SFT conversation data from synthetic + real TRAIN/VAL splits.

Each example reuses the exact eval prompt (eval/prompts.build_messages) and appends the gold
label as the assistant turn — so we train on precisely the format we score. Output is the
cookbook's conversations JSONL: {"messages": [system, user, assistant]}.

Train mixes synthetic (data/it) + real (data/real, already confidence-gated by fetch_real).
Noise augmentation is already baked into the synthetic rows (generate.py --noise-rate).

Run:  python train/build_sft_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from prompts import build_messages  # noqa: E402

LABEL = {1: "TRIGGER", 0: "PASS"}

# What kind of in-scope content a positive discloses (by subcategory).
POS_KIND = {
    "secret_credential": "a real credential or secret",
    "access_control": "an enacted access grant or change",
    "security_policy": "an internal security policy or control",
    "infra_config": "infrastructure configuration with sensitive values",
    "vuln_incident": "security-incident or vulnerability details affecting us",
    "pii_handling": "sensitive personal data (PII)",
}
# Hardening-specific nuance appended to a positive's rationale.
POS_EXTRA = {
    "obfuscation": " The secret is only obscured (encoded/spaced), which still counts as disclosure.",
    "intent_only": " It discloses this even without pasting a literal value.",
    "casual": " Although written casually, it still discloses this.",
}
# Why a negative does NOT trigger (by hardening).
NEG_WHY = {
    "counterfactual": ("it only references the topic without a real disclosure — e.g. a "
                       "placeholder/example value, a request not yet enacted, third-party news, "
                       "or an invalidated credential"),
    "near_boundary": ("it is general security news/commentary, public boilerplate, a vendor "
                      "claim, or an abstract question — not our sensitive content"),
    "casual": "it is casual chatter or a public tip that discloses nothing sensitive",
    "core": "it is routine, non-sensitive content",
}
OPENERS = ["Reasoning:", "Let me check:", "Assessing the text:"]


def cot_rationale(r):
    """Short reasoning (1-2 sentences) ending in the label, derived from known structure."""
    opener = OPENERS[hash(r["id"]) % len(OPENERS)]
    if r["label"] == 1:
        kind = POS_KIND.get(r["subcategory"], "sensitive IT content")
        extra = POS_EXTRA.get(r.get("hardening", "core"), "")
        return f"{opener} the text discloses {kind}.{extra} That is in scope.\nTRIGGER"
    why = NEG_WHY.get(r.get("hardening", "core"), "it does not disclose sensitive IT content")
    # the real remaining gap: low-sensitivity PII in ordinary prose
    if r["subcategory"] == "pii_handling":
        why = ("it only mentions low-sensitivity identifiers (a name or email) in ordinary "
               "text, not sensitive PII like SSNs or financial data")
    return f"{opener} the text sounds IT-related, but {why}.\nPASS"


def load(split):
    rows = []
    for sub in ("it", "real"):
        fp = ROOT / "data" / sub / f"{split}.jsonl"
        if fp.exists():
            rows += [json.loads(l) for l in fp.open() if l.strip()]
    return rows


def to_conversation(r, cot=False):
    msgs = build_messages(r["text"], cot=cot)
    content = cot_rationale(r) if cot else LABEL[r["label"]]
    msgs.append({"role": "assistant", "content": content})
    return {"messages": msgs}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cot", action="store_true", help="emit reasoning + label (CoT) targets")
    args = ap.parse_args()
    suffix = "_cot" if args.cot else ""
    out = ROOT / "train"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, base in [("train", "sft_train"), ("val", "sft_val")]:
        name = base + suffix
        rows = load(split)
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(to_conversation(r, cot=args.cot)) + "\n")
        pos = sum(r["label"] for r in rows)
        bysrc = {}
        for r in rows:
            bysrc[r.get("source", "synthetic")] = bysrc.get(r.get("source", "synthetic"), 0) + 1
        counts[name] = (len(rows), pos, len(rows) - pos, bysrc)
    for name, (n, p, ng, bysrc) in counts.items():
        print(f"{name}: {n} ({p} pos / {ng} neg) by_source={bysrc}")


if __name__ == "__main__":
    main()
