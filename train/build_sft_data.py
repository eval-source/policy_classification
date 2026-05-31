"""
Build SFT conversation data from synthetic + real TRAIN/VAL splits, for any domain.

Each example reuses the exact eval prompt (the domain spec's build_messages) and appends the
gold label (or a CoT rationale) as the assistant turn — so we train on precisely the format we
score. Output is the cookbook's conversations JSONL: {"messages": [system, user, assistant]}.

Reads synthetic (data/<domain>) + real (data/real) splits. Noise augmentation is already baked
into the synthetic rows (generate.py --noise-rate).

Run:  python train/build_sft_data.py --domain it [--cot]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import domains  # noqa: E402

LABEL = {1: "TRIGGER", 0: "PASS"}


def load(split, domain):
    rows = []
    for sub in (domain, "real"):
        fp = ROOT / "data" / sub / f"{split}.jsonl"
        if fp.exists():
            rows += [json.loads(l) for l in fp.open() if l.strip()]
    return rows


def to_conversation(r, spec, cot=False):
    msgs = spec.build_messages(r["text"], cot=cot)
    content = spec.cot_rationale(r) if cot else LABEL[r["label"]]
    msgs.append({"role": "assistant", "content": content})
    return {"messages": msgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="it")
    ap.add_argument("--cot", action="store_true", help="emit reasoning + label (CoT) targets")
    args = ap.parse_args()
    spec = domains.get(args.domain)
    suffix = "_cot" if args.cot else ""
    out = ROOT / "train"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, base in [("train", "sft_train"), ("val", "sft_val")]:
        name = base + suffix
        rows = load(split, args.domain)
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(to_conversation(r, spec, cot=args.cot)) + "\n")
        pos = sum(r["label"] for r in rows)
        bysrc = {}
        for r in rows:
            bysrc[r.get("source", "synthetic")] = bysrc.get(r.get("source", "synthetic"), 0) + 1
        counts[name] = (len(rows), pos, len(rows) - pos, bysrc)
    for name, (n, p, ng, bysrc) in counts.items():
        print(f"{name}: {n} ({p} pos / {ng} neg) by_source={bysrc}")


if __name__ == "__main__":
    main()
