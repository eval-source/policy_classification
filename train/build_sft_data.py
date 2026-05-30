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


def load(split):
    rows = []
    for sub in ("it", "real"):
        fp = ROOT / "data" / sub / f"{split}.jsonl"
        if fp.exists():
            rows += [json.loads(l) for l in fp.open() if l.strip()]
    return rows


def to_conversation(r, cot=False):
    msgs = build_messages(r["text"], cot=cot)
    msgs.append({"role": "assistant", "content": LABEL[r["label"]]})
    return {"messages": msgs}


def main():
    out = ROOT / "train"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, name in [("train", "sft_train"), ("val", "sft_val")]:
        rows = load(split)
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(to_conversation(r)) + "\n")
        pos = sum(r["label"] for r in rows)
        bysrc = {}
        for r in rows:
            bysrc[r.get("source", "synthetic")] = bysrc.get(r.get("source", "synthetic"), 0) + 1
        counts[name] = (len(rows), pos, len(rows) - pos, bysrc)
    for name, (n, p, ng, bysrc) in counts.items():
        print(f"{name}: {n} ({p} pos / {ng} neg) by_source={bysrc}")


if __name__ == "__main__":
    main()
