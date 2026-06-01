"""
Score a candidate pool with a model and write per-example preds (id, label, pred, correct, ...).

Used by the co-evolution loop (notes/coevolve_spec.md): the ORACLE (few-shot Qwen3.6-27B) and the
STUDENT (Qwen3.5-4B, zero-shot) are each scored with this, then joined by `id` to find the FRONTIER
(T-correct & S-wrong). Unlike eval/run_eval.py this does NOT append to results/history.jsonl — these
are internal scoring passes, not logged experiment iterations.

  python scripts/label_oracle.py --data pool.jsonl --model Qwen/Qwen3.6-27B --fewshot --out preds_T.jsonl
  python scripts/label_oracle.py --data pool.jsonl --model Qwen/Qwen3.5-4B           --out preds_S.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))
import _env  # noqa: F401  (loads the Tinker API key)
import domains  # noqa: E402
from run_eval import run_model  # noqa: E402  (reuse the exact sampling/encode/parse path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--domain", default="legal")
    ap.add_argument("--fewshot", action="store_true", help="prepend in-context labeled examples (oracle)")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = domains.get(args.domain)
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    max_tokens = args.max_tokens if args.cot else 16
    preds, raws, usage = run_model(
        rows, spec, args.model, args.cot, args.temperature, max_tokens, args.concurrency,
        fewshot=args.fewshot,
    )

    n_corr = 0
    with open(args.out, "w") as f:
        for r, p, raw in zip(rows, preds, raws):
            corr = (p == r["label"])
            n_corr += int(corr)
            f.write(json.dumps(dict(
                id=r["id"], label=r["label"], pred=p, correct=corr,
                seed_id=r.get("seed_id"), subcategory=r.get("subcategory"),
                hardening=r.get("hardening"), source=r.get("source", "synthetic"),
                text=r["text"],
            )) + "\n")
    acc = n_corr / len(rows) if rows else 0.0
    print(f"{args.model} fewshot={args.fewshot}: acc={acc:.3f} ({n_corr}/{len(rows)}) "
          f"tokens={usage['prompt_tokens']}+{usage['completion_tokens']} -> {args.out}")


if __name__ == "__main__":
    main()
