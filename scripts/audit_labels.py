"""
Label audit: cross-check dataset labels against independent signals and surface the
disagreements for human adjudication (the brief's human + LLM-judge + regex stack).

Signals per row:
  - label : the dataset's stored label (what we're auditing)
  - regex : deterministic literal-secret detector (high precision, narrow)
  - judge : a strong LLM judge's prediction (passed in via --judge-preds; we use the
            few-shot Qwen3.6-27B — competent F1~92, but SAME family so a weaker signal)

A disagreement is a CANDIDATE label error, not a confirmed one (the judge is imperfect). The
tool quantifies agreement (incl. Cohen's kappa), buckets disagreements by source/subcategory,
and writes a review queue. A human / independent judge (Claude) then adjudicates the queue.

Run:  python scripts/audit_labels.py \
        --data data/it/test.jsonl,data/real/test_realistic.jsonl \
        --judge-preds results/preds_runs/<judge-run>__model.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import regex_baseline  # noqa: E402


def cohen_kappa(pairs):
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    # marginals
    pa1 = sum(1 for a, _ in pairs if a == 1) / n
    pb1 = sum(1 for _, b in pairs if b == 1) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/it/test.jsonl,data/real/test_realistic.jsonl")
    ap.add_argument("--judge-preds", required=True)
    ap.add_argument("--out", default="results/label_audit.jsonl")
    args = ap.parse_args()

    rows = {}
    for p in args.data.split(","):
        for l in open(p.strip()):
            if l.strip():
                r = json.loads(l); rows[r["id"]] = r
    judge = {}
    for l in open(args.judge_preds):
        if l.strip():
            d = json.loads(l); judge[d["id"]] = d.get("pred")

    audited, queue = [], []
    lj_pairs = []
    for rid, r in rows.items():
        if rid not in judge or judge[rid] is None:
            continue
        lab = r["label"]; jpred = judge[rid]
        rpred = regex_baseline.predict(r["text"])[0]
        lj_pairs.append((lab, jpred))
        rec = dict(id=rid, source=r.get("source", "synthetic"),
                   subcategory=r["subcategory"], hardening=r.get("hardening", "core"),
                   label=lab, judge=jpred, regex=rpred, text=r["text"][:280])
        audited.append(rec)
        # candidate label error: judge disagrees, OR regex fires but label says PASS
        if jpred != lab or (rpred == 1 and lab == 0):
            queue.append(rec)

    n = len(audited)
    agree = sum(1 for x in audited if x["judge"] == x["label"])
    kappa = cohen_kappa(lj_pairs)
    print(f"Audited {n} rows (judge available).")
    print(f"label–judge agreement: {100*agree/n:.1f}%   Cohen's kappa: {kappa:.3f}")
    print(f"candidate label errors (judge!=label or regex/label conflict): {len(queue)} ({100*len(queue)/n:.1f}%)")

    # direction of judge disagreements
    jt_lp = sum(1 for x in audited if x["judge"] == 1 and x["label"] == 0)  # judge TRIGGER, label PASS
    jp_lt = sum(1 for x in audited if x["judge"] == 0 and x["label"] == 1)  # judge PASS, label TRIGGER
    print(f"  judge=TRIGGER/label=PASS (candidate missed-positive): {jt_lp}")
    print(f"  judge=PASS/label=TRIGGER (candidate over-labeled):    {jp_lt}")

    def bucket(key):
        g = defaultdict(lambda: [0, 0])  # [n, disagreements]
        for x in audited:
            g[x[key]][0] += 1
            if x["judge"] != x["label"]:
                g[x[key]][1] += 1
        print(f"\n  -- disagreement by {key} --")
        print(f"    {'slice':<22} {'n':>4} {'disagree':>9} {'rate':>6}")
        for k, (tot, dis) in sorted(g.items(), key=lambda kv: -kv[1][1]):
            if dis:
                print(f"    {str(k):<22} {tot:>4} {dis:>9} {100*dis/tot:>5.0f}%")

    bucket("source")
    bucket("subcategory")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for x in queue:
            f.write(json.dumps(x) + "\n")
    print(f"\nWrote {len(queue)} candidates to {args.out} (for human/independent adjudication)")


if __name__ == "__main__":
    main()
