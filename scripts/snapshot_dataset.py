"""
Register a named DATASET VERSION (ds-vX) = the composition of the data on disk RIGHT NOW
(synthetic data/it + real data/real), with a content fingerprint. Pure dataset — no eval.

This is deliberately separate from experiment ITERATIONS (results/history.jsonl). An iteration
is (model/config) run against one dataset version; the iteration records `dataset_version`.
That separation is what makes ablation clean: hold the dataset fixed and vary the model
(training gains), or hold the model fixed and vary the dataset (benchmark difficulty).

Run:  python scripts/snapshot_dataset.py --version ds-v4 --note "synthetic v4 (casual+noise) + real"
"""
import argparse
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DSPATH = ROOT / "results" / "datasets.jsonl"


def load_rows(include_real=True):
    dirs = [ROOT / "data" / "it"]
    if include_real:
        dirs.append(ROOT / "data" / "real")
    splits, rows = {}, []
    for d in dirs:
        for sp in ("train", "val", "test"):
            fp = d / f"{sp}.jsonl"
            if fp.exists():
                items = [json.loads(l) for l in fp.open() if l.strip()]
                splits[sp] = splits.get(sp, 0) + len(items)
                rows += items
    return rows, splits


def stats_for(rows, splits):
    pos = sum(1 for r in rows if r["label"] == 1)
    return dict(
        total=len(rows), pos=pos, neg=len(rows) - pos, splits=splits,
        by_source=dict(Counter(r.get("source", "synthetic") for r in rows)),
        by_subcategory=dict(Counter(r["subcategory"] for r in rows)),
        by_hardening=dict(Counter(r.get("hardening", "core") for r in rows)),
        by_difficulty=dict(Counter(r["difficulty"] for r in rows)),
    )


def fingerprint(rows):
    # order-independent content hash over (id|label|text)
    h = hashlib.sha1()
    for s in sorted(f"{r['id']}|{r['label']}|{r['text']}" for r in rows):
        h.update(s.encode())
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="dataset version label, e.g. ds-v4")
    ap.add_argument("--note", default="")
    ap.add_argument("--no-real", action="store_true", help="synthetic only")
    args = ap.parse_args()

    rows, splits = load_rows(include_real=not args.no_real)
    rec = dict(
        version=args.version,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        note=args.note,
        fingerprint=fingerprint(rows),
        stats=stats_for(rows, splits),
    )
    DSPATH.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(l) for l in DSPATH.open() if l.strip()] if DSPATH.exists() else []
    existing = [e for e in existing if e["version"] != args.version]
    existing.append(rec)
    existing.sort(key=lambda e: e["date"])
    with DSPATH.open("w") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")
    print(f"Registered {args.version}: {rec['stats']['total']} rows "
          f"(fp {rec['fingerprint']}, by_source {rec['stats']['by_source']}) -> {DSPATH}")


if __name__ == "__main__":
    main()
