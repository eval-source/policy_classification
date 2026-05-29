"""
Snapshot a labeled VERSION = (dataset stats + a chosen eval iteration) -> results/versions.jsonl.

A version is a checkpoint of the benchmark+model state you want to compare over time
(e.g. v0 easy baseline, v1 after hard negatives, v2 after SFT). The Versions tab reads this
file and renders each version plus the diff/improvement vs. the previous one.

Usage:
  python scripts/snapshot_version.py --version v0 --note "easy baseline" --domain it
  python scripts/snapshot_version.py --version v1 --note "added hard negs" --iteration 3
"""
import argparse
import datetime
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def dataset_stats(domain, include_real=True):
    """Counts over BOTH synthetic (data/<domain>) and real (data/real) when present,
    with a by_source breakdown so the Versions tab reflects the full dataset."""
    dirs = [("synthetic", ROOT / "data" / domain)]
    if include_real:
        dirs.append(("real", ROOT / "data" / "real"))
    splits, rows = {}, []
    for _src, d in dirs:
        for sp in ("train", "val", "test"):
            fp = d / f"{sp}.jsonl"
            if not fp.exists():
                continue
            items = [json.loads(l) for l in fp.open() if l.strip()]
            splits[sp] = splits.get(sp, 0) + len(items)
            rows += items
    pos = sum(1 for r in rows if r["label"] == 1)
    return dict(
        domain=domain,
        total=len(rows),
        pos=pos,
        neg=len(rows) - pos,
        splits=splits,
        by_source=dict(Counter(r.get("source", "synthetic") for r in rows)),
        by_subcategory=dict(Counter(r["subcategory"] for r in rows)),
        by_difficulty=dict(Counter(r["difficulty"] for r in rows)),
        by_format=dict(Counter(r["format"] for r in rows)),
    )


def pick_iteration(iteration):
    hp = ROOT / "results" / "history.jsonl"
    if not hp.exists():
        return None
    hist = [json.loads(l) for l in hp.open() if l.strip()]
    if not hist:
        return None
    if iteration in (None, "latest"):
        rec = hist[-1]
    else:
        rec = next((h for h in hist if h["iteration"] == int(iteration)), hist[-1])
    # keep only headline fields per system (the tab computes diffs from these)
    systems = [
        dict(
            name=s["name"], f1=s["f1"], precision=s["precision"], recall=s["recall"],
            accuracy=s["accuracy"], f1_ci=s.get("f1_ci"),
            by_subcategory={k: v["f1"] for k, v in s.get("by_subcategory", {}).items()},
            by_difficulty={k: v["f1"] for k, v in s.get("by_difficulty", {}).items()},
            by_hardening={k: v["f1"] for k, v in s.get("by_hardening", {}).items()},
            est_cost_usd=s.get("usage", {}).get("est_cost_usd", 0.0),
        )
        for s in rec.get("systems", [])
    ]
    total_cost = sum(s["est_cost_usd"] for s in systems)
    return dict(iteration=rec["iteration"], run_id=rec["run_id"], config=rec["config"],
                systems=systems, est_cost_usd=round(total_cost, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="version label, e.g. v0")
    ap.add_argument("--note", default="", help="what changed / hypothesis")
    ap.add_argument("--domain", default="it")
    ap.add_argument("--iteration", default="latest", help="eval iteration id or 'latest'")
    args = ap.parse_args()

    now = datetime.datetime.now()
    record = dict(
        version=args.version,
        date=now.strftime("%Y-%m-%d %H:%M"),
        note=args.note,
        dataset=dataset_stats(args.domain),
        eval=pick_iteration(args.iteration),
    )

    vp = ROOT / "results" / "versions.jsonl"
    vp.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(l) for l in vp.open() if l.strip()] if vp.exists() else []
    # replace if same version label already exists, else append
    existing = [e for e in existing if e["version"] != args.version]
    existing.append(record)
    existing.sort(key=lambda e: e["date"])
    with vp.open("w") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")

    mf1 = next((s["f1"] for s in (record["eval"]["systems"] if record["eval"] else []) if "MODEL" in s["name"]), None)
    print(f"Snapshotted {args.version}: {record['dataset']['total']} rows, "
          f"model F1={'%.3f' % mf1 if mf1 is not None else 'n/a'} -> {vp}")


if __name__ == "__main__":
    main()
