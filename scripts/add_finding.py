"""
Append a finding (observation + suggestions) to results/findings.jsonl.

A finding is a qualitative takeaway from an iteration/version — what we learned and what to
do about it. The Findings tab renders these newest-first, grouped by version.

Usage:
  python scripts/add_finding.py --version v1 \
    --title "Counterfactual pairs are the only weak slice" \
    --finding "Model F1 100->95.5; all 6 errors are engineered boundaries (request-vs-grant, news-vs-incident)." \
    --suggestion "Add CoT that reasons 'is this enacted? is this us?'" \
    --suggestion "Push counterfactual volume/variety for a steeper curve" \
    --tags counterfactual,over-trigger
"""
import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="", help="version/iteration this relates to, e.g. v1")
    ap.add_argument("--title", required=True)
    ap.add_argument("--finding", required=True, help="the observation")
    ap.add_argument("--suggestion", action="append", default=[], help="repeatable")
    ap.add_argument("--tags", default="", help="comma-separated")
    args = ap.parse_args()

    fp = ROOT / "results" / "findings.jsonl"
    fp.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(l) for l in fp.open() if l.strip()] if fp.exists() else []
    now = datetime.datetime.now()
    record = dict(
        id=f"f{len(existing)+1:03d}",
        date=now.strftime("%Y-%m-%d %H:%M"),
        version=args.version,
        title=args.title,
        finding=args.finding,
        suggestions=args.suggestion,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
    )
    with fp.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Added {record['id']} ({args.version}): {args.title}")


if __name__ == "__main__":
    main()
