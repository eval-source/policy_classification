"""
Co-evolution loop controller — Legal pilot. See notes/coevolve_spec.md.

One round: MINE a candidate pool (synthetic CF-heavy + real draw) -> SCORE with the oracle T
(few-shot Qwen3.6-27B) and the student S (zero-shot) -> BUCKET:
  FRONTIER     = T-correct & S-wrong   (hard-but-solvable -> training signal)
  HUMAN_QUEUE  = T-wrong               (both-wrong shared blind spot, or possible mislabel)
  (learned     = T-correct & S-correct -> discarded)

`--dry-run` stops after bucketing + report (NO training), so we can eyeball the mined frontier and
confirm the oracle gate yields sensible, correctly-labeled hard examples before committing the loop.

  python scripts/coevolve.py --round 0 --student Qwen/Qwen3.5-4B --dry-run
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
CO = ROOT / "data" / "legal" / "coevolve"
ANCHOR = ROOT / "data" / "legal" / "anchor"


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def oneline(t, n=140):
    return re.sub(r"\s+", " ", (t or "").strip())[:n]


def load(p):
    p = Path(p)
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def sh(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--student", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--oracle", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--pool-cf", type=int, default=300)
    ap.add_argument("--pool-easy", type=int, default=120)
    ap.add_argument("--pool-real", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true", help="mine+bucket only; no training")
    args = ap.parse_args()
    N = args.round
    CO.mkdir(parents=True, exist_ok=True)
    pool = CO / f"pool_r{N}.jsonl"
    predsT = CO / f"preds_T_r{N}.jsonl"
    predsS = CO / f"preds_S_r{N}.jsonl"
    frontier_f = CO / f"frontier_r{N}.jsonl"
    human_f = CO / f"human_queue_r{N}.jsonl"

    # 1. MINE -------------------------------------------------------------------
    sh([PY, ROOT / "data" / "legal" / "generate.py", "--variant", "v1", "--seed", N,
        "--pool", pool, "--cf-pairs", args.pool_cf, "--easy", args.pool_easy,
        "--intent", 60, "--nearbound", 90, "--casual", 40])
    rows = load(pool)
    real = load(ROOT / "data" / "legal" / "real" / "train.jsonl") + \
        load(ROOT / "data" / "legal" / "real" / "test.jsonl")
    rng = random.Random(N)
    rng.shuffle(real)
    rows += real[:args.pool_real]
    # dedup vs the frozen anchor (no contamination) and within the pool
    anchor_norm = {norm(r["text"]) for r in load(ANCHOR / "gold.jsonl") + load(ANCHOR / "crosssource.jsonl")}
    seen, dd = set(), []
    for r in rows:
        h = norm(r["text"])
        if h and h not in anchor_norm and h not in seen:
            seen.add(h)
            dd.append(r)
    rows = dd
    with pool.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[mine] pool={len(rows)} rows (deduped vs anchor)")

    # 2. SCORE ------------------------------------------------------------------
    sh([PY, ROOT / "scripts" / "label_oracle.py", "--data", pool, "--model", args.oracle,
        "--fewshot", "--domain", "legal", "--out", predsT])
    sh([PY, ROOT / "scripts" / "label_oracle.py", "--data", pool, "--model", args.student,
        "--domain", "legal", "--out", predsS])
    T = {r["id"]: r for r in load(predsT)}
    S = {r["id"]: r for r in load(predsS)}

    # 3. BUCKET -----------------------------------------------------------------
    frontier, human = [], []
    for r in rows:
        t, s = T.get(r["id"]), S.get(r["id"])
        if not t or not s:
            continue
        if t["correct"] and not s["correct"]:
            frontier.append(r)
        elif not t["correct"]:
            human.append(r)
    for path, items in [(frontier_f, frontier), (human_f, human)]:
        with path.open("w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    accT = sum(x["correct"] for x in T.values()) / max(1, len(T))
    accS = sum(x["correct"] for x in S.values()) / max(1, len(S))

    def dist(items, k):
        d = {}
        for r in items:
            d[r.get(k)] = d.get(r.get(k), 0) + 1
        return dict(sorted(d.items(), key=lambda x: -x[1]))

    print(f"\n{'='*64}\nROUND {N} — bucket report\n{'='*64}")
    print(f"  oracle  T  {args.oracle:<22} acc = {accT:.3f}")
    print(f"  student S  {args.student:<22} acc = {accS:.3f}")
    print(f"  GAP  acc(T) - acc(S) = {accT - accS:+.3f}")
    print(f"  FRONTIER (T-correct & S-wrong) : {len(frontier)}")
    print(f"  HUMAN_QUEUE (T-wrong)          : {len(human)}")
    print(f"  learned/discarded              : {len(rows) - len(frontier) - len(human)}")
    print(f"\n  frontier by subcategory: {dist(frontier, 'subcategory')}")
    print(f"  frontier by hardening  : {dist(frontier, 'hardening')}")
    print(f"  frontier by source     : {dist(frontier, 'source')}")
    print("\n  --- sample FRONTIER (gold-label | text) — should be correctly-labeled HARD cases ---")
    for r in frontier[:10]:
        print(f"    [{'TRIGGER' if r['label'] == 1 else 'PASS '}] {oneline(r['text'])}")
    print("\n  --- sample HUMAN_QUEUE (T wrong: possible mislabel OR shared-family blind spot) ---")
    for r in human[:6]:
        print(f"    [{'TRIGGER' if r['label'] == 1 else 'PASS '}] {oneline(r['text'])}")

    if args.dry_run:
        print(f"\n[dry-run] stopped after bucketing. Inspect {frontier_f} and {human_f} before "
              f"committing to the training loop.")
        return


if __name__ == "__main__":
    main()
