"""
Co-evolution loop controller — Legal pilot. See notes/coevolve_spec.md.

Per round: MINE a candidate pool -> SCORE with oracle T (few-shot Qwen3.6-27B) and the current
student S -> BUCKET (FRONTIER = T-correct & S-wrong; HUMAN_QUEUE = T-wrong) -> accumulate frontier
(leakage-safe by seed) -> TRAIN S on base + accumulated frontier -> MEASURE on the FROZEN anchor
(true progress) + accumulated eval_hard -> log/commit. Loops until convergence or a kill guard.

  python scripts/coevolve.py --round 0 --student Qwen/Qwen3.5-4B --dry-run     # mine+bucket only
  python scripts/coevolve.py --start-round 1 --student tinker://<S1> --max-rounds 3   # automated loop

Honesty note: with no Anthropic key the automated loop uses the QWEN-T gate only. The cross-family
(Claude) rescue of the human_queue — which pushes past the same-family teacher ceiling — stays a
manual step (see findings f for round 0).
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
ORACLE = "Qwen/Qwen3.6-27B"
ACCUM_TRAIN = CO / "train_hard.jsonl"
ACCUM_EVAL = CO / "eval_hard.jsonl"
CONVERGE_YIELD = 30        # frontier smaller than this -> S has matched T on the solvable region
ANCHOR_F1_DROP = 0.02      # kill: anchor F1 falls this far below the best seen
ANCHOR_RECALL_MIN = 0.90   # kill: over-corrected into under-triggering

sys.path.insert(0, str(ROOT))
import domains  # noqa: E402
SPEC = domains.get("legal")
LAB = {1: "TRIGGER", 0: "PASS"}


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def oneline(t, n=130):
    return re.sub(r"\s+", " ", (t or "").strip())[:n]


def load(p):
    p = Path(p)
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def sh(cmd, capture=False):
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        sys.stderr.write((r.stderr or "")[-2000:])
        raise RuntimeError(f"command failed: {cmd[:3]}")
    return r.stdout if capture else ""


# --------------------------- metrics from a preds file ---------------------------

def metrics(preds):
    tp = fp = fn = tn = 0
    for r in preds:
        t, p = r["label"], r["pred"]
        if p is None:
            fn += (t == 1); fp += (t == 0); continue
        tp += (t == 1 and p == 1); fp += (t == 0 and p == 1)
        fn += (t == 1 and p == 0); tn += (t == 0 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else None
    return dict(n=len(preds), pos=tp + fn, precision=round(prec, 3), recall=round(rec, 3),
                f1=round(f1, 3), specificity=(round(spec, 3) if spec is not None else None))


def score(pool, model, fewshot, out):
    cmd = [PY, ROOT / "scripts" / "label_oracle.py", "--data", pool, "--model", model,
           "--domain", "legal", "--out", out]
    if fewshot:
        cmd.append("--fewshot")
    sh(cmd)
    return load(out)


def acc(preds):
    return sum(x["correct"] for x in preds) / max(1, len(preds))


# --------------------------- pieces ---------------------------

def anchor_all():
    p = CO / "anchor_all.jsonl"
    rows = load(ANCHOR / "gold.jsonl") + load(ANCHOR / "crosssource.jsonl")
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def mine(N, cf, easy, real_k):
    pool = CO / f"pool_r{N}.jsonl"
    sh([PY, ROOT / "data" / "legal" / "generate.py", "--variant", "v1", "--seed", N,
        "--pool", pool, "--cf-pairs", cf, "--easy", easy, "--intent", 70, "--nearbound", 90,
        "--casual", 40])
    rows = load(pool)
    real = load(ROOT / "data" / "legal" / "real" / "train.jsonl") + \
        load(ROOT / "data" / "legal" / "real" / "test.jsonl")
    rng = random.Random(N); rng.shuffle(real)
    rows += real[:real_k]
    # dedup vs frozen anchor + already-accumulated hard sets (no contamination / no re-mining)
    block = {norm(r["text"]) for r in (load(ANCHOR / "gold.jsonl") + load(ANCHOR / "crosssource.jsonl")
                                       + load(ACCUM_TRAIN) + load(ACCUM_EVAL))}
    seen, dd = set(), []
    for r in rows:
        h = norm(r["text"])
        if h and h not in block and h not in seen:
            seen.add(h); dd.append(r)
    with pool.open("w") as f:
        for r in dd:
            f.write(json.dumps(r) + "\n")
    return pool, dd


def split_accumulate(frontier, N):
    # leakage-safe split by seed_id; dedup vs the accumulated sets; append
    rng = random.Random(N)
    sids = sorted({r["seed_id"] for r in frontier}); rng.shuffle(sids)
    trs = set(sids[: int(0.7 * len(sids))])
    tr = [r for r in frontier if r["seed_id"] in trs]
    ev = [r for r in frontier if r["seed_id"] not in trs]
    tr_txt = {norm(r["text"]) for r in tr}
    ev = [r for r in ev if norm(r["text"]) not in tr_txt]
    for path, new in [(ACCUM_TRAIN, tr), (ACCUM_EVAL, ev)]:
        have = {norm(r["text"]) for r in load(path)}
        with path.open("a") as f:
            for r in new:
                if norm(r["text"]) not in have:
                    f.write(json.dumps(r) + "\n"); have.add(norm(r["text"]))


def build_and_train(name):
    sh([PY, ROOT / "train" / "build_sft_data.py", "--domain", "legal"])
    with open(ROOT / "train" / "sft_train.jsonl", "a") as f:
        for r in load(ACCUM_TRAIN):
            m = SPEC.build_messages(r["text"], cot=False)
            m.append({"role": "assistant", "content": LAB[r["label"]]})
            f.write(json.dumps({"messages": m}) + "\n")
    out = sh([PY, ROOT / "train" / "sft.py", "--name", name, "--epochs", 2], capture=True)
    m = re.search(r"(tinker://\S+sampler_weights/final)", out)
    if not m:
        raise RuntimeError("could not parse checkpoint from sft.py output")
    return m.group(1)


def commit(msg):
    sh(["git", "add", "results/coevolve_legal.jsonl", "results/findings.jsonl",
        "results/history.jsonl", "results/summary.json"])
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=str(ROOT))


# --------------------------- one round ---------------------------

def run_round(N, student, cfg, dry=False):
    print(f"\n{'#'*64}\n# ROUND {N}  (student = {student[:48]})\n{'#'*64}")
    pool, rows = mine(N, cfg["cf"], cfg["easy"], cfg["real"])
    print(f"[mine] pool={len(rows)}")
    T = {r["id"]: r for r in score(pool, ORACLE, True, CO / f"preds_T_r{N}.jsonl")}
    S = {r["id"]: r for r in score(pool, student, False, CO / f"preds_S_r{N}.jsonl")}
    frontier = [r for r in rows if T.get(r["id"], {}).get("correct") and not S.get(r["id"], {}).get("correct")]
    human = [r for r in rows if not T.get(r["id"], {}).get("correct")]
    accT = acc(list(T.values())); accS = acc(list(S.values()))
    fpos = sum(r["label"] for r in frontier)
    print(f"[bucket] oracle_acc={accT:.3f} student_acc={accS:.3f} gap={accT-accS:+.3f} "
          f"frontier={len(frontier)} ({fpos} pos / {len(frontier)-fpos} neg) human_queue={len(human)}")
    for r in frontier[:6]:
        print(f"    frontier [{LAB[r['label']]:7}] {oneline(r['text'])}")
    with (CO / f"frontier_r{N}.jsonl").open("w") as f:
        for r in frontier:
            f.write(json.dumps(r) + "\n")
    with (CO / f"human_queue_r{N}.jsonl").open("w") as f:
        for r in human:
            f.write(json.dumps(r) + "\n")
    if dry:
        print("[dry-run] stop after bucketing.")
        return None, dict(round=N, frontier=len(frontier), gap=round(accT - accS, 3))

    split_accumulate(frontier, N)
    ckpt = build_and_train(f"legal_coev_r{N}")
    a_all = anchor_all()
    am = metrics(score(a_all, ckpt, False, CO / f"preds_anchor_r{N}.jsonl"))
    em = metrics(score(ACCUM_EVAL, ckpt, False, CO / f"preds_evalhard_r{N}.jsonl"))
    rec = dict(round=N, student_in=student[:46], ckpt=ckpt[:46],
               pool=len(rows), oracle_acc=round(accT, 3), student_acc=round(accS, 3),
               gap=round(accT - accS, 3), frontier=len(frontier), frontier_pos=fpos,
               human_queue=len(human), accum_train=len(load(ACCUM_TRAIN)), accum_eval=len(load(ACCUM_EVAL)),
               anchor_f1=am["f1"], anchor_precision=am["precision"], anchor_recall=am["recall"],
               eval_hard_f1=em["f1"], eval_hard_spec=em["specificity"], eval_hard_n=em["n"])
    with open(ROOT / "results" / "coevolve_legal.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[measure] anchor F1={am['f1']} P={am['precision']} R={am['recall']} | "
          f"eval_hard F1={em['f1']} spec={em['specificity']} n={em['n']}")
    sh([PY, ROOT / "scripts" / "add_finding.py", "--version", f"legal-coev-r{N}",
        "--title", f"Co-evolution round {N}: frontier {len(frontier)} ({fpos} pos), "
                   f"anchor F1 {am['f1']} R {am['recall']}, gap {accT-accS:+.3f}",
        "--finding", f"Round {N} mined vs student {student[:40]}. pool={len(rows)}, oracle(27B)acc={accT:.3f}, "
                     f"student_acc={accS:.3f}, gap={accT-accS:+.3f}. frontier={len(frontier)} ({fpos} pos / "
                     f"{len(frontier)-fpos} neg), human_queue={len(human)}. Trained on base+accum_train "
                     f"({len(load(ACCUM_TRAIN))}). Anchor (frozen referee) F1={am['f1']} P={am['precision']} "
                     f"R={am['recall']}; eval_hard F1={em['f1']} spec={em['specificity']} n={em['n']}.",
        "--suggestion", "Automated Qwen-T-gated round. Cross-family (Claude) rescue of human_queue stays "
                        "manual (no API key). Watch anchor recall (over-trigger-only frontier risk).",
        "--tags", f"legal,coevolution,round{N},automated"])
    commit(f"coevolve legal round {N}: frontier={len(frontier)} ({fpos} pos), anchor F1={am['f1']} "
           f"R={am['recall']} gap={accT-accS:+.3f} -> {ckpt[:40]}")
    return ckpt, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=None, help="single round (with --dry-run)")
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--student", required=True, help="current student ckpt (tinker://...) or base model")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--cf", type=int, default=250)
    ap.add_argument("--easy", type=int, default=100)
    ap.add_argument("--real", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    CO.mkdir(parents=True, exist_ok=True)
    cfg = dict(cf=args.cf, easy=args.easy, real=args.real)

    if args.dry_run:
        run_round(args.round if args.round is not None else args.start_round, args.student, cfg, dry=True)
        return

    # seed the accumulated sets from round 0's artifacts if present and accum is empty
    if not ACCUM_TRAIN.exists() and (CO / "train_hard_r0.jsonl").exists():
        ACCUM_TRAIN.write_text((CO / "train_hard_r0.jsonl").read_text())
        ACCUM_EVAL.write_text((CO / "eval_hard_r0.jsonl").read_text())

    student = args.student
    best_anchor = 0.0
    for N in range(args.start_round, args.start_round + args.max_rounds):
        try:
            ckpt, rec = run_round(N, student, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[round {N}] FAILED: {e}")
            break
        if rec["frontier"] < CONVERGE_YIELD:
            print(f"[STOP] converged: frontier {rec['frontier']} < {CONVERGE_YIELD} "
                  f"(student matches oracle on the solvable region).")
            break
        if best_anchor and rec["anchor_f1"] < best_anchor - ANCHOR_F1_DROP:
            print(f"[STOP] kill: anchor F1 {rec['anchor_f1']} dropped >{ANCHOR_F1_DROP} below best {best_anchor}.")
            break
        if rec["anchor_recall"] < ANCHOR_RECALL_MIN:
            print(f"[STOP] kill: anchor recall {rec['anchor_recall']} < {ANCHOR_RECALL_MIN} (over-corrected).")
            break
        best_anchor = max(best_anchor, rec["anchor_f1"])
        student = ckpt
    print("\n[loop done]")


if __name__ == "__main__":
    main()
