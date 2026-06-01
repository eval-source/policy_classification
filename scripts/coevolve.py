"""
Co-evolution loop controller (domain-parameterized). See notes/coevolve_spec.md.

Per round: MINE a candidate pool -> SCORE with oracle T (few-shot Qwen3.6-27B) and the current
student S -> BUCKET (FRONTIER = T-correct & S-wrong; HUMAN_QUEUE = T-wrong) -> accumulate frontier
(leakage-safe by seed) -> TRAIN S on base + accumulated frontier -> MEASURE on the FROZEN anchor
(true progress) + accumulated eval_hard -> log/commit. Loops until convergence or a kill guard.

  python scripts/coevolve.py --domain it --round 0 --student Qwen/Qwen3.5-4B --dry-run
  python scripts/coevolve.py --domain it --start-round 0 --student Qwen/Qwen3.5-4B --max-rounds 4

Honesty note: with no Anthropic key the automated loop uses the QWEN-T gate only. The cross-family
(Claude) rescue of the human_queue — which pushes past the same-family teacher ceiling — stays a
manual step (see the round-0 Legal findings).
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
ORACLE = "Qwen/Qwen3.6-27B"
CONVERGE_YIELD = 30        # frontier smaller than this -> S has matched T on the solvable region
ANCHOR_F1_DROP = 0.02      # kill: anchor F1 falls this far below the best seen
ANCHOR_RECALL_MIN = 0.85   # kill: over-corrected into under-triggering

sys.path.insert(0, str(ROOT))
import domains  # noqa: E402
LAB = {1: "TRIGGER", 0: "PASS"}

# --- per-domain config, set by configure() ---
DOMAIN = None
SPEC = None
CO = None
ANCHOR = None
GEN = None
REAL_DIRS = []
ACCUM_TRAIN = None
ACCUM_EVAL = None


def configure(domain):
    global DOMAIN, SPEC, CO, ANCHOR, GEN, REAL_DIRS, ACCUM_TRAIN, ACCUM_EVAL
    DOMAIN = domain
    SPEC = domains.get(domain)
    CO = ROOT / "data" / domain / "coevolve"
    ANCHOR = ROOT / "data" / domain / "anchor"
    GEN = ROOT / "data" / domain / "generate.py"
    # IT's real corpus lives in data/real; every other domain in data/<domain>/real
    REAL_DIRS = [ROOT / "data" / "real"] if domain == "it" else [ROOT / "data" / domain / "real"]
    ACCUM_TRAIN = CO / "train_hard.jsonl"
    ACCUM_EVAL = CO / "eval_hard.jsonl"
    CO.mkdir(parents=True, exist_ok=True)


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
           "--domain", DOMAIN, "--out", out]
    if fewshot:
        cmd.append("--fewshot")
    sh(cmd)
    return load(out)


def acc(preds):
    return sum(x["correct"] for x in preds) / max(1, len(preds))


def anchor_all():
    p = CO / "anchor_all.jsonl"
    rows = load(ANCHOR / "gold.jsonl") + load(ANCHOR / "crosssource.jsonl")
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def real_rows():
    out = []
    for d in REAL_DIRS:
        out += load(d / "train.jsonl") + load(d / "test.jsonl")
    return out


def mine(N, cfg):
    pool = CO / f"pool_r{N}.jsonl"
    cmd = [PY, GEN, "--variant", "v1", "--seed", N, "--pool", pool,
           "--cf-pairs", cfg["cf"], "--easy", cfg["easy"], "--intent", cfg["intent"],
           "--nearbound", cfg["nearbound"], "--casual", cfg["casual"]]
    if DOMAIN == "it":
        cmd += ["--obf", cfg["obf"]]   # IT-only knob (obfuscation family)
    sh(cmd)
    rows = load(pool)
    real = real_rows()
    rng = random.Random(N); rng.shuffle(real)
    rows += real[:cfg["real"]]
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
    sh([PY, ROOT / "train" / "build_sft_data.py", "--domain", DOMAIN])
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
    sh(["git", "add", "results/coevolve_legal.jsonl", f"results/coevolve_{DOMAIN}.jsonl",
        "results/findings.jsonl", "results/history.jsonl", "results/summary.json"])
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=str(ROOT))


def run_round(N, student, cfg, dry=False):
    print(f"\n{'#'*64}\n# [{DOMAIN}] ROUND {N}  (student = {student[:42]})\n{'#'*64}")
    pool, rows = mine(N, cfg)
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
    ckpt = build_and_train(f"{DOMAIN}_coev_r{N}")
    am = metrics(score(anchor_all(), ckpt, False, CO / f"preds_anchor_r{N}.jsonl"))
    em = metrics(score(ACCUM_EVAL, ckpt, False, CO / f"preds_evalhard_r{N}.jsonl"))
    rec = dict(domain=DOMAIN, round=N, student_in=student[:46], ckpt=ckpt[:46],
               pool=len(rows), oracle_acc=round(accT, 3), student_acc=round(accS, 3),
               gap=round(accT - accS, 3), frontier=len(frontier), frontier_pos=fpos,
               human_queue=len(human), accum_train=len(load(ACCUM_TRAIN)), accum_eval=len(load(ACCUM_EVAL)),
               anchor_f1=am["f1"], anchor_precision=am["precision"], anchor_recall=am["recall"],
               eval_hard_f1=em["f1"], eval_hard_spec=em["specificity"], eval_hard_n=em["n"])
    with open(ROOT / "results" / f"coevolve_{DOMAIN}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[measure] anchor F1={am['f1']} P={am['precision']} R={am['recall']} | "
          f"eval_hard F1={em['f1']} spec={em['specificity']} n={em['n']}")
    sh([PY, ROOT / "scripts" / "add_finding.py", "--version", f"{DOMAIN}-coev-r{N}",
        "--title", f"[{DOMAIN}] co-evolution round {N}: frontier {len(frontier)} ({fpos} pos), "
                   f"anchor F1 {am['f1']} R {am['recall']}, gap {accT-accS:+.3f}",
        "--finding", f"[{DOMAIN}] round {N} mined vs student {student[:40]}. pool={len(rows)}, "
                     f"oracle(27B)acc={accT:.3f}, student_acc={accS:.3f}, gap={accT-accS:+.3f}. "
                     f"frontier={len(frontier)} ({fpos} pos / {len(frontier)-fpos} neg), human_queue={len(human)}. "
                     f"Anchor(frozen) F1={am['f1']} P={am['precision']} R={am['recall']}; "
                     f"eval_hard F1={em['f1']} spec={em['specificity']} n={em['n']}.",
        "--suggestion", "Automated Qwen-T-gated round. Cross-family (Claude) rescue of human_queue stays "
                        "manual (no API key). Watch anchor recall.",
        "--tags", f"{DOMAIN},coevolution,round{N},automated"])
    commit(f"[{DOMAIN}] coevolve round {N}: frontier={len(frontier)} ({fpos} pos), anchor F1={am['f1']} "
           f"R={am['recall']} gap={accT-accS:+.3f} -> {ckpt[:40]}")
    return ckpt, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="legal")
    ap.add_argument("--round", type=int, default=None, help="single round (with --dry-run)")
    ap.add_argument("--start-round", type=int, default=0)
    ap.add_argument("--student", required=True, help="current student ckpt (tinker://...) or base model")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--cf", type=int, default=250)
    ap.add_argument("--easy", type=int, default=100)
    ap.add_argument("--real", type=int, default=120)
    ap.add_argument("--intent", type=int, default=60)
    ap.add_argument("--nearbound", type=int, default=90)
    ap.add_argument("--casual", type=int, default=40)
    ap.add_argument("--obf", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configure(args.domain)
    cfg = dict(cf=args.cf, easy=args.easy, real=args.real, intent=args.intent,
               nearbound=args.nearbound, casual=args.casual, obf=args.obf)

    if args.dry_run:
        run_round(args.round if args.round is not None else args.start_round, args.student, cfg, dry=True)
        return

    student = args.student
    best_anchor = 0.0
    for N in range(args.start_round, args.start_round + args.max_rounds):
        try:
            ckpt, rec = run_round(N, student, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[round {N}] FAILED: {e}")
            break
        if rec["frontier"] < CONVERGE_YIELD:
            print(f"[STOP] converged: frontier {rec['frontier']} < {CONVERGE_YIELD}.")
            break
        if best_anchor and rec["anchor_f1"] < best_anchor - ANCHOR_F1_DROP:
            print(f"[STOP] kill: anchor F1 {rec['anchor_f1']} dropped >{ANCHOR_F1_DROP} below best {best_anchor}.")
            break
        if rec["anchor_recall"] < ANCHOR_RECALL_MIN:
            print(f"[STOP] kill: anchor recall {rec['anchor_recall']} < {ANCHOR_RECALL_MIN}.")
            break
        best_anchor = max(best_anchor, rec["anchor_f1"])
        student = ckpt
    print("\n[loop done]")


if __name__ == "__main__":
    main()
