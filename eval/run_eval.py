"""
Sliceable eval harness for policy classification.

Runs one or both of:
  - model : frozen Qwen3.5-4B (or a Tinker checkpoint) zero-shot, via Tinker sampling
  - regex : deterministic secret-pattern baseline

Reports precision / recall / F1 / accuracy overall and per slice (subcategory, difficulty,
source, format), plus a confusion matrix and a bootstrap 95% CI on F1. Writes per-example
predictions and a summary to results/.

Usage:
  python eval/run_eval.py --data data/it/test.jsonl --mode both
  python eval/run_eval.py --data data/it/test.jsonl --mode model --cot --max 40
  python eval/run_eval.py --data data/it/test.jsonl --mode model --model tinker://<ckpt>
"""
import argparse
import datetime
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: F401  (loads API key)
import random

from prompts import build_messages, parse_label
import regex_baseline

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


# --------------------------- model client ---------------------------

def make_sampler(model: str):
    import tinker
    svc = tinker.ServiceClient()
    if model.startswith("tinker://"):
        sc = svc.create_sampling_client(model_path=model)
    else:
        sc = svc.create_sampling_client(base_model=model)
    return sc, sc.get_tokenizer()


def encode_chat(tok, messages, cot: bool):
    kwargs = dict(add_generation_prompt=True, tokenize=True)
    # hybrid Qwen3.5: disable thinking unless we explicitly want CoT
    try:
        ids = tok.apply_chat_template(messages, enable_thinking=cot, **kwargs)
    except TypeError:
        ids = tok.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def run_model(rows, model, cot, temperature, max_tokens, concurrency):
    import tinker
    from tinker import types

    sc, tok = make_sampler(model)
    params = types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature, stop=["<|im_end|>"]
    )

    def one(row):
        ids = encode_chat(tok, build_messages(row["text"], cot=cot), cot=cot)
        prompt = types.ModelInput.from_ints(ids)
        for attempt in range(3):
            try:
                resp = sc.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
                comp = resp.sequences[0].tokens
                out = tok.decode(comp)
                return out, parse_label(out), len(ids), len(comp)
            except Exception as e:  # transient API errors -> retry
                if attempt == 2:
                    return f"<error: {e}>", None, len(ids), 0
        return "", None, len(ids), 0

    preds, raws = [None] * len(rows), [None] * len(rows)
    prompt_tokens = completion_tokens = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, (out, pred, p_tok, c_tok) in enumerate(ex.map(one, rows)):
            raws[i], preds[i] = out, pred
            prompt_tokens += p_tok
            completion_tokens += c_tok
    usage = dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return preds, raws, usage


def load_prices(model):
    """Per-million-token prices for `model` from config/tinker_prices.json."""
    fp = Path(__file__).resolve().parent.parent / "config" / "tinker_prices.json"
    prices = json.loads(fp.read_text()) if fp.exists() else {}
    p = prices.get(model) or prices.get("default") or {"prefill": 0.0, "sample": 0.0}
    return p, bool(prices.get("_confirmed", False))


def estimate_cost(usage, price):
    """USD estimate: prompt tokens at prefill rate + completion at sample rate."""
    return (usage["prompt_tokens"] * price["prefill"]
            + usage["completion_tokens"] * price["sample"]) / 1e6


def run_regex(rows):
    preds, raws = [], []
    for r in rows:
        label, matches = regex_baseline.predict(r["text"])
        preds.append(label)
        raws.append(",".join(matches))
    return preds, raws


# --------------------------- metrics ---------------------------

def confusion(y_true, y_pred):
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if p is None:
            # unparsed -> count as the wrong class (conservative: a miss either way)
            if t == 1:
                fn += 1
            else:
                fp += 1
            continue
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def prf(tp, fp, fn, tn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return prec, rec, f1, acc


def bootstrap_f1_ci(y_true, y_pred, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(y_true)
    if n == 0:
        return (0.0, 0.0)
    idx = list(range(n))
    f1s = []
    for _ in range(n_boot):
        sample = [rng.choice(idx) for _ in range(n)]
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        _, _, f1, _ = prf(*confusion(yt, yp))
        f1s.append(f1)
    f1s.sort()
    lo = f1s[int(0.025 * n_boot)]
    hi = f1s[int(0.975 * n_boot)]
    return (lo, hi)


def slice_report(rows, y_true, y_pred, key):
    groups = defaultdict(lambda: ([], []))
    for r, t, p in zip(rows, y_true, y_pred):
        groups[r[key]][0].append(t)
        groups[r[key]][1].append(p)
    out = {}
    for g, (yt, yp) in sorted(groups.items()):
        tp, fp, fn, tn = confusion(yt, yp)
        prec, rec, f1, acc = prf(tp, fp, fn, tn)
        n_pos = sum(yt)
        # specificity (TNR) is the meaningful number for negative-heavy slices,
        # where precision/recall/F1 are degenerate (e.g. all-negative near_boundary).
        spec = tn / (tn + fp) if (tn + fp) else None
        out[g] = dict(n=len(yt), n_pos=n_pos, n_neg=len(yt) - n_pos,
                      precision=prec, recall=rec, f1=f1, accuracy=acc,
                      specificity=spec, fp=fp, fn=fn)
    return out


def fmt_pct(x):
    return f"{100*x:5.1f}"


def print_report(name, rows, y_true, y_pred, raws):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    prec, rec, f1, acc = prf(tp, fp, fn, tn)
    lo, hi = bootstrap_f1_ci(y_true, y_pred)
    n_unparsed = sum(1 for p in y_pred if p is None)

    print(f"\n{'='*64}\n{name}  (n={len(rows)})\n{'='*64}")
    print(f"  Precision {fmt_pct(prec)}  Recall {fmt_pct(rec)}  F1 {fmt_pct(f1)}  Acc {fmt_pct(acc)}")
    print(f"  F1 95% CI: [{fmt_pct(lo)}, {fmt_pct(hi)}]")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}   unparsed={n_unparsed}")

    for key in ("subcategory", "hardening", "difficulty", "format", "source"):
        if key not in rows[0]:
            continue
        print(f"\n  -- by {key} --")
        rep = slice_report(rows, y_true, y_pred, key)
        print(f"    {'slice':<22} {'n':>4} {'pos':>4} {'P':>6} {'R':>6} {'F1':>6} {'Spec':>6}")
        for g, m in rep.items():
            spec = fmt_pct(m["specificity"]) if m["specificity"] is not None else "    -"
            print(f"    {str(g):<22} {m['n']:>4} {m['n_pos']:>4} "
                  f"{fmt_pct(m['precision']):>6} {fmt_pct(m['recall']):>6} {fmt_pct(m['f1']):>6} {spec:>6}")

    return dict(
        name=name, n=len(rows), precision=prec, recall=rec, f1=f1, accuracy=acc,
        f1_ci=[lo, hi], tp=tp, fp=fp, fn=fn, tn=tn, unparsed=n_unparsed,
        by_subcategory=slice_report(rows, y_true, y_pred, "subcategory"),
        by_difficulty=slice_report(rows, y_true, y_pred, "difficulty"),
        by_hardening=(slice_report(rows, y_true, y_pred, "hardening")
                      if "hardening" in rows[0] else {}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/it/test.jsonl")
    ap.add_argument("--mode", choices=["model", "regex", "both"], default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max", type=int, default=0, help="limit number of examples (0=all)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--note", default="", help="human label for this iteration")
    args = ap.parse_args()

    # --data accepts one or more comma-separated files (rows keep their own `source` field,
    # so a combined run yields a synthetic-vs-real slice).
    rows = []
    for path in args.data.split(","):
        path = path.strip()
        rows.extend(json.loads(l) for l in open(path) if l.strip())
    if args.max:
        rows = rows[: args.max]
    y_true = [r["label"] for r in rows]

    # CoT needs room to reason; direct label needs very little.
    if not args.cot and args.max_tokens > 16:
        args.max_tokens = 16

    Path(args.out).mkdir(parents=True, exist_ok=True)
    iteration, run_id, now, config = make_run_meta(args, len(rows))
    summaries = []

    if args.mode in ("regex", "both"):
        preds, raws = run_regex(rows)
        s = print_report("REGEX baseline", rows, y_true, preds, raws)
        s["usage"] = dict(prompt_tokens=0, completion_tokens=0, est_cost_usd=0.0)
        s["preds_file"] = _dump(args.out, "regex", run_id, rows, y_true, preds, raws)
        summaries.append(s)

    if args.mode in ("model", "both"):
        tag = f"model{'_cot' if args.cot else ''}"
        preds, raws, usage = run_model(
            rows, args.model, args.cot, args.temperature, args.max_tokens, args.concurrency
        )
        price, confirmed = load_prices(args.model)
        cost = estimate_cost(usage, price)
        usage = dict(usage, est_cost_usd=round(cost, 6), price_basis=price, price_confirmed=confirmed)
        label = f"MODEL {args.model}{' +CoT' if args.cot else ''}"
        s = print_report(label, rows, y_true, preds, raws)
        s["usage"] = usage
        s["preds_file"] = _dump(args.out, tag, run_id, rows, y_true, preds, raws)
        summaries.append(s)
        tok_total = usage["prompt_tokens"] + usage["completion_tokens"]
        print(f"  Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion "
              f"= {tok_total}  |  est. cost ${cost:.4f}{'' if confirmed else ' (PLACEHOLDER price)'}")

    with open(Path(args.out) / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote summary -> {args.out}/summary.json")

    append_history(args, iteration, run_id, now, config, summaries)


def make_run_meta(args, n_rows):
    hist_path = Path(args.out) / "history.jsonl"
    prev = [json.loads(l) for l in hist_path.open() if l.strip()] if hist_path.exists() else []
    iteration = len(prev) + 1
    now = datetime.datetime.now()
    config = dict(
        model=args.model, mode=args.mode, cot=args.cot,
        temperature=args.temperature, max_tokens=args.max_tokens,
        data=args.data, n=n_rows, concurrency=args.concurrency,
    )
    run_id = "it{:03d}-{}-{}".format(
        iteration, now.strftime("%Y%m%d_%H%M%S"),
        hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6],
    )
    return iteration, run_id, now, config


def append_history(args, iteration, run_id, now, config, summaries):
    """Append this invocation as one iteration to results/history.jsonl."""
    hist_path = Path(args.out) / "history.jsonl"
    record = dict(
        iteration=iteration, run_id=run_id,
        timestamp=now.isoformat(timespec="seconds"),
        date=now.strftime("%Y-%m-%d %H:%M"),
        note=args.note, config=config, systems=summaries,
    )
    with hist_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Logged iteration #{iteration} ({run_id}) -> {hist_path}")
    print("View live at viz/index.html (run `python viz/serve.py` if not already serving)")


def _dump(out, tag, run_id, rows, y_true, y_pred, raws):
    """Write per-run predictions (with correctness + source/slice fields) and return the
    repo-relative path the dashboard can fetch. Also refresh the 'latest' preds_<tag> file."""
    def records():
        for r, t, p, raw in zip(rows, y_true, y_pred, raws):
            yield {
                "id": r["id"], "label": t, "pred": p,
                "correct": (p == t),
                "raw": raw, "source": r.get("source", "synthetic"),
                "subcategory": r["subcategory"], "hardening": r.get("hardening", "core"),
                "difficulty": r["difficulty"], "text": r["text"][:300],
            }
    # latest (overwritten) — handy for quick inspection
    with open(Path(out) / f"preds_{tag}.jsonl", "w") as f:
        for rec in records():
            f.write(json.dumps(rec) + "\n")
    # per-run (immutable, keyed by run_id) — powers the Results success/failure subtabs
    runs_dir = Path(out) / "preds_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rel = f"{out}/preds_runs/{run_id}__{tag}.jsonl"
    with open(Path(out) / "preds_runs" / f"{run_id}__{tag}.jsonl", "w") as f:
        for rec in records():
            f.write(json.dumps(rec) + "\n")
    return rel


if __name__ == "__main__":
    main()
