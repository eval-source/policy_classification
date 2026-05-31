"""
Experiment supervisor — the sensing + decision brain for the autonomous self-improve loop.

Reads the logs (history/datasets/findings) + the on-disk datasets, evaluates per-domain stage
success criteria, runs a billing precheck, and prints the recommended NEXT ACTION (with the exact
command). The /loop agent (experiment-supervisor skill) executes the recommendation, verifies,
updates results/plan.jsonl, logs a finding, and commits to the yolo branch.

Stages per domain (in order):  dataset -> discriminative -> real_data -> sft
Success criteria are computable from the logs so decisions are grounded, not re-derived.

  python scripts/supervisor.py --status          # human table + JSON next-action
  python scripts/supervisor.py --billing-check    # 0 if Tinker reachable, 2 if 402-blocked
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import domains  # noqa: E402

HIST = ROOT / "results" / "history.jsonl"
DSV = ROOT / "results" / "datasets.jsonl"

# discriminative target: frozen F1 should sit in this band (hard enough to measure deltas,
# not so hard it's broken). Above HI => saturated (harden). Below LO => maybe over-hardened.
DISC_LO, DISC_HI = 0.80, 0.94
SFT_MARGIN = 0.03  # an SFT model must beat the frozen baseline by this on the same data


def load_jsonl(p):
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def model_sys(it):
    for s in it.get("systems", []):
        if "regex" not in s["name"].lower():
            return s
    return None


def domain_iters(hist, domain):
    return [it for it in hist if it.get("config", {}).get("domain") == domain]


def billing_ok():
    try:
        import _env  # noqa: F401
        import tinker
        tinker.ServiceClient().create_sampling_client(base_model="Qwen/Qwen3.5-4B")
        return True, "ok"
    except Exception as e:
        msg = str(e)
        if "402" in msg or "billing" in msg.lower():
            return False, "BILLING BLOCKED (402)"
        return False, f"unreachable: {msg[:80]}"


def assess(domain, hist):
    """Return (status_dict, next_action|None)."""
    iters = domain_iters(hist, domain)
    has_data = (ROOT / "data" / domain / "test.jsonl").exists()
    frozen = [it for it in iters if not it["config"]["model"].startswith("tinker://")
              and not it["config"].get("cot")]
    trained = [it for it in iters if it["config"]["model"].startswith("tinker://")]
    fz = model_sys(frozen[-1]) if frozen else None
    tr = model_sys(trained[-1]) if trained else None
    st = {"domain": domain, "has_data": has_data,
          "frozen_f1": round(fz["f1"], 3) if fz else None,
          "trained_f1": round(tr["f1"], 3) if tr else None,
          "n_iters": len(iters)}

    # stage 1: dataset
    if not has_data:
        return st, {"stage": "dataset", "why": "no test split", "action": "generate synthetic v1",
                    "cmd": f"python data/{domain}/generate.py --variant v1 --seed 0 --out data/{domain}"}
    # stage 2: discriminative (frozen baseline in band)
    if fz is None:
        return st, {"stage": "discriminative", "why": "no frozen baseline yet", "action": "eval frozen",
                    "cmd": f"python eval/run_eval.py --data data/{domain}/test.jsonl --mode both "
                           f"--model Qwen/Qwen3.5-4B --domain {domain} --dataset-version {domain}-v1 "
                           f"--note 'frozen baseline'"}
    if fz["f1"] > DISC_HI:
        st["verdict"] = "SATURATED"
        return st, {"stage": "discriminative", "why": f"frozen F1 {fz['f1']:.2f} > {DISC_HI} (saturated)",
                    "action": "HARDEN: add subtler counterfactuals/near-boundary to the generator, regenerate, re-eval frozen",
                    "cmd": f"(worker) edit data/{domain}/generate.py CF/near_boundary, then "
                           f"python data/{domain}/generate.py --variant v1 --seed 0 --out data/{domain} "
                           f"&& re-run frozen eval"}
    # stage 3: real data (real rows present in an eval — they carry hardening="real", or the
    # eval combined a data/real file)
    has_real = any(("real" in (model_sys(it) or {}).get("by_hardening", {}))
                   or ("real" in it["config"].get("data", "")) for it in iters)
    real_fetcher = (ROOT / "data" / domain / "fetch_real.py").exists() or domain == "it"
    if not has_real:
        return st, {"stage": "real_data", "why": "no source=real slice measured yet",
                    "action": ("build a real HF fetcher (worker) then eval on synthetic+real"
                               if not real_fetcher else "fetch real + combined eval"),
                    "cmd": f"(worker) create data/{domain}/fetch_real.py, then fetch + "
                           f"run_eval --data ...test + real --domain {domain}"}
    # stage 4: SFT must beat frozen by margin
    if tr is None:
        return st, {"stage": "sft", "why": "no trained model yet", "action": "build SFT data + train + eval",
                    "cmd": f"python train/build_sft_data.py --domain {domain} && python train/sft.py "
                           f"--name {domain}_sft --epochs 2  # then eval --model tinker://<ckpt> --domain {domain}"}
    if tr["f1"] < fz["f1"] + SFT_MARGIN:
        st["verdict"] = "SFT_NOT_WINNING"
        return st, {"stage": "sft", "why": f"trained F1 {tr['f1']:.2f} not > frozen {fz['f1']:.2f}+{SFT_MARGIN}",
                    "action": "iterate SFT (more epochs / data / check labels)", "cmd": "(worker) tune SFT"}
    st["verdict"] = "COMPLETE (SFT beats frozen; benchmark discriminative)"
    return st, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--billing-check", action="store_true")
    args = ap.parse_args()

    if args.billing_check:
        ok, msg = billing_ok()
        print(msg)
        sys.exit(0 if ok else 2)

    hist = load_jsonl(HIST)
    doms = domains.available()
    print(f"{'domain':<10} {'data':>5} {'frozen':>7} {'trained':>8} {'iters':>5}  next")
    next_actions = []
    for d in doms:
        st, nxt = assess(d, hist)
        verdict = st.get("verdict", "")
        nstr = "— done" if nxt is None else f"[{nxt['stage']}] {nxt['action'][:48]}"
        print(f"{d:<10} {str(st['has_data']):>5} {str(st['frozen_f1']):>7} {str(st['trained_f1']):>8} "
              f"{st['n_iters']:>5}  {nstr}")
        if nxt:
            next_actions.append({"domain": d, **nxt})
    ok, bmsg = billing_ok()
    print(f"\nbilling: {bmsg}")
    print("\nNEXT_ACTIONS_JSON: " + json.dumps(next_actions))


if __name__ == "__main__":
    main()
