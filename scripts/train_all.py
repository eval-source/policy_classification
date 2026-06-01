"""
Train SFT and SFT+OPD on the base model for all three domains, and eval each stage
(frozen baseline / SFT / SFT+OPD) on that domain's synthetic+real test. Every eval auto-logs to
results/history.jsonl, so the runs appear in the dashboard (Results tab, per-domain sub-tabs).

  python scripts/train_all.py --domains it,legal,marketing --steps 24
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
TEACHER = "Qwen/Qwen3.6-27B"   # few-shot teacher (zero-shot 27B is worse than the specialized 4B)
BASE = "Qwen/Qwen3.5-4B"
# synthetic test + real positive-rare test, per domain (gives a source slice; comparable across domains)
EVAL_DATA = {
    "it": "data/it/test.jsonl,data/real/test_realistic.jsonl",
    "legal": "data/legal/test.jsonl,data/legal/real/test_realistic.jsonl",
    "marketing": "data/marketing/test.jsonl,data/marketing/real/test_realistic.jsonl",
}


def sh(cmd, capture=False):
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        sys.stderr.write((r.stderr or "")[-3000:])
        raise RuntimeError(f"failed: {cmd[:3]}")
    return r.stdout if capture else ""


def sft_ckpts(out):
    m = re.search(r"CHECKPOINTS:\s*(\{.*\})", out)
    d = json.loads(m.group(1))
    return d["state_path"], d["sampler_path"]


def opd_sampler(out):
    m = re.search(r"(tinker://\S+sampler_weights/final)", out)
    if not m:
        raise RuntimeError("no OPD checkpoint parsed")
    return m.group(1)


def evl(dom, model, note):
    sh([PY, ROOT / "eval" / "run_eval.py", "--data", EVAL_DATA[dom], "--mode", "model",
        "--model", model, "--domain", dom, "--dataset-version", f"{dom}-sftopd", "--note", note])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="it,legal,marketing")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    results = {}
    for dom in args.domains.split(","):
        print(f"\n{'='*64}\n===== {dom} =====\n{'='*64}", flush=True)
        try:
            sh([PY, ROOT / "train" / "build_sft_data.py", "--domain", dom])
            sft_out = sh([PY, ROOT / "train" / "sft.py", "--name", f"{dom}_sft", "--epochs", 2], capture=True)
            sft_state, sft_sampler = sft_ckpts(sft_out)
            evl(dom, BASE, "frozen baseline (sft/opd batch)")
            evl(dom, sft_sampler, "SFT")
            opd_out = sh([PY, ROOT / "train" / "opd.py", "--name", f"{dom}_opd", "--domain", dom,
                          "--student-ckpt", sft_state, "--teacher", TEACHER, "--teacher-fewshot",
                          "--steps", args.steps], capture=True)
            opd_s = opd_sampler(opd_out)
            evl(dom, opd_s, "SFT+OPD")
            results[dom] = dict(sft=sft_sampler, opd=opd_s)
            print(f"[{dom}] DONE  SFT={sft_sampler}  OPD={opd_s}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{dom}] FAILED: {e}", flush=True)
            results[dom] = dict(error=str(e))
    print("\nRESULTS_JSON: " + json.dumps(results))


if __name__ == "__main__":
    main()
