"""
Thinking SFT (STaR / rejection-sampling) for Qwen3.5-4B (hybrid, native <think>).

f015 showed TEMPLATED prompted-CoT hurts. This is the principled alternative: let the model
generate its OWN native <think> reasoning, KEEP only traces that reach the correct label, and
SFT on those — on the EXACT tokens (generation-prompt + sampled completion), so train and
inference align (avoids the hybrid-renderer mask trap where loss lands on the prompt's <think>).

Phase 1 (generate): sample thinking rollouts from frozen Qwen3.5-4B, filter correct -> cache.
Phase 2 (train):    SFT on the cached correct traces (LoRA), save sampler checkpoint.

  python train/sft_think.py --n 700 --epochs 2
Then eval (thinking on):
  python eval/run_eval.py --data data/it/test.jsonl,data/real/test_realistic.jsonl \
      --model tinker://<ckpt> --cot --dataset-version ds-v5 --note "SFT-think"
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))
import _env  # noqa: F401
from prompts import build_messages, parse_label  # noqa: E402

import torch
import tinker
from tinker import types
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.tokenizer_utils import get_tokenizer

STUDENT = "Qwen/Qwen3.5-4B"
CACHE = ROOT / "train" / "think_traces.jsonl"


def prompt_ids_for(tok, text):
    ids = tok.apply_chat_template(build_messages(text, cot=False),
                                  add_generation_prompt=True, enable_thinking=True, tokenize=True)
    return list(ids["input_ids"]) if hasattr(ids, "input_ids") else list(ids)


def parse_after_think(decoded):
    ans = decoded.split("</think>")[-1] if "</think>" in decoded else decoded
    return parse_label(ans)


def load_train_rows(n, seed):
    rows = []
    for sub in ("it", "real"):
        fp = ROOT / "data" / sub / "train.jsonl"
        if fp.exists():
            rows += [json.loads(l) for l in fp.open() if l.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n]


async def generate(n, seed, temperature, max_tokens, concurrency):
    svc = tinker.ServiceClient()
    sc = await svc.create_sampling_client_async(base_model=STUDENT)
    tok = get_tokenizer(STUDENT)
    imend = tok.encode("<|im_end|>", add_special_tokens=False)
    rows = load_train_rows(n, seed)
    params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=["<|im_end|>"])
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        async with sem:
            pids = prompt_ids_for(tok, r["text"])
            try:
                resp = await sc.sample_async(prompt=types.ModelInput.from_ints(pids),
                                             sampling_params=params, num_samples=1)
            except Exception:
                return None
            comp = list(resp.sequences[0].tokens)
            decoded = tok.decode(comp)
            pred = parse_after_think(decoded)
            if pred is None or pred != r["label"]:
                return None  # reject: wrong or unparseable
            if not comp or comp[-1] not in imend:
                comp = comp + imend  # ensure it learns to stop
            return {"prompt_ids": pids, "completion_ids": comp, "gold": r["label"],
                    "subcategory": r["subcategory"], "source": r.get("source", "synthetic")}

    results = await asyncio.gather(*[one(r) for r in rows])
    kept = [x for x in results if x]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w") as f:
        for x in kept:
            f.write(json.dumps(x) + "\n")
    pos = sum(1 for x in kept if x["gold"] == 1)
    print(f"generated {len(rows)} -> kept {len(kept)} correct traces ({pos} pos / {len(kept)-pos} neg)")
    from collections import Counter
    print("kept by subcategory:", dict(Counter(x["subcategory"] for x in kept)))
    return kept


async def train(traces, epochs, lr, lora_rank, batch_size, log_path):
    svc = tinker.ServiceClient()
    tc = await svc.create_lora_training_client_async(base_model=STUDENT, rank=lora_rank)
    datums = []
    for t in traces:
        ids = t["prompt_ids"] + t["completion_ids"]
        w = torch.tensor([0.0] * len(t["prompt_ids"]) + [1.0] * len(t["completion_ids"]))
        datums.append(datum_from_model_input_weights(types.ModelInput.from_ints(ids), w, max_length=2048))
    order = list(range(len(datums)))
    for ep in range(epochs):
        random.Random(ep).shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = [datums[j] for j in order[i:i + batch_size]]
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            await tc.optim_step_async(tinker.AdamParams(learning_rate=lr))
        print(f"epoch {ep+1}/{epochs} done ({len(datums)} examples)")
    Path(log_path).mkdir(parents=True, exist_ok=True)
    paths = await checkpoint_utils.save_checkpoint_async(
        training_client=tc, name="final", log_path=str(log_path), loop_state={"epochs": epochs}, kind="both")
    print("CHECKPOINTS:", json.dumps(paths))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gen-max-tokens", type=int, default=220)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--regen", action="store_true", help="force regenerate traces")
    args = ap.parse_args()

    if args.regen or not CACHE.exists():
        traces = asyncio.run(generate(args.n, args.seed, args.temperature, args.gen_max_tokens, args.concurrency))
    else:
        traces = [json.loads(l) for l in CACHE.open() if l.strip()]
        print(f"using cached {len(traces)} traces ({CACHE})")
    if not traces:
        print("no traces; aborting"); return
    asyncio.run(train(traces, args.epochs, args.lr, args.lora_rank, args.batch_size,
                       ROOT / "train" / "runs" / "sft_think"))


if __name__ == "__main__":
    main()
