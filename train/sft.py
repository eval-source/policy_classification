"""
SFT (LoRA) on Qwen3.5-4B for policy classification, via the Tinker cookbook.

Trains on train/sft_train.jsonl (synthetic + real, label-as-assistant-turn). Eval is done
separately through our harness against the held-out test sets, so we keep test_size=0 here.

  python train/sft.py --smoke              # 2 steps, validate the pipeline end-to-end
  python train/sft.py --name sft_v1 --epochs 2
After it finishes it prints the sampler checkpoint path; eval with:
  python eval/run_eval.py --data data/it/test.jsonl,data/real/test_realistic.jsonl \
      --model tinker://<ckpt> --note "SFT v1"
"""
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _env  # noqa: F401  (maps THINKING_MACHINE_API_KEY -> TINKER_API_KEY)

from tinker_cookbook import model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

MODEL = "Qwen/Qwen3.5-4B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sft_v1")
    ap.add_argument("--data", default=str(ROOT / "train" / "sft_train.jsonl"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 steps to validate the pipeline")
    args = ap.parse_args()

    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    log_path = ROOT / "train" / "runs" / (args.name + ("_smoke" if args.smoke else ""))
    if log_path.exists():
        shutil.rmtree(log_path)

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,  # train on the label tokens
    )
    dataset = FromConversationFileBuilder(common_config=common, file_path=args.data, test_size=0)

    config = train.Config(
        log_path=str(log_path),
        model_name=MODEL,
        renderer_name=renderer_name,
        dataset_builder=dataset,
        learning_rate=args.lr,
        lr_schedule="linear",
        num_epochs=1 if args.smoke else args.epochs,
        lora_rank=args.lora_rank,
        save_every=0,
        eval_every=0,
        max_steps=2 if args.smoke else args.max_steps,
    )
    print(f"SFT {args.name}: model={MODEL} renderer={renderer_name} lr={args.lr} "
          f"rank={args.lora_rank} bs={args.batch_size} epochs={config.num_epochs} "
          f"max_steps={config.max_steps}\nlog_path={log_path}")
    asyncio.run(train.main(config))

    # report the final checkpoint path for eval
    ckpts = log_path / "checkpoints.jsonl"
    if ckpts.exists():
        last = [json.loads(l) for l in ckpts.open() if l.strip()][-1]
        print("\nCHECKPOINTS:", json.dumps(last))
        print("Sampler path candidates:", {k: v for k, v in last.items() if "path" in k.lower()})


if __name__ == "__main__":
    main()
