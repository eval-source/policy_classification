"""
On-policy distillation (OPD) for policy classification, via the Tinker cookbook.

Student = our SFT checkpoint (Qwen3.5-4B). Teacher = a larger Qwen (default Qwen3.5-27B).
The student samples CoT rollouts on our prompts; the teacher provides per-token reverse-KL
supervision on those on-policy rollouts (no reward — pure distillation).

NOTE (finding f017): the larger Qwen teachers tested ZERO-SHOT are WORSE than the SFT-4B on
our hard slices, so we expect OPD to regress the student here — this run is to use the loop
for real and measure that, per the brief ("decide, with evidence, when OPD is/ isn't worth it").

  python train/opd.py --smoke
  python train/opd.py --name opd_v1 --teacher Qwen/Qwen3.5-27B --student-ckpt tinker://<state> --steps 8
"""
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))
import _env  # noqa: F401
from prompts import build_messages  # noqa: E402

import chz
from tinker_cookbook import model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig, PromptOnlyDataset, PromptOnlyDatasetBuilder, TeacherConfig,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers

STUDENT = "Qwen/Qwen3.5-4B"


@chz.chz
class PolicyPromptBuilder(PromptOnlyDatasetBuilder):
    """Prompt-only dataset from OUR classification prompts (CoT) instead of deepmath/tulu3."""
    prompts: list[str] = chz.field(default_factory=list)
    system: str = "You are a precise content-policy classifier."

    async def __call__(self):
        tok = get_tokenizer(self.model_name_for_tokenizer)
        rend = renderers.get_renderer(self.renderer_name, tokenizer=tok)
        ds = PromptOnlyDataset(
            prompts=self.prompts, batch_size=self.groups_per_batch, group_size=self.group_size,
            renderer=rend, tokenizer=tok, max_prompt_tokens=self.max_prompt_tokens,
            convo_prefix=[{"role": "system", "content": self.system}], dataset_name="policy_it",
        )
        return ds, None


def load_prompts(n):
    rows = []
    for sub in ("it", "real"):
        fp = ROOT / "data" / sub / "train.jsonl"
        if fp.exists():
            rows += [json.loads(l) for l in fp.open() if l.strip()]
    # user-content of the CoT prompt (system is supplied via convo_prefix)
    prompts = [build_messages(r["text"], cot=True)[1]["content"] for r in rows]
    return prompts[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="opd_v1")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--student-ckpt", default=None, help="tinker:// state checkpoint to init from")
    ap.add_argument("--groups-per-batch", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    renderer_name = model_info.get_recommended_renderer_name(STUDENT)
    gpb = 8 if args.smoke else args.groups_per_batch
    n_prompts = gpb * (2 if args.smoke else args.steps)
    prompts = load_prompts(n_prompts)
    log_path = ROOT / "train" / "runs" / (args.name + ("_smoke" if args.smoke else ""))
    if log_path.exists():
        shutil.rmtree(log_path)

    builder = PolicyPromptBuilder(
        dataset_name="policy_it", groups_per_batch=gpb, group_size=args.group_size,
        model_name_for_tokenizer=STUDENT, renderer_name=renderer_name, prompts=prompts,
    )
    dataset_config = DistillationDatasetConfig(
        dataset_builder=builder,
        teacher_config=TeacherConfig(base_model=args.teacher),
        groups_per_batch=gpb,
    )
    config = train_on_policy.Config(
        learning_rate=args.lr,
        dataset_configs=[dataset_config],
        model_name=STUDENT,
        renderer_name=renderer_name,
        lora_rank=args.lora_rank,
        max_tokens=args.max_tokens,
        kl_penalty_coef=args.kl_coef,
        log_path=str(log_path),
        load_checkpoint_path=args.student_ckpt,
        eval_every=0,
        save_every=0,
        max_steps=2 if args.smoke else args.steps,
    )
    print(f"OPD {args.name}: student={STUDENT} (init={args.student_ckpt}) teacher={args.teacher} "
          f"gpb={gpb} group_size={args.group_size} steps={config.max_steps} kl={args.kl_coef}\n"
          f"prompts={len(prompts)} log_path={log_path}")
    asyncio.run(train_on_policy.main(config))

    ckpts = log_path / "checkpoints.jsonl"
    if ckpts.exists():
        last = [json.loads(l) for l in ckpts.open() if l.strip()][-1]
        print("\nCHECKPOINTS:", json.dumps(last))


if __name__ == "__main__":
    main()
