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
from typing import cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import _env  # noqa: F401
import domains  # noqa: E402

SPEC = None  # set in main() from --domain

import chz
import torch
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig, PromptOnlyDataset, PromptOnlyDatasetBuilder, TeacherConfig,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook import renderers

STUDENT = "Qwen/Qwen3.5-4B"

# Asymmetric few-shot teacher: the teacher scores the student's (zero-shot) rollouts conditioned
# on a few-shot PREFIX, so few-shot-teacher knowledge distills into a zero-shot student.
_TEACHER_PREFIX: list[int] = []


async def _kl_penalty_fewshot_teacher(data_D, teacher_clients_D, dataset_indices_D,
                                      kl_penalty_coef, kl_discount_factor):
    """incorporate_kl_penalty, but the teacher sees _TEACHER_PREFIX before each sequence.
    We prepend P prefix tokens to the teacher input and shift the logprob slice by 1+P so the
    rollout-token alignment with the student is preserved."""
    TOP = train_on_policy
    P = _TEACHER_PREFIX
    full_sequence_inputs_D = [
        tinker.types.ModelInput.from_ints(
            P + datum.model_input.to_ints()
            + [cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])]
        )
        for datum in data_D
    ]
    teacher_logprobs_D = await asyncio.gather(*[
        tc.compute_logprobs_async(si)
        for tc, si in zip(teacher_clients_D, full_sequence_inputs_D)
    ])
    sampled_logprobs_D = [d.loss_fn_inputs["logprobs"].to_torch() for d in data_D]
    float_masks = [d.loss_fn_inputs["mask"].to_torch().float() for d in data_D]
    off = 1 + len(P)  # original is [1:]; +len(P) drops the prefix logprobs
    reverse_kl = [
        (sl - torch.tensor(tl[off:])) * m
        for tl, sl, m in TOP.safezip(teacher_logprobs_D, sampled_logprobs_D, float_masks)
    ]
    per_dataset_kl: dict = {}
    for i, datum in enumerate(data_D):
        kl_adv = -kl_penalty_coef * float_masks[i] * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_adv = TOP.discounted_future_sum_vectorized(kl_adv, kl_discount_factor)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_adv
        )
        di = dataset_indices_D[i]
        prev = per_dataset_kl.get(di, (0.0, 0.0))
        per_dataset_kl[di] = (prev[0] + reverse_kl[i].sum().item(), prev[1] + float_masks[i].sum().item())
    avg = sum(d.sum() for d in reverse_kl) / sum(m.sum() for m in float_masks)
    metrics = {"teacher_kl": float(avg)}
    for di, (ks, ms) in per_dataset_kl.items():
        if ms > 0:
            metrics[f"teacher_kl/dataset_{di}"] = float(ks / ms)
    return metrics


def _build_teacher_prefix():
    tok = get_tokenizer(STUDENT)
    msgs = []
    for t, lab in SPEC.fewshot:
        msgs.append({"role": "user", "content": f'TEXT:\n"""\n{t}\n"""'})
        msgs.append({"role": "assistant", "content": "TRIGGER" if lab == 1 else "PASS"})
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=False, enable_thinking=False, tokenize=True)
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


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


def load_prompts(n, only_real=False, exclude_hardening=()):
    """Prompt set for OPD rollouts. Selective distillation = prompt selection:
    --only-real distills just the real (PII-heavy) distribution where the teacher is strong
    and the student over-triggers, leaving synthetic counterfactuals UNSAMPLED (so the SFT
    student's counterfactual ability is preserved — no KL pull there)."""
    rows = []
    for sub in ("it", "real"):
        fp = ROOT / "data" / sub / "train.jsonl"
        if fp.exists():
            rows += [json.loads(l) for l in fp.open() if l.strip()]
    if only_real:
        rows = [r for r in rows if r.get("source") == "real"]
    if exclude_hardening:
        rows = [r for r in rows if r.get("hardening") not in exclude_hardening]
    prompts = [SPEC.build_messages(r["text"], cot=True)[1]["content"] for r in rows]
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
    ap.add_argument("--teacher-fewshot", action="store_true",
                    help="teacher scores rollouts with a few-shot prefix (student stays zero-shot)")
    ap.add_argument("--only-real", action="store_true",
                    help="selective distillation: rollouts on REAL prompts only (preserve counterfactual)")
    ap.add_argument("--exclude-hardening", default="",
                    help="comma-list of hardening families to skip (e.g. counterfactual,obfuscation)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--domain", default="it")
    args = ap.parse_args()
    global SPEC
    SPEC = domains.get(args.domain)

    if args.teacher_fewshot:
        global _TEACHER_PREFIX
        _TEACHER_PREFIX = _build_teacher_prefix()
        train_on_policy.incorporate_kl_penalty = _kl_penalty_fewshot_teacher
        print(f"Asymmetric few-shot teacher: prefix = {len(_TEACHER_PREFIX)} tokens")

    renderer_name = model_info.get_recommended_renderer_name(STUDENT)
    gpb = 8 if args.smoke else args.groups_per_batch
    n_prompts = gpb * (2 if args.smoke else args.steps)
    excl = tuple(x.strip() for x in args.exclude_hardening.split(",") if x.strip())
    prompts = load_prompts(n_prompts, only_real=args.only_real, exclude_hardening=excl)
    print(f"prompt pool: {len(prompts)} (only_real={args.only_real}, exclude={excl})")
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
