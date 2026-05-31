---
name: policy-domain
description: >
  Build or extend a policy-classification domain (binary: does text TRIGGER a natural-language
  policy?) end-to-end — synthetic data, a sliceable eval harness, real HF data, a label audit,
  and SFT/OPD training on Tinker — reusing the shared infrastructure in this repo. Use when
  adding a new domain (e.g. Legal, Marketing) or hardening/measuring an existing one. The value
  is the METHODOLOGY below, distilled from the IT build (findings f001–f032); follow the
  sequence and the principles, don't re-derive them.
---

# Policy-domain builder

You are doing applied ML research: hypothesize, run a small controlled experiment, read it
honestly, iterate. **The dataset/eval is usually the bottleneck, not the model.**

## Architecture (already domain-parameterized — reuse it)

- **`domains/<name>.py`** — a `DomainSpec` (policy text, few-shot examples, CoT-rationale vocab,
  regex patterns). This + `domains/_base.py` is ALL the domain-coupled inference/labeling logic.
  Register the name in `domains/__init__.py`.
- **`data/<name>/generate.py`** — synthetic generator (domain-specific content; same row schema
  + hard-family structure as IT). **`data/<name>/fetch_real.py`** — real HF source handlers.
- **Shared, take `--domain`**: `eval/run_eval.py`, `train/{build_sft_data,sft,opd,sft_think}.py`,
  `scripts/audit_labels.py`. Dashboard `viz/` + `scripts/snapshot_dataset.py` + `add_finding.py`
  are domain-agnostic. **Row schema**: `id, seed_id, text, label, subcategory, difficulty,
  hardening, pair_id, source, format, noisy`.

## The recipe (sequence)

1. **Write the spec** (`domains/<name>.py`): policy in/out-of-scope text (copy the brief's
   wording), 8–10 few-shot boundary examples, `pos_kind`/`neg_why` CoT maps keyed on the domain's
   subcategories, and regex patterns (may be sparse/empty for non-IT domains — that's fine).
2. **Synthetic generator** (`data/<name>/generate.py`): start from `data/it/generate.py`'s
   structure. Per-domain POSITIVE subcategories + NEGATIVE subcategories, then the **hard
   families** (the discriminative signal):
   - `counterfactual` pairs — minimal edits that flip the label (share a `seed_id` so they never
     straddle a split). **This is where the signal is.**
   - `intent_only` (in-scope, no literal cue), `near_boundary` (sounds in-scope, isn't),
     `obfuscation` (in-scope but disguised), `casual` (long/lowercase/rambling register), and a
     `noise` injector (typos; PROTECT any literal-cue spans like secrets).
3. **Baseline eval**: `run_eval.py --mode both --domain <name>` (frozen model + regex). Expect a
   high F1 on easy data → **harden until discriminative** (frozen F1 should drop into a range with
   headroom). Register dataset versions: `snapshot_dataset.py --version ds-vN`.
4. **Real HF data** (`fetch_real.py`): pull sources that match the domain; label by metadata +
   regex (cheap, bulk). Hold out a `source=real` test slice (positive-rare + a balanced cut).
5. **Label audit** (`audit_labels.py`): cross-check labels vs regex + a strong judge; adjudicate
   the disagreement queue yourself (independent). Fix systematic errors → new ds version.
6. **Train**: `build_sft_data.py --domain` → `sft.py`. Then ablate (CoT, OPD) only if warranted.
   Eval each via `run_eval.py --model tinker://<ckpt> --domain <name> --dataset-version ds-vN`.
7. **Log**: every eval auto-appends to `results/history.jsonl` (tagged `dataset_version`); record
   takeaways with `add_finding.py`. Snapshot dataset versions so the dashboard ablation tables work.

## Principles (hard-won — apply, don't relearn)

- **Hold one axis fixed.** Compare models on the SAME dataset version (training gains) OR the
  frozen model across versions (benchmark difficulty) — never both at once (f007).
- **Report SPECIFICITY on all-negative slices** (P/R/F1 are degenerate there) (f005).
- **Counterfactual pairs are the discriminative signal**; the dominant failure is OVER-triggering
  on near-boundary negatives (f003/f006/f008).
- **Real data's value is highest IN TRAINING when it matches the deployment distribution**; a
  held-out real eval reliably catches distribution gaps — then retrain on the new sources (f024/f025).
- **Label quality silently moves the headline metric** (lenient labels inflate it). Audit before
  trusting numbers. **Agreement with a same-family biased judge is NOT ground truth** (f028/f029/f030).
- **SFT is the workhorse.** CoT (templated) tends to pattern-match and not help; STaR thinking has
  a rejection-sampling coverage bias (drops the hard cases) (f015/f023).
- **OPD is operating-point-dependent**: a bigger zero-shot teacher is often WORSE than the
  specialized small model (teacher quality-ceiling). A few-shot teacher can become competent +
  complementary; distilling it helps ONLY when its strength matches the student's bottleneck AND
  the eval's weight (f017/f027/f031). LoRA updates shared weights, so prompt-selection alone can't
  isolate a slice (f032).
- **Some in-scope categories have no clean public source** (sensitive content isn't published) →
  synthesize them; don't force a bad real source.
- **Domain character differs** (per the brief): tune the operating point — Legal is precise/formal
  with costly misses (favor RECALL); Marketing lives in tone/claims gray areas; IT favors PRECISION
  (alert fatigue).

## Pitfalls

- Hybrid models (Qwen3.5): native `<think>` mode mismatches plain prompted-CoT — train and eval
  must match. For thinking SFT, train on RAW tokens (generation prompt + sampled completion), not
  the conversation-file renderer (it puts loss on the prompt's `<think>`).
- Protect literal-cue spans from the noise injector and from secret-format strings being committed
  (GitHub push-protection blocks them) — datasets are gitignored, reproducible from seeds.
- `num_loss_tokens` is misleading; INSPECT the training mask.

## Commands

```bash
python data/<name>/generate.py --variant v1 --seed 0
python data/<name>/fetch_real.py --limit 700 --seed 0
python eval/run_eval.py --data data/<name>/test.jsonl,data/real/test_realistic.jsonl \
    --mode both --domain <name> --dataset-version ds-vN --note "..."
python scripts/snapshot_dataset.py --version ds-vN --note "..."
python train/build_sft_data.py --domain <name> && python train/sft.py --name sft
python scripts/audit_labels.py --domain <name> --judge-preds <judge-run>
python viz/serve.py    # dashboard
```
