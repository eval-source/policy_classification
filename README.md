# Policy Classification

Train and evaluate a base model (**Qwen3.5-4B** via [Tinker](https://tinker-docs.thinkingmachines.ai))
to classify text against natural-language **policies** — does a piece of text *trigger* a policy?
This repo currently covers the **IT** domain (sensitive technical / security / access content)
end to end: synthetic dataset, a sliceable eval harness, and a dashboard for tracking iterations.

## Layout

```
data/it/generate.py     synthetic IT-domain generator (v0 easy core + v1 hard families)
eval/                   sliceable eval harness (Tinker sampling + regex baseline), metrics, cost
viz/                    single-page dashboard (Results · Dataset · Datasets · Findings) + serve.py
train/                  SFT data builder + Tinker SFT runner (Part 3)
scripts/                env bootstrap, version snapshots, findings logging
results/                history.jsonl, versions.jsonl, findings.jsonl, summary.json
notes/log.md            running experiment log
config/tinker_prices.json   per-token price table for cost estimates (UPDATE from console)
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python tinker transformers scikit-learn jinja2 python-dotenv numpy
echo "THINKING_MACHINE_API_KEY=<your-tinker-key>" > .env   # gitignored; mapped to TINKER_API_KEY
```

## Reproduce the dataset

Datasets are **not committed** — they are deterministic from the seeded generator (and contain
fake secret-*format* strings that trip secret scanners). Reproduce with:

```bash
python data/it/generate.py --variant v1 --seed 0 --out data/it   # v1 = easy core + hard families
python data/it/generate.py --variant v0 --seed 0 --out data/it   # v0 = easy core only
```

Hard families (`hardening` field): `counterfactual` (minimal-edit label flips, kept together by
`seed_id` so they never straddle a split), `intent_only`, `near_boundary`, `obfuscation`.

## Evaluate

```bash
python eval/run_eval.py --data data/it/test.jsonl --mode both        # frozen model + regex
python eval/run_eval.py --data data/it/test.jsonl --mode model --cot # CoT variant
```

Reports P/R/F1 + **specificity** per slice (subcategory / hardening / difficulty / format), bootstrap
F1 CIs, token usage and an estimated Tinker cost. Each run appends an iteration to
`results/history.jsonl`.

## Dashboard

```bash
python viz/serve.py     # serves the repo root; open http://127.0.0.1:8000/viz/index.html
```

Live-loads the JSONL files (no DB, no build step) — refresh after any run.
- **Results** — every eval iteration, sortable; shows its `dataset` version; expand for confusion
  matrix, per-slice detail, cost, and ✗failures/✓successes per run.
- **Dataset** — browse/filter every row (synthetic + real, by source) + stats dashboard.
- **Datasets** — dataset versions (`ds-vX`, via `scripts/snapshot_dataset.py`): composition + diff
  vs the previous version, AND an **ablation table** of the iterations run against each version.
- **Findings** — `scripts/add_finding.py` observations + suggestions per iteration.

## Concepts: dataset versions vs iterations

- A **dataset version** (`ds-vX`, in `results/datasets.jsonl`) is the data composition + a content
  fingerprint. Snapshot with `scripts/snapshot_dataset.py --version ds-vX`.
- An **iteration** (`results/history.jsonl`) is one eval run = (model/config) × dataset version;
  it records `dataset_version`. Pass `--dataset-version ds-vX` to `run_eval.py`.
- Ablation = hold one fixed, vary the other: same `ds-vX` across models → **training gains**;
  same model across `ds-vX` → **benchmark difficulty**.

## Train (Part 3)

```bash
python train/build_sft_data.py                 # synthetic+real train -> conversations JSONL
python train/sft.py --smoke                     # validate the pipeline (2 steps)
python train/sft.py --name sft_v1 --epochs 2    # prints the tinker:// sampler checkpoint
python eval/run_eval.py --data data/it/test.jsonl,data/real/test_realistic.jsonl \
    --model tinker://<ckpt> --dataset-version ds-v4 --note "SFT v1"
```

## Status / results

See **[DELIVERABLES.md](DELIVERABLES.md)** for the full summary (datasets, eval, ablation table,
findings, next steps). Headline:

- 8 dataset versions (ds-v0→ds-v7); frozen-model F1 **98→75** as the benchmark hardened + added real
  data (model fixed = benchmark-difficulty curve).
- **SFT (LoRA) is the production win** — solves every slice except PII-prose.
- **CoT and STaR thinking aren't worth it** (templated CoT pattern-matches; STaR rejection-sampling
  drops the hard cases). **OPD is operating-point-dependent**: a few-shot 27B teacher wins on the
  real/PII-heavy distribution (ds-v7 real F1 62→74) but at a counterfactual + cost trade.
- **Label audit** (regex+judge+human, κ=0.906) found/fixed systematic label errors; label quality
  itself moves the headline metric.

ds-v7 production menu: **SFT v4** (counterfactual-critical/cheap) · **OPD v8** (balanced) ·
**OPD v7** (max real precision). No single model dominates all slices — pick by operating point.
