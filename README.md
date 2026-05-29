# Policy Classification

Train and evaluate a base model (**Qwen3.5-4B** via [Tinker](https://tinker-docs.thinkingmachines.ai))
to classify text against natural-language **policies** — does a piece of text *trigger* a policy?
This repo currently covers the **IT** domain (sensitive technical / security / access content)
end to end: synthetic dataset, a sliceable eval harness, and a dashboard for tracking iterations.

## Layout

```
data/it/generate.py     synthetic IT-domain generator (v0 easy core + v1 hard families)
eval/                   sliceable eval harness (Tinker sampling + regex baseline), metrics, cost
viz/                    single-page dashboard (Results · Dataset · Versions · Findings) + serve.py
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
- **Results** — every eval iteration, sortable, with confusion matrix + per-slice detail + cost.
- **Dataset** — browse/filter every row + stats dashboard.
- **Versions** — `scripts/snapshot_version.py` snapshots (dataset stats + eval); shows the diff/ΔF1.
- **Findings** — `scripts/add_finding.py` observations + suggestions per iteration.

## Status

Frozen-model F1 curve (benchmark difficulty, model fixed): **v0 98.2 → v1 96.2 → v2 95.8 → v3 93.3**.
The dominant failure mode is over-triggering on near-boundary negatives (documented example keys,
third-party breach news, access requests-vs-grants). Next: real-data mix, then SFT/CoT training.
