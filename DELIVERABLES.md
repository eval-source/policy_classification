# Policy Classification — Deliverables Summary

One-policy binary classification (does text **trigger** an IT-security policy?), built on **Qwen3.5-4B**
via Tinker. This summarizes the datasets, eval harness, trained models, ablations, and conclusions.
Full blow-by-blow is in `notes/log.md`; per-iteration observations are in the dashboard **Findings**
tab (f001–f032). All numbers are reproducible (seeded generator + fetcher; eval over Tinker).

## TL;DR

- **The benchmark, not the model, was the early bottleneck.** A frozen Qwen3.5-4B scores **F1 98** on
  an easy synthetic set — so I hardened the data through 8 versions (counterfactual pairs, casual
  register, typo noise, real HF data) until it was discriminative and realistic.
- **SFT (LoRA) is the production win**; it solves every slice except PII-prose.
- **CoT and naive OPD are *not* worth it**, with demonstrated reasons (teacher quality-ceiling,
  rejection-sampling coverage bias, templated-rationale pattern-matching).
- **OPD becomes worth it when the teacher's strength matches the student's bottleneck and the eval's
  weight** — a few-shot 27B teacher fixed SFT's PII over-triggering on the real-heavy distribution.
- **Label quality silently moves the headline metric**; a regex+judge+human audit (κ=0.906) found and
  fixed the systematic label errors.
- **Across all three domains, clean-synthetic F1 does not predict real-world transfer** — and
  synthetic-only SFT can *regress* it. The reliable lever is training on data that matches the
  deployment distribution (incl. the real hard-negative/positive classes). See §5.

## 1. Datasets (`data/it/generate.py`, `data/it/fetch_real.py`)

Domain chosen: **IT** (most tractable — literal secrets are regex-anchorable, in/out scope concrete).
Leakage-safe splits by `seed_id` (counterfactual pairs + paraphrases never straddle splits; verified 0).
Datasets are **gitignored** (reproducible from seeds; contain fake secret-format strings + real PII).

| version | rows | what it added | frozen-model F1 |
|---|---|---|---|
| ds-v0 | 800 | easy synthetic core (6 pos / 4 neg subcats) | **98.2** |
| ds-v1 | 950 | hard families: counterfactual pairs, intent-only, near-boundary, obfuscation | 96.2 |
| ds-v2 | 1110 | +counterfactual pair types | 95.8 |
| ds-v3 | 1230 | harder counterfactuals (docs-example keys, dead-key, redaction-leak…) | 93.3 |
| ds-v3+real | 3072 | +real HF (PII, support tickets, code; **CVE dropped** as ambiguous) | real F1 ~70 |
| ds-v4 | 3022 | +casual register (Slack/Reddit) +typo noise (secrets protected) | — |
| ds-v5 | 3108 | PII rubric v2 (multi-identifier records → positive) | — |
| ds-v6 | 4009 | +terraform (real infra_config), +infosec-policy docs (real security_policy), +Enron email | 77.5 |
| ds-v7 | 3970 | **label-audit fixes**: sensitive-identifier PII rule + password-disclosure regex | 74.9 |

The monotonic frozen-F1 drop **98→75** is the benchmark getting harder/more realistic with the model
held fixed (not the model degrading — see `notes/log.md` on holding one axis fixed).

**Coverage note:** `secret_credential`, `access_control` have **no clean public HF source** (sensitive
content isn't published) → they stay synthetic. Real data covers PII, support, code, infra, policy, email.

## 2. Eval harness (`eval/run_eval.py`)

One command → P/R/F1 + **specificity** per slice (subcategory · hardening · difficulty · format ·
**source** · noisy), bootstrap F1 CIs, confusion matrix, **token usage + est. cost**, and a deterministic
**regex baseline**. Accepts multiple data files (synthetic+real → a `source` slice). Every run logs an
iteration to `results/history.jsonl` (tagged with its `dataset_version`) and an immutable per-run preds
file (powers the dashboard's ✗failures/✓successes view). Dashboard (`viz/`, no DB): Results · Dataset ·
Dataset Versions (with ablation tables) · Findings.

**Mixed scoring stack** (per the brief): regex (literal secrets), LLM-judge (few-shot Qwen3.6-27B for
scale; Claude for an independent gold check), human adjudication of disagreements.

## 3. Trained models + ablation (headline: ds-v7, correctly-labeled data)

| model | overall F1 | real F1 | PII spec | counterfactual spec | cost/ex | when to ship |
|---|---|---|---|---|---|---|
| frozen Qwen3.5-4B | 74.9 | 52.6 | 85 | (over-triggers) | low | baseline |
| **SFT v4 (LoRA)** | 80.8 | 62.0 | 51 | **100** | ~1k tok | counterfactual-critical / cheap |
| OPD v8 (selective, real-only) | 83.7 | 68.0 | 72 | 76 | ~43k tok | balanced |
| **OPD v7 (global few-shot)** | **85.7** | **73.9** | 70 | 57 | ~45k tok | **max real-world precision** |

SFT lifts the engineered weaknesses decisively (across versions: counterfactual spec 48→95, typo-noise
spec 74→100, real recall 80→~97). On the real distribution the few-shot-teacher OPD then fixes SFT's
remaining PII over-triggering. **No single model dominates all slices** — it's an operating-point choice.

## 4. Key findings (what worked / what didn't)

**Data / eval methodology**
- **Saturation → harden** (f001): easy benchmarks can't measure training deltas; counterfactual pairs are
  the discriminative signal (f003/f008); the dominant model failure is **over-triggering**.
- **Eval honesty** (f005/f007): report **specificity** on all-negative slices (F1 is degenerate there);
  slice-F1 isn't comparable across versions when slice content changes — hold model *or* data fixed.
- **Real eval earns its keep** (f010/f024): it exposed both a synthetic→real gap *and* a label bug (CVE),
  and reliably caught a distribution gap that **retraining on the new sources fixed** (real F1 49→74).
- **Label quality moves the metric** (f030): lenient PII labels *inflated* real F1 by scoring the model's
  over-triggers as correct; correct labels revealed PII as the true bottleneck.
- **Label audit** (f028/f029): regex+judge cross-check → κ=0.906 (high); adjudication found ~1% genuine
  errors (PII narrative over-labeling; Enron under-labeling incl. a missed password). **Agreement with a
  same-family biased judge is *not* ground truth.**

**Training**
- **SFT wins** and is cheap. **CoT (templated) not worth it** (f015/f026): pattern-matched the rationale,
  hurt precision, 15× cost. **STaR native thinking underperforms** (f023): rejection sampling discards
  the hard cases the base gets wrong, so it can't teach them.
- **OPD is operating-point-dependent** (f017→f032): zero-shot larger Qwen teachers (235B/27B) are *worse*
  than the specialized 4B (teacher quality-ceiling) → naive OPD regresses. A **few-shot** teacher becomes
  competent + complementary (strong PII); distilling it **wins when PII is the bottleneck and the eval is
  real-heavy** (ds-v7), loses when counterfactuals are weighted (ds-v5). More steps just converge harder
  to the teacher's profile (f022). **Selective (real-only) distillation only partially isolates a slice**
  (f032) — LoRA updates shared weights; a mixed KL+SFT-anchor objective is the real fix.

## 5. Multi-domain real-world transfer (Legal + Marketing)

The IT recipe is domain-parameterized (`domains/` registry + a reusable `policy-domain` skill), so
Legal and Marketing reuse the same generator/eval/SFT code. On **synthetic** data the base model was
already strong (Legal F1 100, Marketing 96) — low SFT headroom, opposite of IT. The interesting test
was **real HF data**, built the same way (a real fetcher per domain, leakage-safe by-row splits, a
realistic positive-rare view):

| domain | real source: **pos** / hard-neg / easy-neg | frozen on REAL | synthetic-only SFT | **synth+real SFT** |
|---|---|---|---|---|
| Legal | LEDGAR clauses / **unfair-ToS** / AG-news | F1 93.4 (ToS spec 77) | **87.6** — *regresses* (ToS spec 37) | **100** (ToS spec 100) |
| Marketing | Amazon product copy / **reviews** / AG-news | F1 37.1 (recall 24) | 51.4 (recall 36) | **98.7** (recall 100) |

**The headline cross-domain finding:** synthetic metrics are not predictive of transfer **in either
direction**. Legal *over*-triggered on real Terms-of-Service (same legalese as binding clauses, opposite
label) — and synthetic-only SFT made it *worse*. Marketing *under*-triggered on real product copy (it
doesn't look like synthetic taglines/promos). **Opposite symptoms, identical cause** (synthetic train
distribution ≠ real) and **identical fix**: put the real hard class in training → F1 → ~99 on both, and
it holds on the positive-rare view (Legal 98.6, Marketing 97.3). This generalizes the IT lesson (real
eval earns its keep) into a sharper claim: *clean-synthetic-only SFT is a ceiling on the easy case, not
a deployment estimate, and may regress the real operating point.*

**Cross-source generalization (no retraining)** — tested whether the synth+real SFT learned the
*policy* or memorized the training source, by evaluating on **unseen sources**:

| domain | unseen pos / hard-neg / easy-neg | same-source F1 | **cross-source F1** | what generalized |
|---|---|---|---|---|
| Legal | CUAD contracts / held-out ToS / Wikipedia | 100 | **98.6** (−1.4) | both: CUAD recall 97, held-out ToS spec 85→**100** |
| Marketing | bprateek copy / movie reviews / Wikipedia | 98.7 | **87.1** (−11.6) | positives only: recall 28→**99**; neg spec **drops** (reviews 73, wiki 84) |

**Cross-source generalization is domain-dependent.** Legal's *binding-instrument-vs-boilerplate* boundary
is sharp → it generalizes to an entirely different contract corpus and suppresses over-triggering on
held-out boilerplate (caveat essentially removed). Marketing's *outbound-claim-vs-any-descriptive-prose*
boundary is fuzzy → the model generalizes on positives but **over-triggers on unseen negatives** (movie
reviews, even Wikipedia), so same-source F1 (98.7) overstates real cross-source performance (87.1). The
honest takeaway: **same-source metrics can be optimistic for fuzzy-boundary policies** — broaden the
negative training distribution or deploy high-recall + a precision gate (§6).

## 6. What I'd do next

- **Mixed-objective OPD** (KL-to-teacher on real + SFT/self-anchor on counterfactual) for a single
  dominant model.
- **Human-labeled PII gold set + explicit sensitivity threshold** with stakeholders — the residual PII
  noise is irreducible from metadata; or run a high-recall model + human precision gate in production.
- **Cross-source generalization** — **done** (§5): Legal generalizes (98.6), Marketing partially (87.1,
  over-triggers on unseen negatives). **Next:** broaden Marketing's negative training distribution
  (diverse non-marketing prose across genres) and re-run the cross-source probe to confirm the precision
  recovers; consider a precision gate for the fuzzy-boundary domain.
- **Held-out-*policy* generalization** (Part 2's open axis): train on IT, eval on an unseen IT-adjacent
  policy to separate "learned the format" from "learned the semantics."
- **Extend to Legal + Marketing** domains — **done** (§5); both reuse the IT recipe via `domains/` + the
  `policy-domain` skill, and the autonomous supervisor (`scripts/supervisor.py`) drove them to completion.
- **Calibration + per-domain operating points** (IT: precision-favoring to avoid alert fatigue).
- **Confirm Tinker prices** (`config/tinker_prices.json` is a placeholder) for calibrated $ costs.

## 6. Reproduce

```bash
python data/it/generate.py --variant v1 --seed 0        # synthetic
python data/it/fetch_real.py --limit 700 --seed 0       # real HF (-> ds-v7 labeling)
python eval/run_eval.py --data data/it/test.jsonl,data/real/test_realistic.jsonl --mode both \
    --dataset-version ds-v7 --note baseline               # frozen + regex
python train/sft.py --name sft --epochs 2               # SFT (prints tinker:// ckpt)
python train/opd.py --name opd --teacher-fewshot --teacher Qwen/Qwen3.6-27B \
    --student-ckpt tinker://<sft-state> --steps 32        # few-shot OPD
python scripts/audit_labels.py --judge-preds <judge-run> # label audit
python viz/serve.py                                      # dashboard at /viz/index.html

# other domains (same recipe; --domain selects the spec in domains/)
python data/legal/fetch_real.py --limit 500 --seed 0     # LEDGAR / unfair-ToS / AG-news
python data/marketing/fetch_real.py --limit 500 --seed 0 # Amazon product-copy / reviews / AG-news
python train/build_sft_data.py --domain legal && python train/sft.py --name legal_sft_real
python eval/run_eval.py --data data/legal/real/test_balanced.jsonl --mode model \
    --model tinker://<ckpt> --domain legal --dataset-version legal-real-v1
python scripts/supervisor.py --status                    # per-domain stage + next action
```
