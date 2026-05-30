# Experiment log — Policy Classification

## 2026-05-29 — Setup + IT domain v0 (easy) dataset, eval, baseline

### Why IT first
Easiest of the three domains: the in-scope literal-secret cases (API keys, connection
strings) are pattern-matchable, so we get a deterministic regex anchor for free and a clean
contrast against the semantic cases (access changes, security policy, incidents, PII) that
*need* a model. Concrete in/out-of-scope lists in the brief make labelling unambiguous.

### Environment
- `uv` venv (Python 3.12). Installed `tinker` (0.22.2), `transformers`, `scikit-learn`.
- Tinker key from `.env` (`THINKING_MACHINE_API_KEY` -> `TINKER_API_KEY`).
- Base model: `Qwen/Qwen3.5-4B` (hybrid: thinking + non-thinking). Sampling verified.
- Eval uses non-thinking mode (direct label) for the no-CoT baseline.

### Dataset (`data/it/`, generator `data/it/generate.py`)
- 800 rows, ~50/50 pos/neg. Leakage-safe split by `seed_id` -> train 560 / val 120 / test 120.
- 6 positive subcategories, 4 negative subcategories. Coverage axes varied: format
  (slack/email/jira/doc/config/commit), persona, length.
- Positives embed FORMAT-VALID but FAKE secrets (random values) so the regex baseline has
  real signal and we never ship a real credential.
- Small "hard" negative set (56 rows): placeholder/secret-adjacent tutorials
  (`YOUR_API_KEY`, `***`, `ghp_xxxx`, `postgres://USER:PASSWORD@HOST`) — seeds the
  difficulty axis so we can see where v0 is weak.

### Eval harness (`eval/run_eval.py`)
- Modes: `model` (Tinker sampling), `regex` (deterministic), `both`.
- Reports P/R/F1/Acc + confusion matrix, overall and per slice
  (subcategory, difficulty, format, source), with a bootstrap 95% CI on F1.
- Concurrent sampling (ThreadPool), robust label parse, writes per-example preds.

### Results on test (n=120)

| System | Precision | Recall | F1 | Acc | F1 95% CI |
|---|---|---|---|---|---|
| Regex baseline | 95.2 | 34.5 | 50.6 | 67.5 | [36.1, 63.4] |
| Qwen3.5-4B zero-shot | 98.3 | 100.0 | **99.1** | 99.2 | [97.1, 100.0] |

**Regex**: high precision, low recall by construction — 100% on `secret_credential` and
`infra_config`, ~67% on PII (SSN), and **0% on `access_control` / `security_policy` /
`vuln_incident`** (no literal secret to match). Exactly the "anchors the easy cases, blind
to intent" story. 1 FP = the `postgres://USER:PASSWORD@HOST` hard placeholder.

**Model**: saturates. Recall 100% (caught all 58 positives), 1 FP. The single error is the
sharpest signal we have:
> `GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx  # placeholders only`
The model fired on the `ghp_` prefix + token shape despite it being all `x`s — i.e. it is
doing surface pattern-matching, not reasoning about whether the secret is *real*. It got the
other 6 placeholder hard-negatives right.

### Reading
v0 is too easy to discriminate models — F1 99 leaves no headroom to measure SFT/CoT/OPD
deltas. The benchmark, not the model, is now the bottleneck. The one model error and the
regex's recall holes tell us exactly where to add difficulty.

### Next (benchmark hardening — see README "Hardening plan")
Make the eval discriminative before training. Priorities: counterfactual pairs (real vs.
placeholder secret with minimal edits), obfuscation (base64/split keys), intent-only
positives with no literal secret, near-boundary negatives (security *news* vs. *our* policy),
and held-out-policy generalization.

### Loose ends
- 2 unused-var lints in `generate.py` (cosmetic).
- Eval is synthetic-only so far; need real data (HF) for a `source` slice that matters.

## 2026-05-29 — v1 benchmark hardening

### What changed
Rebuilt `generate.py` with `--variant {v0,v1}`. v1 = easy core (500) + 4 hard families,
new first-class `hardening` axis. Rubric encoded in the file: a POSITIVE *discloses/
effectuates*; a NEGATIVE only *references*.
- counterfactual pairs (180 rows / 90 pairs): minimal edits that flip the label; both
  members share a `seed_id` so they never straddle a split (verified: 0 straddling).
- intent_only positives (90): in-scope, no literal secret.
- near_boundary negatives (110): security news, EULA/ToS, vendor security claims,
  abstract questions, generic advice.
- obfuscation positives (70): real secrets base64'd / spaced / chunked (defeat regex).
Total 950 rows; test = 141 (67 hard). Leakage-safe split by seed_id holds.

### Eval honesty fix
Per-slice now reports **support (pos count) + specificity (TNR)**. All-negative slices
(near_boundary, helpdesk, etc.) have degenerate P/R/F1 — F1=0 there was misleading;
specificity is the right number.

### Results on v1 test (n=141)

| System | P | R | F1 | Acc | F1 CI |
|---|---|---|---|---|---|
| Regex | 92.0 | 30.3 | 45.5 | 61.0 | [32, 57] |
| Qwen3.5-4B zero-shot | 92.6 | 98.7 | **95.5** | 95.0 | [92, 99] |

Model F1 **100 → 95.5** (real drop, CIs barely overlap). Headroom now exists.

### Per-hardening (model) — where the difficulty actually is

| hardening | n | F1 | Specificity |
|---|---|---|---|
| core | 74 | 100 | 100 |
| **counterfactual** | 24 | **78.6** | **58.3** |
| intent_only | 13 | 100 | – (all pos) |
| near_boundary | 18 | – (all neg) | 94.4 |
| obfuscation | 12 | 100 | – (all pos) |

### Reading
- All 6 model errors are the engineered boundaries: over-triggers on access **requests**
  (vs grants) and third-party breach **news** (vs our incident); 1 FN on a policy directive.
- **Counterfactual pairs are the entire signal** (spec 58%). The model leans on surface
  topic features and can't reliably separate minimally-different pos/neg.
- Negative results worth keeping: the model is **already robust** to obfuscation (base64/
  spaced secrets → 100%) and to intent-only disclosures (100%). Those families add coverage
  but not difficulty — don't over-invest there.

### Next
- This is a good training target: SFT (esp. with CoT that reasons "is this enacted? is this
  us?") should lift counterfactual specificity. Each stage → new version → Versions tab diff.
- Consider pushing counterfactual volume / variety if we want a steeper curve, and add real
  (HF) data for a meaningful `source` slice.

## 2026-05-29 — Logging upgrades: Findings tab + Tinker cost

### Findings tab
`results/findings.jsonl` + `scripts/add_finding.py` (--version/--title/--finding/--suggestion*
/--tags). New dashboard tab renders observations + suggestions newest-first, version-tagged.
Seeded f001–f005 (saturation, regex foil, counterfactual target, obfuscation negative-result,
specificity methodology).

### Tinker cost per run
Eval harness now counts exact tokens (prompt = tokenized chat input, completion = returned
tokens) per request and sums them. USD = prompt*prefill + completion*sample, rates in
`config/tinker_prices.json` (PLACEHOLDER, `_confirmed:false` — update from the console; cost
is shown as "est."). Stored per-system in history `usage` and surfaced in the Results table
(est $ column + token detail) and Versions (est. cost). Observed: v0 ~$0.0033 (31.8k tok),
v1 ~$0.0039 (38k tok); prompt-dominated since non-CoT emits ~1 completion token (CoT will
shift the ratio toward sample cost — worth watching).

### Note
v0/v1 history+versions were rebuilt cleanly so both carry cost. Prices are placeholders;
confirm Qwen3.5-4B prefill/sample/train rates against the Tinker console before quoting cost.

## 2026-05-29 — v2: expanded counterfactuals

### What changed
CF_GENS 7 -> 15 pair types; cf-pairs 90 -> 170 (340 cf rows). New axes: docs/example secret
vs real config (canonical AWS docs example key), redacted vs present, env-var ref vs
inline value, training/fake vs real, synthetic test PII vs real PII, incident drill vs real
incident, CVE-advisory-for-unused-lib vs confirmed vuln, revoke-enacted vs propose-to-revoke.
Total 1110 rows; test = 161 (40 counterfactual rows, 20/20). Leakage-safe split holds (0
pairs straddle).

### Frozen-model result on v2 test (n=161)
F1 95.8 (CI [92.2, 98.4]); **recall 100%, all 7 errors are false positives**.
- counterfactual: F1 88.9, spec 75% ; near_boundary spec 88.2% ; core/intent/obf saturated.
- FPs: AWS docs example key (x2), access requests/proposals (x2), synthetic test PII,
  third-party breach news (x2).

### Reading
- Dominant failure mode is **over-triggering** on surface-similar negatives. Coherent
  "is it real / enacted / us?" cluster → strong, single training objective (cut FPs).
- The new docs-example / revoke-proposal / synthetic-PII pairs successfully add difficulty
  (they account for most FPs). drill/CVE-advisory/redacted/env-ref pairs the model mostly
  handled — keep but they're not the binding constraint.
- METHOD CAUTION (finding f007): v1->v2 metric deltas are pure DATA-composition effects
  (model frozen). Slice F1 isn't comparable across versions with different slice content.
  Read frozen-model diffs as "benchmark got harder/easier"; read fixed-data diffs as
  "training helped". The counterfactual F1 "rising" 78.6->88.9 is NOT model improvement.

### Next
- Either push counterfactual difficulty further (the docs-example/synthetic-PII style are the
  potent ones), and/or bring in real (HF) data. Then start training with CoT targeting FPs.

## 2026-05-29 — v3: harder counterfactuals

### What changed
CF_GENS 15 -> 21; cf-pairs 170 -> 230 (460 cf rows). New pair types target the over-trigger
axis: vendor example/test keys (Stripe/Google docs), redaction-LEAK (says "redacted" but the
full secret is present — adversarial vs the v2 [REDACTED]→neg shortcut), last-4-only vs full
PAN, expired/dead key vs live (rubric call), test-fixture dummy vs real prod token, policy
question vs stated policy. Total 1230 rows; test = 180 (60 cf rows, 30/30). 0 pairs straddle.

### Frozen-model result on v3 test (n=180)
F1 **93.3** (CI [89.2, 96.8]); recall 100%, 12 FPs. Headroom now ~7 pts.
- counterfactual F1 88.2 / spec 73.3 ; near_boundary spec 80.0 ; core/intent/obf saturated.
- FP clusters: AWS docs example key x4, Stripe test key x1, third-party breach NEWS x4,
  expired/dead key x1, revoke-proposal x1, synthetic test PII x1, last-4 card x1.
- regex precision fell 92->81 (it flags the AWS example key and partial/test cases too).

### Reading
- Two crisp, durable training targets: (a) documented example/placeholder secrets that LOOK
  real; (b) third-party news vs OUR incident. Both are "is it real / ours / enacted?" — a
  natural fit for CoT supervision.
- breach-NEWS (near_boundary) is now the weakest negative template (4 of 12 FPs) — a candidate
  to expand if we want more near-boundary pressure later.
- Rubric call logged (f009): dead/revoked credential = PASS. Documented, one-line to flip.

### Curve so far (frozen Qwen3.5-4B, F1) — benchmark difficulty, model fixed
v0 98.2 -> v1 96.2 -> v2 95.8 -> v3 93.3. Monotonic as designed.

## 2026-05-29 — real HF data (real-v1)

### Pipeline
`data/it/fetch_real.py`: streams 4 HF sources, labels via metadata+regex, dedups (incl. vs
synthetic), splits leakage-safe, emits data/real/{train,val,test,test_realistic,test_balanced}.
Sources (700 each, 2800 total kept): ai4privacy/pii-masking-300k (pii_handling, entity-type
gated), Tobi-Bueck/customer-support-tickets (support −, breach-tagged + incident-kw → vuln
+), AlicanKiraz0 CVE records (→ security_news), iamtarun python code (public_snippet −).
secret_credential/infra_config/access_control/our-policy = synthetic-only (no clean source).
`run_eval.py --data` now accepts comma-separated files (→ source slice).

### Verification loop (Claude as non-Qwen judge, sampled)
Caught a ~45% false-positive rate in the first support-positive rule (generic Security/
Outage/Incident tags swept in outages, bug reports, security *inquiries*). Tightened to
(security tag) AND (incident keyword) → re-verified clean. PII positives mostly OK (a few
borderline from truncation); ai4privacy text reads artificially (distribution caveat).

### Headline: combined eval (synthetic v3 test + real realistic, 12% pos), frozen model
- synthetic F1 94.9 / spec 90.6
- REAL F1 41.0 / spec 69.6  ← but decomposes:
  - REAL CVE only: model flags 91/93 as TRIGGER (spec 2.2%)
  - REAL excl. CVE: F1 76.9 / spec 93.8 (code 99 / ticket 98 / support 97% spec)
- est. cost $0.0215 (210k tokens), placeholder price.

### Reading (findings f010/f011)
- Real eval immediately earned its keep: it exposed a LABELING BUG, not just model error.
  I mapped CVE→security_news→negative, but the policy lists "vulnerability/incident details"
  IN-SCOPE. The model agrees with the policy. 91 of 107 real FPs are CVEs. This also
  contradicts synthetic cf_cve_advisory → the rubric is inconsistent and must be resolved.
- Setting CVE aside, the synthetic→real gap is real but moderate: model transfers well to
  real negatives; residual weakness is precision on PII prose (spec 74%). NOT a collapse.
- Method note: "F1 41" is a slice artifact of one debatable label — always decompose before
  concluding "the model is bad."

### Next: resolve the CVE rubric, realign synthetic cf_cve_advisory, re-measure real.

## 2026-05-29 — v4: casual register + typo noise (realism)

### Why
Real eval showed the model under-triggers on real positives (recall 80%) because real text is
long/casual/messy while synthetic was short/clean. Closing that register gap.

### What
- `casual` hardening family: long lowercase Slack/Reddit/forum posts. Negatives = techie PSAs/
  vents/war-stories that disclose nothing (e.g. the Claude-Code rate-limit PSA). Positives =
  rambling messages that BURY a real secret/PII/access/incident. Openers/closers/slots give
  combinatorial variety.
- `inject_noise`: realistic typos (transpose/drop/double), contraction-stripping, dropped caps
  on ~40% of rows. Protects structured tokens (keys, emails, URLs, numbers, ALLCAPS) so secrets
  and regex labels stay valid. New `noisy` field + eval slice.
- Global text-dedup before split → 0 identical texts across train/test (verified); also pruned
  ~50 accidental template dups. cf pairs still 0 straddling. data/it now 1180 rows.
- Eval slicer hardened to tolerate rows missing a key (real rows lack `noisy`).

### Result (synthetic v4 + real, frozen model)
- synthetic F1 92.8 (v3 was ~94.9 — casual+noise made it modestly harder)
- **noisy slice: clean F1 95.6 / spec 89.4  vs  noisy F1 88.9 / spec 74.3** → ~15pt
  specificity drop = typo brittleness (over-triggers on misspelled negatives; recall stays 100).
- casual slice handled well in this test cut; real F1 unchanged at 70.

### Reading (finding f013)
Realism treatment did its job: surfaced a new, trainable weakness (typo robustness) and added
the casual register real data has. Noise augmentation in TRAINING is now an obvious lever and
should also help real-data specificity.

## 2026-05-29 — Part 3: SFT (stage 1)

### Setup
- `train/build_sft_data.py`: synthetic+real train/val → conversations JSONL (eval prompt +
  label as assistant turn). 2038 train (718 pos), 497 val. Noise already baked into synthetic.
- `train/sft.py`: cookbook SFT (FromConversationFileBuilder + supervised.train), Qwen3.5-4B,
  LoRA r32, TrainOnWhat.ALL_ASSISTANT_MESSAGES (num_loss_tokens=1/example confirms label-only
  loss). Smoke (2 steps) then full (2 epochs, ~30 steps, train NLL ~0.005). Eval via our
  harness on the SAME held-out test (--model tinker://<ckpt>).

### SFT v1 vs frozen (model changed, data fixed — the valid comparison, f007)
| slice            | frozen v4 | SFT v1 |
|------------------|-----------|--------|
| synthetic F1     | 92.8      | 99.4   |
| real F1          | 70.0      | 73.7   |
| counterfactual sp| 47.6      | 95.2   |
| noisy spec       | 74.3      | 100    |
| real recall      | 80        | 100    |
- recall now 100% everywhere (0 FN). Both engineered weaknesses (counterfactual over-trigger,
  typo brittleness) essentially solved. Real under-triggering fixed (recall 80->100).
- Residual: real PII-prose precision (prose spec 55%, pii spec 57%) — over-fires on casual
  names/emails; partly debatable labels (low-sensitivity PII rubric).

### Next
- Stage 2: CoT SFT (reason "real/enacted/ours?" then label) — target real PII-prose precision.
- Stage 3: on-policy distillation.
- Ablations: CoT vs no-CoT (+ latency/cost), hard-negs in/out, synthetic-only vs mixed,
  per-policy adapter vs unified, LoRA rank, data-scale curve, held-out-policy.

## 2026-05-29 — separate dataset versions from iterations (for ablation)

The old `versions.jsonl` conflated dataset versions (v0..v4) with experiments (real-v1, sft-v1).
Split into:
- **`results/datasets.jsonl`** (`scripts/snapshot_dataset.py`): named `ds-vX` = data composition
  + content fingerprint. Migrated ds-v0..ds-v3 (synthetic), ds-v3+real, ds-v4 (synth v4 + real,
  fp 06d0912).
- **`history.jsonl`** iterations now carry `dataset_version` (run_eval `--dataset-version`);
  backfilled iters 1-6→ds-v0..ds-v4, iter7 (SFT)→ds-v4.
- Dashboard: **Versions tab → Datasets tab** — composition + diff + an **ablation table** per
  dataset version (iterations/models run against it). Results table gained a `dataset` column.
- Retired `snapshot_version.py` + `versions.jsonl`.

Ablation now reads directly: on **ds-v4**, frozen (iter6) vs SFT (iter7) → F1 86→91,
counterfactual-spec 48→95, real-F1 70→74. Hold data fixed, vary model = training gains.
