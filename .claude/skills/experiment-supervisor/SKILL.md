---
name: experiment-supervisor
description: >
  Autonomous self-improving experiment loop for the policy-classification project. Each tick:
  sense state (scripts/supervisor.py), run the highest-value next experiment, VERIFY it against
  computable success criteria + the methodology critic, update the plan, log a finding, and
  commit to the yolo branch. Use when running the project unattended ("yolo mode").
---

# Experiment supervisor (yolo loop)

Run on the `yolo/auto-experiments` branch. NEVER push to `main`. Follow the `policy-domain`
skill for the per-domain methodology and its principles.

## Each tick

1. **Sense**: `python scripts/supervisor.py --status`. It prints, per domain, the next action
   (with command) and the billing status.
2. **Billing gate**: if billing is BLOCKED (402), STOP the loop, write a finding, and notify —
   do NOT spin. (This is error handling, not a budget cap.)
3. **Act**: do the recommended next action for the most-incomplete domain:
   - `dataset` → run the generator.
   - `discriminative` + SATURATED → **harden**: add subtler counterfactual / near-boundary
     content to `data/<domain>/generate.py` (boilerplate-that-reads-binding, quotes-in-news,
     aspirational-vs-binding for Legal; claim-in-internal-note, opinion-with-superlatives,
     competitor-claim for Marketing), regenerate, re-eval frozen. Goal: frozen F1 into [0.80, 0.94].
   - `real_data` → build `data/<domain>/fetch_real.py` (HF sources per domain), fetch, combined eval.
   - `sft` → `build_sft_data --domain` → `sft.py` → eval the checkpoint.
4. **Verify (critic — don't self-delude)**: re-run `--status`. Apply the principles: is the change
   real or a label artifact (f030)? is n/CI too small to trust? is the benchmark still saturated
   (then keep hardening, don't declare victory)? compare model-fixed OR data-fixed, never both (f007).
5. **Record**: `add_finding.py` for the takeaway; `snapshot_dataset.py` for new dataset versions;
   `git add -A && git commit` to the yolo branch (concise message). Push the yolo branch.
6. **Continue or stop**: if all domains show "— done" → stop. Else `ScheduleWakeup` (or next /loop
   tick) and repeat. Cap retries per stage at ~3 (avoid noise-chasing); if stuck, log it and move on.

## Full-yolo defaults (auto-decide, log loudly)

Contestable rubric calls are auto-decided with a documented default and flagged in a finding:
- third-party advisories/news → NEGATIVE (reference, not our disclosure)
- explicitly dead/revoked credentials → PASS
- low-sensitivity-only PII (lone name/email) → PASS; multi sensitive identifier → TRIGGER
Operating points: Legal favors RECALL; IT/Marketing favor PRECISION.

## Guardrails

Branch-only (never main). Billing precheck each tick. Max ~3 retries/stage. Datasets gitignored
(reproducible from seeds). Inspect training masks; report outcomes faithfully (failures included).
