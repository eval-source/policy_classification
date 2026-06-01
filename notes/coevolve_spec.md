# Co-evolution loop — Legal pilot spec

Adversarial / dynamic-benchmark co-evolution (lineage: Dynabench, Adversarial NLI, expert-iteration —
**not** literally a GAN). Alternately (a) **mine** Legal examples the current student fails but a strong
oracle solves, (b) **train** the student on them, until the student matches the oracle on the *solvable*
region — while tracking true progress on a **frozen anchor** so we don't fool ourselves.

This formalizes the harden→train→recover loop we ran by hand for IT (cf-spec 48→95) and Legal
(frozen 100→83 via minimal-edit CFs → SFT 99). The new content: an oracle-gated **difficulty signal**,
an automated **iterate-until-frontier-dry** controller, and **anchor tracking** to prevent Goodharting.

## Three roles (the key reframing)

- **Student S** — Qwen3.5-4B, the player being trained. `S_t` at round t (frozen 4B at t=0).
- **Oracle T** — Qwen3.6-27B, **few-shot**. A **label/solvability oracle**, *not* a discriminator to fool.
  Must be few-shot: zero-shot 27B is *worse* than the specialized 4B (teacher-quality-ceiling finding).
- **Referee** — a **constant** anchor set. Measures true model progress; never changes. (Holds the
  "hold one axis fixed" principle, findings f005/f007. Without it you can't tell "model improved" from
  "benchmark got harder.")

## The two losses

- **Benchmark loss (adversary minimizes):** an example is *valuable* iff **T correct (per rubric) ∧ S_t wrong**.
  Aggregate difficulty signal: `gap = acc(T) − acc(S_t)` on a held-out hard eval. The T-correct gate
  auto-discards mislabeled / ill-posed examples (where even T fails) — the #1 risk we keep hitting.
- **Model loss (student minimizes):** training loss, but **tracked** as F1 on the **frozen anchor**
  (never on the moving benchmark). Progress is real only if anchor-F1 holds/rises while the gap closes.
- **Equilibrium:** the generator can't produce correctly-labeled examples S fails → `S ≈ T` on the
  solvable region.

## State / artifacts

```
data/legal/anchor/                 # FROZEN referee — built once, never changed
   gold.jsonl                      #   snapshot of pre-hardening Legal test (known-good labels)
   crosssource.jsonl               #   CUAD/wiki/held-out-ToS (different generator → anti-Goodhart)
data/legal/coevolve/
   pool_rN.jsonl                   # round-N candidate pool (oversampled, labeled, seed_id'd)
   frontier_rN.jsonl               # kept: T-correct ∧ S-wrong  (the training signal)
   human_queue_rN.jsonl            # both-wrong + T-wrong/S-right  → human audit
   train_hard.jsonl                # ACCUMULATED frontier train split (rounds 1..N)
   eval_hard.jsonl                 # ACCUMULATED frontier eval split (leakage-safe by seed)
results/coevolve_legal.jsonl       # per-round metrics log (dashboard reads this)
benchmark versions: legal-hard-rN  # snapshot_dataset each round (versioned, ablatable)
checkpoints: legal_coev_rN         # tinker:// per round
```

## Components

Reuse: `data/legal/generate.py`, `eval/run_eval.py`, `train/build_sft_data.py`, `train/sft.py`,
`scripts/snapshot_dataset.py`, `scripts/add_finding.py`, the supervisor billing precheck.

New:
1. **`scripts/label_oracle.py`** — few-shot Qwen3.6-27B over a pool → per-example preds
   (`id,label,pred,correct`). Reuse `domains.legal.SPEC.build_messages_fewshot`. (Can't reuse `run_eval`
   for the oracle: its prompt is zero-shot.)
2. **`scripts/coevolve.py`** — the round/loop controller.
3. **`generate.py --pool N`** flag — emit one unsplit, oversampled, `seed_id`'d candidate file
   (CF-heavy + intent/near-boundary + real draw) instead of writing train/val/test.

## Frontier gate (per candidate)

Join oracle T preds and student S preds by `id`. Each candidate has a **trusted label**
(synthetic = correct by construction; real = metadata/heuristic, so T also *verifies* it):

| oracle T | student S | bucket | action |
|---|---|---|---|
| ✓ correct | ✗ wrong | **FRONTIER** | hard-but-solvable → train_hard + eval_hard |
| ✓ correct | ✓ correct | learned | discard |
| ✗ wrong | ✓ correct | suspect | → human_queue (possible mislabel / T weakness) |
| ✗ wrong | ✗ wrong | **shared blind spot** | → human_queue (same-family blind spot; most valuable for humans) |

- Synthetic candidates: label known → T-gate is a pure **solvability filter** (drop ill-posed items).
- Real candidates: T also **verifies** the heuristic label (agreement check).
- We **never train** on examples we can't validate (both-wrong → human only).

## Per-round algorithm (`coevolve.py`)

```python
def round(N, S_ckpt):                       # S_ckpt = current student (frozen 4B at N=0)
    billing_ok() or abort()
    # 1. MINE candidates (oversample; diversity-capped to avoid mode collapse)
    run("data/legal/generate.py --pool 1500 --seed N --out pool_rN.jsonl")   # synthetic CF-heavy
    append_real_draw("pool_rN.jsonl", k=300)                                  # fresh real rows
    dedup(pool_rN, against=[anchor, train_hard, eval_hard, prior pools])      # contamination control
    # 2. SCORE pool with oracle T (few-shot 27B) and student S
    run("label_oracle.py --data pool_rN.jsonl --out preds_T_rN.jsonl")
    run(f"run_eval.py --data pool_rN.jsonl --model {S_ckpt} --domain legal --tag S_rN")
    # 3. BUCKET
    frontier    = [x for x in pool if T[x].correct and not S[x].correct]
    human_queue = [x for x in pool if not T[x].correct]
    log(f"yield={len(frontier)} oracle_acc={acc(T)} student_acc={acc(S)} gap={acc(T)-acc(S)}")
    # 4. SPLIT frontier leakage-safe by seed_id; eval_hard also gets a cross-source slice
    tr, ev = split_by_seed(frontier, fracs=(0.7, 0.3))
    accumulate(train_hard, tr); accumulate(eval_hard, ev)
    snapshot_dataset(version=f"legal-hard-r{N}")
    # 5. TRAIN S_{N+1} on base_train + train_hard (keep old frontier → no forgetting)
    build_sft_data(domain="legal", extra=train_hard)
    new_ckpt = sft(name=f"legal_coev_r{N}", epochs=2)
    # 6. MEASURE on frozen referee + moving hard eval
    anchor_f1 = run_eval(new_ckpt, data="anchor/gold + anchor/crosssource")   # TRUE progress
    hard_f1   = run_eval(new_ckpt, data="eval_hard")
    new_gap   = acc(T, eval_hard) - acc(new_ckpt, eval_hard)
    log_round(N, anchor_f1, hard_f1, len(frontier), new_gap)
    add_finding(...); git_commit(f"coevolve legal r{N}")
    return new_ckpt, metrics

# controller loop
S = "Qwen/Qwen3.5-4B"; base = anchor_f1(S)
for N in range(MAX_ROUNDS):                 # MAX_ROUNDS = 4
    S, m = round(N, S)
    if m.anchor_f1 < base - 0.02:  STOP("Goodhart: anchor regressed")           # kill
    if m.frontier_yield < 30:      STOP("converged: S matches T on solvable")   # success
    if human_audit_disagree(sample=40) > 0.15: STOP("oracle too weak")          # kill
    base = max(base, m.anchor_f1)
```

## Metrics (`results/coevolve_legal.jsonl` → dashboard sub-tab)

| round | student | anchor-F1 (frozen) | hard-eval-F1 | frontier yield | gap acc(T)−acc(S) | audit-disagree |
|---|---|---|---|---|---|---|
| 0 | frozen 4B | baseline | — | N found | large | — |
| 1 | coev_r0 | hold/↑ | ↑ | shrinking | shrinking | <15% |
| … | | | | →0 | →0 | |

Healthy run: anchor-F1 flat-or-up, gap & yield → 0. Equilibrium when yield < 30.

## Guards (mapped to the tricky parts)

- **Label-validity cheat** (the adversary's cheat code) → T-correct gate + 40-example human audit/round;
  kill if disagreement >15%.
- **Teacher ceiling** → accepted explicitly; both-wrong → human_queue, never trained on. Loop drives S→T,
  not past it.
- **Same-family blind spot** → human_queue is the instrument; optional: re-score human_queue with a
  **different-family oracle (Claude)** to surface shared Qwen blind spots cheaply.
- **Mode collapse** → `--pool` caps per CF-frame/clause-body + tracks frame entropy; dedup vs all prior
  pools + anchor.
- **Goodhart / train-eval leakage** → frozen anchor + cross-source eval_hard slice (different generator
  than train_hard); leakage-safe `seed_id` split; benchmark versioned per round.

## Config (pilot)

`pool=1500 synthetic + 300 real`, `MAX_ROUNDS=4`, `oracle=Qwen3.6-27B few-shot(8)`, `SFT epochs=2 rank=32`,
`frontier split 70/30`. Kill: anchor −2pts. Converge: yield<30. Audit: >15% disagree.
Cost driver = 27B oracle inference on ~1800 candidates/round × 4 rounds.

## Build order

1. `data/legal/anchor/` — freeze gold + cross-source (reuse existing files).
2. `scripts/label_oracle.py` — few-shot 27B pred writer.
3. `generate.py --pool` flag.
4. `scripts/coevolve.py` — controller.
5. **Dry-run round 0 only** (mine + bucket, *no* training) → eyeball frontier + human_queue to confirm
   the gate yields sensible, correctly-labeled hard examples **before** spending on the full loop. This is
   the real de-risk: if the frontier is mislabeled/ambiguous junk, the oracle gate is too weak — fix that
   first.

## Honest cost/benefit

- vs. manual hardening: automates blind-spot discovery + principled difficulty filter (the gap). Real win.
- vs. plain OPD from T: smaller than it looks — OPD already distills T into S on the student's rollouts.
  Extra value = active-learning efficiency (spend supervision on S's *specific* failures) + a genuinely
  hard, versioned eval set as a byproduct. Asymptote is the same (S→T) because of the teacher ceiling.
- Net: worth a small, well-instrumented test; bounded by T and blind to shared-family errors (hence the
  human_queue + optional cross-family oracle).
