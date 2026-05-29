# Interview Exercise — Policy Classification

Welcome, and thanks for taking the time. This is a **hands-on** exercise. We have given you a
working environment with the tooling and credentials already set up (see **Environment**
below) — we want you to **actually build datasets, train models, and run experiments**, not
just describe what you would do.

That said, we care most about **how you think**: the trade-offs you weigh, what you try
first, what you expect to fail, and how you find out either way. State your assumptions, and
design experiments that would prove you right or wrong. We would much rather see a small,
well-controlled experiment with a clear conclusion than a large untested pipeline. Keep a
running log of what you tried and what you found — that log is a big part of what we will
discuss.

---

## Environment & Tooling

The experiment environment is provisioned for you. A separate **`.md` file with all
passwords, API keys, and account credentials** is provided alongside this brief — use it for
every login below.

- **Tinker** — for fine-tuning / training / deploy (SFT and on-policy distillation loops,
  LoRA, and a sampling/deploy path), with base-model access to **Qwen3 4B**.
  - Docs: https://tinker-docs.thinkingmachines.ai
  - Cookbook (recipes for SFT, distillation, RLHF, etc.): https://github.com/thinking-machines-lab/tinker-cookbook
  - **Check the Tinker cookbook for the training recipes you need** — it has worked examples
    of the SFT and distillation loops you'll be building on.
- **Hugging Face** — for sourcing and storing datasets. Account is configured.
- **Claude Code** is installed and included — **use Claude.** It's there to help you move
  fast: feel free to use it for **dataset curation**, writing the eval harness, and driving
  training/ablation runs.
- **Warp** — use it to **manage your agents** (terminal / agent orchestration).
- **OpenAI API key** — if you need one (e.g. for synthetic-data generation), **ping
  tian@classie.ai** and we'll get it to you.
- **Evaluation** — your choice of stack; install whatever you need.

You will not have time to do everything below exhaustively — that is intended. **Scope
deliberately**: pick the experiments that most reduce your uncertainty, run them, and tell us
what you learned and what you would do next with more time.

---

## Background

At Classie we classify text against **policies** — natural-language descriptions of things
that should be flagged. The core question for a single policy is binary:

> Given a policy and a piece of text, does the text **fall into** that policy — i.e. does it
> **trigger**?

You will work three policy domains, so your approach has to hold up across genuinely
different distributions. For each, we sketch what is **in scope** (should trigger) and what is
**out of scope** (should not) — note that the out-of-scope items are deliberately *near* the
boundary, which is exactly where the hard cases live.

**1. Generic legal** — binding/negotiated legal instruments and obligations.
- *In scope (triggers):* NDAs and confidentiality agreements; contracts/MSAs/SOWs/DPAs;
  indemnification, liability, and warranty clauses; privilege and litigation/dispute language;
  contractual compliance obligations.
- *Out of scope (does not trigger):* a website EULA, terms-of-service, or privacy-policy page
  (public boilerplate); general legal news or commentary; casual references to "legal" (the
  team, "legally speaking"); internal logistics about the legal department.

**2. Marketing / GTM** — outbound messaging, claims, and commercial terms.
- *In scope (triggers):* product/efficacy and competitive claims (especially unsubstantiated);
  pricing, discount, and promotional terms; brand-voice/positioning and campaign copy;
  press/PR statements subject to advertising regulation.
- *Out of scope (does not trigger):* internal campaign scheduling, calendar invites, and team
  logistics; third-party market-research or industry-trend facts; a generic mention of the
  word "marketing"; someone's personal opinion about an ad.

**3. Information Technology (IT dept)** — sensitive technical, security, and access content.
- *In scope (triggers):* credentials and secrets (API keys, passwords, tokens, connection
  strings); access-control grants/changes; security policy; infrastructure config containing
  sensitive data; vulnerability/incident details; PII handling.
- *Out of scope (does not trigger):* a routine helpdesk ticket ("my monitor is broken");
  a public tutorial or code snippet that contains **no** secrets; a bare mention of a tool
  or vendor name; general "my laptop is slow" support chatter.

### The task

Train a base model — **Qwen3 4B** — to perform this policy-trigger classification, and back
your design choices with experiments you actually run.

The exercise has three parts. Roughly **Part 1 ≈ 20%, Part 2 ≈ 25%, Part 3 ≈ 55%** of your
time; Part 3 is where we will dig deepest. The parts feed each other — your dataset and eval
exist to serve the training experiments.

---

## Part 1 — Build the Dataset

> **Given a policy, build a dataset for it.** Produce a real, usable dataset for at least one
> domain (more if time allows), and be ready to show the data and explain the choices behind
> it.

Considerations to drive your design — extend this list and justify what you do:

- **Positive / negative examples.** Choose a class balance, and decide whether training
  balance should match the (rare) production base rate. Generate **hard negatives**
  (topically adjacent but non-triggering) and **counterfactual pairs** (minimal edits that
  flip the label) — these are usually where the signal is.
- **Mixing in non-synthetic (real) data.** Decide what real data buys you over synthetic and
  where to spend it (training, eval, or both); pull suitable datasets from HF where they
  exist. Handle sourcing/licensing/PII, and keep the distribution close to realistic traffic.
- **Other directions worth addressing** (pick the ones that matter most for your time budget):
  - Decomposing a policy into sub-clauses and labelling at that granularity.
  - The coverage axes you deliberately vary when generating synthetic data
    (phrasing/register, length, format, persona, language, obfuscation), and how you avoid
    mode collapse and teacher-model artifacts.
  - Your labelling rubric and how you handle ambiguous cases.
  - Deduplication and contamination control against your evaluation set.
  - Overlapping / multi-label policies (e.g. legal-confidentiality vs. IT-secrets).

Each domain has its own character — legal is precise and formal with costly misses; IT mixes
pattern-matchable signal (credentials) with semantic intent; marketing lives in gray areas of
tone and claims. We want to see how your data design adapts to that.

**Produce:** a dataset (with splits — see Part 3), a short note on how you generated/sourced
it, and a few example rows that illustrate your hard negatives and counterfactuals.

---

## Part 2 — Build the Evaluation

> **Design and implement the evaluation, then use it.** Coverage is the central question:
> what are you measuring, and have you measured the things that will actually break?

Considerations:

- **What "coverage" means here** — across policy sub-clauses, domains, difficulty (easy vs.
  hard negatives), data sources (synthetic vs. real), formats/registers, and failure modes
  (over-triggering vs. under-triggering). Build a held-out eval set that is sliceable along
  these axes and report per-slice, not just one aggregate.
- **Held-out examples vs. held-out policies.** Decide whether you are measuring generalization
  to new *examples* of trained policies, or to *policies the model has never seen* — and build
  the eval set that actually tests it.
- **Metrics and operating point.** Pick metrics and report them over slices; choose the
  operating point given that the cost of a false negative vs. a false positive differs by
  domain. Look at calibration, and at confidence intervals — your eval set will be small
  enough that noise matters.
- **A mixed scoring stack — human + LLM-as-judge + regexes.** In practice we score with a
  blend of all three, and we want you to build with that in mind. Which signals are cheap and
  deterministic enough for **regexes / rule-based checks** (e.g. literal secret patterns for
  IT)? Where is an **LLM-as-judge** the right tool — how do you prompt it, validate it, and
  guard against its biases (including that the judge may share the model family you are
  training)? Where do you spend scarce **human labels** — ground truth, auditing the judge,
  adjudicating disagreements? Measure agreement between the three and decide what to do when
  they conflict.
- **Trustworthiness of the eval itself** — contamination between train and eval, and how your
  offline numbers would relate to behaviour after deployment.

**Recommended reading (evaluation methodology):**
- Sang T. Truong, Andreas Haupt, Sanmi Koyejo — *Machine Learning from Human Preferences*
  (Stanford CS329H): https://mlhp.stanford.edu/Machine-Learning-from-Human-Preferences.pdf
- Schaeffer, Miranda, Koyejo — *Are Emergent Abilities of Large Language Models a Mirage?*:
  https://arxiv.org/abs/2304.15004 — on how metric choice can manufacture apparent jumps in
  capability, and why that matters for how you measure progress.

**Produce:** a runnable eval harness, a held-out eval set, and a baseline number (e.g. the
frozen Qwen3 4B with a zero/few-shot prompt) that everything in Part 3 must beat.

---

## Part 3 — Train with Tinker (and run the experiments)

> **Train the model.** The **base** approach is **SFT**. The **extended** approach is
> **on-policy distillation (OPD)** — the student samples its own completions, and a teacher
> provides dense, per-token supervision on those on-policy rollouts. Run real training jobs
> and measure them against your Part 2 eval.
>
> **Hint:** you can use **chain-of-thought to extend the reasoning** for these
> classifications — have the model reason before it emits a label, and treat that reasoning as
> something you can supervise and distill.

Do the work and report results:

- **Train / test split.** Implement a split that avoids leakage (paraphrases and
  counterfactual variants of the same seed kept together, real vs. synthetic, by-policy,
  by-source, temporal where relevant) and supports the held-out-policy test from Part 2.
- **Train, in stages, against a baseline.** Start from the frozen-model baseline, then **SFT**,
  then **CoT-style reasoning**, then **on-policy distillation** — running each on Tinker and
  measuring the delta on your eval. Show your work at each step.
- **Pros and cons, demonstrated.** Show what on-policy distillation buys you over plain SFT on
  teacher traces (training on the states the student actually visits; dense supervision vs.
  sparse reward), and where it costs you or fails (teacher dependence and quality ceiling,
  collapse, compute). Decide, with evidence, when it is and isn't worth it here.
- **Run ablations.** This is the heart of the exercise. Pick and run the ablations that most
  reduce your uncertainty — e.g. CoT vs. no-CoT (and its latency cost), hard negatives in/out,
  synthetic-only vs. mixed, SFT-only vs. SFT→OPD, per-policy adapter vs. a unified
  multi-policy model, a data-scale curve, LoRA rank / decoding / threshold sensitivity, and
  held-out-policy generalization. Keep each ablation a clean, one-variable-at-a-time
  comparison against a fixed baseline, and present the results as a small table with your
  reading of them.
- **Tinker mechanics.** Use the SFT loop, the on-policy distillation loop, and the
  deploy/sampling path for real. Decide between one base model with swappable per-policy LoRA
  adapters and a single unified model, and justify it with what you observe.

**Produce:** at least one trained model that beats your baseline, an ablation table with
conclusions, and the deployed/sampling path working end-to-end on at least one policy.

---

## What to deliver

By the end you should have, and be ready to walk us through:

- The **datasets** you built (with splits) and a short note on how.
- A **runnable eval harness** and your headline + per-slice numbers, baseline included.
- **Trained models** and an **ablation table** with your conclusions.
- A short **experiment log**: what you tried, what worked, what didn't, and what you would do
  next with more time.

We are not expecting all of this to be finished or polished. We are looking for sound
judgment under a time budget: good baselines, controlled experiments, honest reading of
noisy results, and clear reasoning about the trade-offs. Bring your findings — and the dead
ends — and we will dig in together.
