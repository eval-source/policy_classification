"""
Fetch + label REAL Legal-domain data from Hugging Face (bulk = dataset-provenance metadata; a
Claude/human judge verifies a sample separately). Writes data/legal/real/ (gitignored).

The crux of the legal policy is binding/negotiated instrument vs public boilerplate — and both
share the same legalese ("the parties agree", "hereinafter"), so regex is useless (matching
domains.legal, regex_patterns={}). We pick real sources that sit on opposite sides of that exact
line, which makes this a real over-trigger stress test (legal favors RECALL → expect the model to
over-fire on boilerplate, hurting specificity on the ToS negatives):

  - lex_glue/ledgar       -> POSITIVE. LEDGAR = contract provisions extracted from SEC EDGAR
       material contracts (negotiated, party-specific instruments). subcategory mapped from the
       provision label (indemnification/confidentiality/...). confidence=med (provenance, verify).
  - lex_glue/unfair_tos   -> NEGATIVE. Sentences from online Terms-of-Service = public boilerplate;
       the policy explicitly excludes EULA/ToS/privacy. subcategory=eula_tos_privacy. The HARD
       negative — same legalese, opposite label. confidence=med.
  - ag_news               -> NEGATIVE. General news (world/sports/biz/sci) = routine non-legal
       content (the "legal_news"/core negative + diversity). confidence=high.

Categories with no clean public source — privileged/litigation material, executed NDAs/MSAs with
real party names (privacy/availability) — stay synthetic-only. Logged, not hidden.

Run:  python data/legal/fetch_real.py --limit 500 --seed 0
"""
import argparse
import json
import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "legal" / "real"

# map a LEDGAR provision-label substring -> our positive subcategory (domains.legal POS_KIND).
LEDGAR_SUBCAT = [
    (("confidential", "non-disclosure", "nondisclosure"), "nda_confidentiality"),
    (("indemnif", "liabilit", "warrant"), "indemnity_liability"),
    (("litigation", "dispute", "arbitration", "waiver of jury"), "privilege_litigation"),
    (("compliance", "covenant"), "compliance_obligation"),
]
# lex_glue is mirrored under a few ids/configs across versions; probe in order.
LEX_GLUE_IDS = ("coastalcph/lex_glue", "lex_glue")


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def truncate(text, n=2000):
    text = (text or "").strip()
    return text if len(text) <= n else text[:n]


def _load_lex_glue(config):
    from datasets import load_dataset
    last = None
    for rid in LEX_GLUE_IDS:
        try:
            return load_dataset(rid, config, split="train", streaming=True), rid
        except Exception as e:  # noqa: BLE001
            last = f"{rid}/{config}: {type(e).__name__}"
            continue
    raise RuntimeError(f"lex_glue/{config} unavailable (last: {last})")


def _ledgar_label_names():
    # the streaming row gives an int label; get the ClassLabel names for the human-readable type.
    from datasets import load_dataset
    for rid in LEX_GLUE_IDS:
        try:
            info = load_dataset(rid, "ledgar", split="train", streaming=True).features
            return info["label"].names
        except Exception:  # noqa: BLE001
            continue
    return None


def from_ledgar(limit, rng):
    ds, rid = _load_lex_glue("ledgar")
    names = _ledgar_label_names()
    out = []
    for row in ds:
        text = truncate(row.get("text"))
        if not text or len(text) < 40:
            continue
        lab_idx = row.get("label")
        lname = (names[lab_idx] if names and isinstance(lab_idx, int) and lab_idx < len(names)
                 else str(lab_idx)).lower()
        sub = "contract"  # default: a clause within a negotiated instrument
        for keys, s in LEDGAR_SUBCAT:
            if any(k in lname for k in keys):
                sub = s
                break
        out.append((text, 1, sub, "med", "metadata", "clause", f"{rid}/ledgar"))
        if len(out) >= limit:
            break
    return out


def from_unfair_tos(limit, rng):
    ds, rid = _load_lex_glue("unfair_tos")
    out = []
    for row in ds:
        text = truncate(row.get("text"))
        if not text or len(text) < 30:
            continue
        # public ToS boilerplate -> NEGATIVE regardless of unfairness label (policy excludes ToS).
        out.append((text, 0, "eula_tos_privacy", "med", "metadata", "tos", f"{rid}/unfair_tos"))
        if len(out) >= limit:
            break
    return out


def from_news(limit, rng):
    from datasets import load_dataset
    last = None
    for rid in ("ag_news", "fancyzhx/ag_news"):
        try:
            ds = load_dataset(rid, split="train", streaming=True)
            out = []
            for row in ds:
                text = truncate(row.get("text"))
                if not text or len(text) < 40:
                    continue
                # general news = routine non-legal content (core/legal-news negative).
                out.append((text, 0, "legal_news", "high", "metadata", "news", rid))
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            last = f"{rid}: {type(e).__name__}"
            continue
    raise RuntimeError(f"news source unavailable (last: {last})")


# ---- v2 HARD negatives: confusable legalese the policy clearly/defensibly EXCLUDES. These exist
# to push the frozen baseline down into the discriminative band (over-triggering is the failure). ----

def from_privacy(limit, rng):
    # privacy-policy prose (legalbench). Public boilerplate -> NEGATIVE (policy excludes privacy/ToS).
    from datasets import load_dataset
    ds = load_dataset("nguha/legalbench", "privacy_policy_qa", split="train", streaming=True)
    out, seen = [], set()
    for row in ds:
        text = truncate(row.get("text"))
        h = norm(text)
        if not text or len(text) < 40 or h in seen:  # same passage repeats across questions
            continue
        seen.add(h)
        out.append((text, 0, "eula_tos_privacy", "med", "metadata", "privacy", "legalbench/privacy_policy_qa"))
        if len(out) >= limit:
            break
    return out


def from_legal_advice(limit, rng):
    # r/legaladvice posts: people describing legal situations in casual register, dense with legal
    # terms but NOT a binding instrument -> NEGATIVE (commentary/discussion). Casual-register bonus.
    from datasets import load_dataset
    ds = load_dataset("jonathanli/legal-advice-reddit", split="train", streaming=True)
    out = []
    for row in ds:
        text = truncate(((row.get("title") or "") + " " + (row.get("body") or "")).strip())
        if not text or len(text) < 80 or text.lower() in ("[removed]", "[deleted]"):
            continue
        out.append((text, 0, "legal_news", "med", "metadata", "advice", "jonathanli/legal-advice-reddit"))
        if len(out) >= limit:
            break
    return out


_OBLIG = re.compile(r"\b(shall|must|may not|is liable|indemnif|liabilit|warrant|obligat)\b", re.I)


def from_legislation(limit, rng):
    # US congressional bills (billsum) — but extract a single OBLIGATION SENTENCE ("X shall ...")
    # rather than the bill header. At the sentence level a statutory obligation is genuinely
    # confusable with a contract clause (the model should over-trigger), which is the point: it is
    # public LEGISLATION, not a NEGOTIATED party-specific instrument -> NEGATIVE. Deliberate RUBRIC
    # CALL: low confidence => excluded from SFT training, kept in the eval as a hard over-trigger
    # probe. Flagged, not hidden.
    from datasets import load_dataset
    ds = load_dataset("FiscalNote/billsum", split="train", streaming=True)
    out = []
    for row in ds:
        body = re.sub(r"\s+", " ", (row.get("text") or "")).strip()
        # split into rough sentences; keep a clause-length obligation sentence
        sent = next((s.strip() for s in re.split(r"(?<=[.;])\s+", body)
                     if 60 <= len(s.strip()) <= 320 and _OBLIG.search(s)
                     and not s.strip().upper().startswith("SECTION")), None)
        if not sent:
            continue
        out.append((sent, 0, "legislation", "low", "metadata", "statute", "FiscalNote/billsum"))
        if len(out) >= limit:
            break
    return out


SOURCES = {"ledgar": from_ledgar, "unfair_tos": from_unfair_tos, "news": from_news,
           "privacy": from_privacy, "legal_advice": from_legal_advice, "legislation": from_legislation}
# per-source caps so the HARD negatives dominate the negative pool (-> harder frozen baseline).
CAPS = {"ledgar": 500, "unfair_tos": 450, "news": 120,
        "privacy": 300, "legal_advice": 300, "legislation": 300}


def load_synthetic_hashes():
    hashes = set()
    for sp in ("train", "val", "test"):
        fp = ROOT / "data" / "legal" / f"{sp}.jsonl"
        if fp.exists():
            for line in fp.open():
                if line.strip():
                    hashes.add(norm(json.loads(line)["text"]))
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="rows pulled per source (pre-dedup)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-rate", type=float, default=0.20, help="positive rate for realistic test view")
    ap.add_argument("--train-min-conf", default="med", choices=["low", "med", "high"])
    args = ap.parse_args()
    rng = random.Random(args.seed)

    syn = load_synthetic_hashes()
    seen = set(syn)
    rows, counter = [], 0
    print("Fetching legal sources (failures are skipped):")
    for name, fn in SOURCES.items():
        try:
            produced = fn(min(args.limit, CAPS.get(name, args.limit)), rng)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:10} SKIPPED: {type(e).__name__}: {str(e)[:120]}")
            continue
        kept = 0
        for (text, lab, sub, conf, lsrc, fmt, hf) in produced:
            h = norm(text)
            if not h or h in seen:
                continue
            seen.add(h)
            rows.append(dict(
                id=f"real-{counter:05d}", seed_id=f"real-legal-{name}-{counter}", domain="legal",
                text=text, label=lab, subcategory=sub, difficulty="natural",
                hardening="real", pair_id=None, source="real", format=fmt, noisy=False,
                label_source=lsrc, confidence=conf, hf_dataset=hf,
            ))
            counter += 1
            kept += 1
        print(f"  {name:10} produced={len(produced):4} kept(after dedup)={kept}")

    if not rows:
        print("No rows fetched (network/gating?). Aborting.")
        return

    rng.shuffle(rows)
    n = len(rows)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    splits = {"train": rows[:n_tr], "val": rows[n_tr:n_tr + n_va], "test": rows[n_tr + n_va:]}

    order = {"low": 0, "med": 1, "high": 2}
    thr = order[args.train_min_conf]
    splits["train"] = [r for r in splits["train"] if order[r["confidence"]] >= thr]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    test = splits["test"]
    pos = [r for r in test if r["label"] == 1]
    neg = [r for r in test if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    k_pos = max(1, int(args.pos_rate / (1 - args.pos_rate) * len(neg)))
    realistic = neg + pos[:k_pos]
    rng.shuffle(realistic)
    m = min(len(pos), len(neg))
    balanced = pos[:m] + neg[:m]
    rng.shuffle(balanced)
    for name, items in [("test_realistic", realistic), ("test_balanced", balanced)]:
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    def dist(items, key):
        d = {}
        for r in items:
            d[r[key]] = d.get(r[key], 0) + 1
        return dict(sorted(d.items()))

    print(f"\nTotal real legal rows kept: {n}  (deduped vs {len(syn)} synthetic)")
    for name in ("train", "val", "test", "test_realistic", "test_balanced"):
        items = splits.get(name) or {"test_realistic": realistic, "test_balanced": balanced}[name]
        p = sum(r["label"] for r in items)
        print(f"  {name:15} {len(items):4} ({p} pos / {len(items)-p} neg)")
    print("subcategory:", dist(rows, "subcategory"))
    print("confidence:", dist(rows, "confidence"))
    print("hf_dataset:", dist(rows, "hf_dataset"))


if __name__ == "__main__":
    main()
