"""
Fetch + label REAL Marketing-domain data from Hugging Face (bulk = dataset provenance; a
Claude/human judge verifies a sample separately). Writes data/marketing/real/ (gitignored).

The marketing policy line is OUR outbound claim/copy/term vs everything else (third-party
opinion, market fact, internal logistics). The hard part: customer reviews are dense with the
same promotional superlatives as real ad copy ("best ever", "10x better") yet are third-party
opinions, not claims we publish -> NEGATIVE. So we pick sources on opposite sides of that line,
making this a genuine over-trigger stress test (parallel to Legal's LEDGAR-vs-ToS):

  - product descriptions  -> POSITIVE. A seller's product description IS outbound marketing copy
       / claims. subcategory=brand_campaign_copy (or pricing_promo / efficacy if the weak regex
       fires). confidence=med (provenance, verify).
  - amazon_polarity reviews -> NEGATIVE. Third-party customer opinion about a product, not a
       claim WE make. The HARD negative — superlative-dense, opposite label. subcategory=
       market_fact-ish 'opinion'. confidence=med.
  - ag_news                -> NEGATIVE. General news = third-party market/industry fact (easy
       negative + diversity). subcategory=market_fact. confidence=high.

Categories with no clean public source (verified internal campaign logistics, real PR subject to
FTC action) stay synthetic-only. Logged, not hidden.

Run:  python data/marketing/fetch_real.py --limit 500 --seed 0
"""
import argparse
import json
import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
from domains.marketing import SPEC as _MKT_SPEC  # weak promo/superlative regex for subcategory only

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "marketing" / "real"

PRODUCT_CANDIDATES = [
    ("ckandemir/amazon-products", None, ["Description"]),
    ("bprateek/amazon_product_description", None, ["About Product", "Description"]),
]
REVIEW_CANDIDATES = [
    ("amazon_polarity", None, ["content", "title"]),
    ("mteb/amazon_polarity", None, ["text", "content"]),
    ("imdb", None, ["text"]),
]


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def truncate(text, n=2000):
    text = (text or "").strip()
    return text if len(text) <= n else text[:n]


def _pos_subcat(text):
    # use the weak marketing regex only to refine the positive subcategory (not the label).
    _, matches = _MKT_SPEC.regex_predict(text)
    keys = " ".join(matches).lower() if matches else ""
    if "promo_pct" in keys or "promo_code" in keys:
        return "pricing_promo"
    if "superlative" in keys:
        return "efficacy_competitive_claim"
    return "brand_campaign_copy"


def _first_field(row, fields):
    return next((truncate(row.get(f)) for f in fields if row.get(f)), "")


def from_products(limit, rng):
    from datasets import load_dataset
    last = None
    for rid, cfg, fields in PRODUCT_CANDIDATES:
        try:
            ds = load_dataset(rid, cfg, split="train", streaming=True)
            out = []
            for row in ds:
                text = _first_field(row, fields)
                if not text or len(text) < 40:
                    continue
                out.append((text, 1, _pos_subcat(text), "med", "metadata", "product_copy", rid))
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            last = f"{rid}: {type(e).__name__}"
            continue
    raise RuntimeError(f"no product-description source available (last: {last})")


def from_reviews(limit, rng):
    from datasets import load_dataset
    last = None
    for rid, cfg, fields in REVIEW_CANDIDATES:
        try:
            ds = load_dataset(rid, cfg, split="train", streaming=True)
            out = []
            for row in ds:
                text = _first_field(row, fields)
                if not text or len(text) < 40:
                    continue
                # third-party opinion about a product -> NEGATIVE (not a claim we publish).
                out.append((text, 0, "opinion_review", "med", "metadata", "review", rid))
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            last = f"{rid}: {type(e).__name__}"
            continue
    raise RuntimeError(f"no review source available (last: {last})")


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
                out.append((text, 0, "market_fact", "high", "metadata", "news", rid))
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception as e:  # noqa: BLE001
            last = f"{rid}: {type(e).__name__}"
            continue
    raise RuntimeError(f"news source unavailable (last: {last})")


SOURCES = {"products": from_products, "reviews": from_reviews, "news": from_news}


def load_synthetic_hashes():
    hashes = set()
    for sp in ("train", "val", "test"):
        fp = ROOT / "data" / "marketing" / f"{sp}.jsonl"
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
    print("Fetching marketing sources (failures are skipped):")
    for name, fn in SOURCES.items():
        try:
            produced = fn(args.limit, rng)
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
                id=f"real-{counter:05d}", seed_id=f"real-marketing-{name}-{counter}", domain="marketing",
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

    print(f"\nTotal real marketing rows kept: {n}  (deduped vs {len(syn)} synthetic)")
    for name in ("train", "val", "test", "test_realistic", "test_balanced"):
        items = splits.get(name) or {"test_realistic": realistic, "test_balanced": balanced}[name]
        p = sum(r["label"] for r in items)
        print(f"  {name:15} {len(items):4} ({p} pos / {len(items)-p} neg)")
    print("subcategory:", dist(rows, "subcategory"))
    print("confidence:", dist(rows, "confidence"))
    print("hf_dataset:", dist(rows, "hf_dataset"))


if __name__ == "__main__":
    main()
