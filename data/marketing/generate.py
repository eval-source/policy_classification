"""
Marketing/GTM synthetic generator (mirrors data/legal/generate.py structure).

POSITIVE: efficacy_competitive_claim, pricing_promo, brand_campaign_copy, press_pr.
NEGATIVE: internal_logistics, market_fact, generic_marketing, personal_opinion_ad.
Hard families: counterfactual pairs (external claim vs internal logistics; unsubstantiated claim
vs third-party market fact; copy vs opinion; press vs news-commentary), intent_only positives,
near_boundary negatives, casual register, typo noise.

Run:  python data/marketing/generate.py --variant v1 --seed 0 --out data/marketing
"""
import argparse
import json
import random
import re
from pathlib import Path

PRODUCTS = ["CloudSync", "DataForge", "PixelPay", "FlowCRM", "NimbusAI", "BrightDesk"]
PEOPLE = ["Alice", "Bob", "Priya", "Diego", "Mei", "Sam", "Jordan", "Chen", "Fatima", "Lars"]
FORMATS = ["doc", "email", "slack", "adcopy", "press"]


def wrap(rng, fmt, body, subject="update"):
    who = rng.choice(PEOPLE)
    if fmt == "slack":
        return f"{who.lower()}: {body}"
    if fmt == "email":
        return f"Subject: {subject}\nHi team,\n\n{body}\n\nThanks,\n{who}"
    if fmt == "press":
        return f"FOR IMMEDIATE RELEASE\n{body}"
    return body  # doc / adcopy


def pos_claim(rng, fmt):
    p = rng.choice(PRODUCTS)
    t = rng.choice([
        f"New tagline: \"{p} is 3x faster than any competitor — guaranteed.\"",
        f"{p} is clinically proven to reduce churn by 50%. The only platform trusted by 10,000+ teams.",
        f"Headline claim: {p} outperforms [Competitor] on every benchmark — the #1 choice for enterprise.",
    ])
    return wrap(rng, fmt, t, subject="claim"), "efficacy_competitive_claim"


def pos_promo(rng, fmt):
    t = rng.choice([
        "Promo: 40% off all annual plans through Friday — use code LAUNCH40 at checkout.",
        "Black Friday: save $200 on the Pro tier, plus a free onboarding session. Limited time only.",
        "Refer a friend and you BOTH get 3 months free — offer ends Sunday at midnight.",
    ])
    return wrap(rng, fmt, t, subject="promo"), "pricing_promo"


def pos_brand(rng, fmt):
    p = rng.choice(PRODUCTS)
    t = rng.choice([
        f"We're positioning {p} as the premium, enterprise-grade option — lean into 'trusted by Fortune 500'.",
        f"Brand voice for the {p} campaign: confident, no jargon, outcome-first. Hero line: 'Ship faster. Sleep better.'",
        f"Campaign copy: \"{p} — the future of work, today.\" Use across the landing page and paid social.",
    ])
    return wrap(rng, fmt, t, subject="campaign"), "brand_campaign_copy"


def pos_press(rng, fmt):
    p = rng.choice(PRODUCTS)
    t = rng.choice([
        f"{p} today announced the industry's first AI-native CRM, setting a new standard for sales teams worldwide.",
        f"Press statement: \"{p} has achieved record growth and now serves more customers than any rival in the space.\"",
        f"PR draft: \"{p} is the most secure platform on the market\" — pending legal/regulatory review before release.",
    ])
    return wrap(rng, "press", t, subject="press release"), "press_pr"


POS = [pos_claim, pos_promo, pos_brand, pos_press]


def neg_logistics(rng, fmt):
    t = rng.choice([
        "Reminder: the campaign creative review is Thursday 2pm, please add your decks to the folder.",
        "Can we move the marketing standup to 10am? The 9am clashes with the eng sync.",
        "Budget for Q3 paid media is due Friday — submit your channel breakdown to Priya.",
    ])
    return wrap(rng, fmt, t, subject="logistics"), "internal_logistics"


def neg_market_fact(rng, fmt):
    t = rng.choice([
        "Gartner reports the martech market grew 12% last year; useful context for planning.",
        "Industry trend: buyers now research 6 vendors on average before a demo (per Forrester).",
        "Third-party survey says email open rates dropped to 18% across SaaS in 2026.",
    ])
    return wrap(rng, fmt, t, subject="market data"), "market_fact"


def neg_generic(rng, fmt):
    t = rng.choice([
        "Does anyone own the marketing wiki? A few links are broken.",
        "Marketing is hiring a content designer this quarter; share the JD if you know anyone.",
        "Moving the marketing folder to the new drive — update your bookmarks.",
    ])
    return wrap(rng, fmt, t, subject="marketing"), "generic_marketing"


def neg_opinion(rng, fmt):
    t = rng.choice([
        "honestly that new competitor ad is kind of cringe, did anyone else see it?",
        "personal opinion: our last campaign's colors were way too loud, but that's just me",
        "I think Super Bowl ads are mostly a waste of money tbh",
    ])
    return wrap(rng, fmt, t, subject="re: ads"), "personal_opinion_ad"


NEG = [neg_logistics, neg_market_fact, neg_generic, neg_opinion]


def cf_claim_vs_logistics(rng, fmt):
    p = rng.choice(PRODUCTS)
    pos = wrap(rng, fmt, f"Approved external claim: \"{p} cuts onboarding time in half.\" Goes live Monday.", subject="claim")
    neg = wrap(rng, fmt, f"Internal note: the {p} onboarding-claim review meeting is Monday — bring data.", subject="logistics")
    return pos, neg, "efficacy_competitive_claim"


def cf_claim_vs_fact(rng, fmt):
    pos = wrap(rng, fmt, "Our ad will say \"the fastest CRM on the market\" — pushing it live this week.", subject="claim")
    neg = wrap(rng, fmt, "Per a third-party benchmark, CRM query speeds vary widely across vendors. Just context.", subject="market data")
    return pos, neg, "efficacy_competitive_claim"


def cf_copy_vs_opinion(rng, fmt):
    p = rng.choice(PRODUCTS)
    pos = wrap(rng, fmt, f"Final hero copy for the {p} site: \"Ship faster. Sleep better.\" Publishing now.", subject="copy")
    neg = wrap(rng, fmt, f"my personal take: the {p} tagline feels a bit generic, but I'm not on the brand team", subject="re: brand")
    return pos, neg, "brand_campaign_copy"


def cf_press_vs_news(rng, fmt):
    p = rng.choice(PRODUCTS)
    pos = wrap(rng, "press", f"{p} today announced it is \"the most advanced platform ever built.\"", subject="press release")
    neg = wrap(rng, fmt, f"TechCrunch wrote up {p}'s funding round — decent coverage, nothing for us to action.", subject="news")
    return pos, neg, "press_pr"


def cf_promo_vs_planning(rng, fmt):
    pos = wrap(rng, fmt, "Going live: \"50% off Pro, this week only — code HALF50.\"", subject="promo")
    neg = wrap(rng, fmt, "Should we run a 50%-off promo this quarter? Opening it for discussion, nothing decided.", subject="promo planning")
    return pos, neg, "pricing_promo"


def cf_claim_in_internal_note(rng, fmt):
    # the claim text appears in BOTH, but one is approved outbound, the other is internal deliberation
    pos = wrap(rng, fmt, "Approved — going live: \"3x faster than any competitor.\" Ships on the site today.", subject="approved claim")
    neg = wrap(rng, fmt, "Internal: should we claim \"3x faster than any competitor\"? We don't have the "
              "benchmark yet, so NOTHING is approved — just discussing.", subject="internal")
    return pos, neg, "efficacy_competitive_claim"


def cf_opinion_with_superlative(rng, fmt):
    p = rng.choice(PRODUCTS)
    pos = wrap(rng, "adcopy", f"Ad headline (publishing): \"{p} — the best CRM, guaranteed.\"", subject="ad")
    neg = wrap(rng, fmt, f"personally i think {p} is the best CRM out there, way better than the rest — "
              f"but that's just my opinion, not a marketing line", subject="re: product")
    return pos, neg, "efficacy_competitive_claim"


def cf_competitor_claim(rng, fmt):
    p = rng.choice(PRODUCTS)
    pos = wrap(rng, "adcopy", f"Our new banner: \"{p} is the #1 rated platform — switch today.\"", subject="banner")
    neg = wrap(rng, fmt, f"heads up, [Competitor]'s ad is claiming THEY are #1 rated — annoying but it's "
              f"their claim, nothing for us to publish", subject="competitor")
    return pos, neg, "efficacy_competitive_claim"


def cf_fact_phrased_as_claim(rng, fmt):
    pos = wrap(rng, fmt, "We'll run the line \"cuts costs by 50%\" in the campaign — pushing live this week.", subject="claim")
    neg = wrap(rng, fmt, "A third-party study reported cost reductions of up to 50% across the category — "
              "useful data, but it's not our claim to make.", subject="research")
    return pos, neg, "efficacy_competitive_claim"


CF = [cf_claim_vs_logistics, cf_claim_vs_fact, cf_copy_vs_opinion, cf_press_vs_news, cf_promo_vs_planning,
      cf_claim_in_internal_note, cf_opinion_with_superlative, cf_competitor_claim, cf_fact_phrased_as_claim]


def near_boundary(rng, fmt):
    t, sub = rng.choice([
        ("Forrester: 70% of B2B buyers prefer self-serve trials. General industry trend, FYI.", "market_fact"),
        ("Hot take: most product taglines sound the same these days. Just my opinion.", "personal_opinion_ad"),
        ("Reminder to use the marketing brand kit for slides — link in the wiki.", "generic_marketing"),
        ("Anyone know the average CAC benchmark for our segment? Saw a report but lost the link.", "market_fact"),
        ("The marketing offsite agenda is finalized; see the calendar invite.", "internal_logistics"),
    ])
    return wrap(rng, fmt, t, subject="fyi"), sub


def intent_only(rng, fmt):
    p = rng.choice(PRODUCTS)
    t, sub = rng.choice([
        (f"We're going to advertise {p} as \"2x faster than the competition\" in the next campaign.", "efficacy_competitive_claim"),
        (f"Plan is to publicly claim {p} is the market leader — flag for legal since it's a comparative claim.", "efficacy_competitive_claim"),
        ("We'll promote a 30%-off launch discount across paid channels next week.", "pricing_promo"),
        (f"The press statement will position {p} as 'the industry's first' — that's the headline we're running.", "press_pr"),
    ])
    return wrap(rng, fmt, t, subject="fyi"), sub


CASUAL_OPENERS = ["", "ok so ", "PSA: ", "heads up — ", "ugh ", "quick one: "]


def casual_pos(rng):
    p = rng.choice(PRODUCTS)
    return rng.choice([
        f"ok final answer on the hero line, going with \"{p}: 3x faster, zero setup\" — pushing it to the "
        f"live site this afternoon so speak now or forever hold your peace",
        f"slipping a cheeky claim into the email blast: \"{p} is the #1 rated CRM\" — i know legal hates "
        f"superlatives but it converts, fight me",
        "promo's locked: 45% off annual, code SPRING45, live tomorrow across email + paid social, please "
        "don't change the code again",
    ]), rng.choice(["efficacy_competitive_claim", "brand_campaign_copy", "pricing_promo"])


def casual_neg(rng):
    return rng.choice([
        "rant: the marketing standup has been moved 4 times this week, can we please just pick a slot",
        "did everyone see that competitor's ad? personally i think it's trying way too hard but idk",
        "saw a report saying SaaS email open rates are tanking, kinda interesting, not our problem rn",
        "who owns the marketing swag budget? need to order stickers for the conference booth",
    ]), rng.choice(["internal_logistics", "personal_opinion_ad", "market_fact", "generic_marketing"])


CONTRACT_MAP = {"don't": "dont", "won't": "wont", "it's": "its", "we're": "were", "that's": "thats", "can't": "cant"}


def _charop(w, rng):
    if len(w) < 4:
        return w
    i = rng.randint(1, len(w) - 2); op = rng.randint(0, 2)
    if op == 0:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == 1:
        return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]


def inject_noise(text, rng, rate=0.07):
    out = []
    for tok in re.split(r"(\s+)", text):
        if not tok or tok.isspace():
            out.append(tok); continue
        low = tok.lower()
        if low in CONTRACT_MAP and rng.random() < 0.6:
            out.append(CONTRACT_MAP[low]); continue
        if any(c.isdigit() for c in tok) or "%" in tok or "$" in tok or sum(c.isupper() for c in tok) >= 3:
            out.append(tok); continue  # protect promo tokens / codes / claims
        if tok.isalpha() and len(tok) >= 5 and rng.random() < rate:
            out.append(_charop(tok, rng)); continue
        out.append(tok)
    s = "".join(out)
    if s[:1].isupper() and rng.random() < 0.3:
        s = s[0].lower() + s[1:]
    return s


def build(variant, seed, sizes, noise_rate):
    rng = random.Random(seed); rows = []; sid = [0]

    def add(text, label, sub, difficulty, hardening, pair_id=None, seed_id=None, fmt="doc"):
        if seed_id is None:
            seed_id = f"mk-seed-{sid[0]}"; sid[0] += 1
        rows.append(dict(id=f"mk-{len(rows):05d}", seed_id=seed_id, domain="marketing", text=text,
                         label=label, subcategory=sub, difficulty=difficulty, hardening=hardening,
                         pair_id=pair_id, source="synthetic", format=fmt, noisy=False))

    n_easy = sizes["easy"]
    for _ in range(n_easy // 2):
        fmt = rng.choice(FORMATS); t, s = rng.choice(POS)(rng, fmt); add(t, 1, s, "easy", "core", fmt=fmt)
    for _ in range(n_easy - n_easy // 2):
        fmt = rng.choice(FORMATS); t, s = rng.choice(NEG)(rng, fmt); add(t, 0, s, "easy", "core", fmt=fmt)
    if variant == "v1":
        for k in range(sizes["cf_pairs"]):
            fmt = rng.choice(FORMATS); pos, neg, s = rng.choice(CF)(rng, fmt)
            sidv, pid = f"mk-cf-{k}", f"cf-{k}"
            add(pos, 1, s, "hard", "counterfactual", pair_id=pid, seed_id=sidv, fmt=fmt)
            add(neg, 0, s, "hard", "counterfactual", pair_id=pid, seed_id=sidv, fmt=fmt)
        for _ in range(sizes["intent"]):
            fmt = rng.choice(FORMATS); t, s = intent_only(rng, fmt); add(t, 1, s, "hard", "intent_only", fmt=fmt)
        for _ in range(sizes["nearbound"]):
            fmt = rng.choice(FORMATS); t, s = near_boundary(rng, fmt); add(t, 0, s, "hard", "near_boundary", fmt=fmt)
        for k in range(sizes["casual"]):
            fmt = rng.choice(["slack", "email", "doc"])
            if k % 2 == 0:
                t, s = casual_pos(rng); add(t, 1, s, "hard", "casual", fmt=fmt)
            else:
                t, s = casual_neg(rng); add(t, 0, s, "hard", "casual", fmt=fmt)

    seen, dd = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"]); dd.append(r)
    rows = dd
    if noise_rate > 0:
        idx = list(range(len(rows))); rng.shuffle(idx)
        for j in idx[: int(noise_rate * len(rows))]:
            rows[j]["text"] = inject_noise(rows[j]["text"], rng); rows[j]["noisy"] = True
    rng.shuffle(rows)
    return rows


def split_by_seed(rows, seed, fracs=(0.7, 0.15, 0.15)):
    rng = random.Random(seed + 1)
    sids = sorted({r["seed_id"] for r in rows}); rng.shuffle(sids)
    n = len(sids); ntr, nva = int(fracs[0] * n), int(fracs[1] * n)
    tr, va = set(sids[:ntr]), set(sids[ntr:ntr + nva])
    out = {"train": [], "val": [], "test": []}
    for r in rows:
        out["train" if r["seed_id"] in tr else "val" if r["seed_id"] in va else "test"].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v0", "v1"], default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/marketing")
    ap.add_argument("--easy", type=int, default=None)
    ap.add_argument("--cf-pairs", type=int, default=170)
    ap.add_argument("--intent", type=int, default=90)
    ap.add_argument("--nearbound", type=int, default=110)
    ap.add_argument("--casual", type=int, default=120)
    ap.add_argument("--noise-rate", type=float, default=0.4)
    ap.add_argument("--pool", default=None,
                    help="emit one UNSPLIT candidate-pool file (path) instead of train/val/test "
                         "splits — for the co-evolution loop (rows keep their seed_id)")
    args = ap.parse_args()
    easy = args.easy if args.easy is not None else (700 if args.variant == "v0" else 480)
    sizes = dict(easy=easy, cf_pairs=args.cf_pairs, intent=args.intent,
                 nearbound=args.nearbound, casual=args.casual)
    rows = build(args.variant, args.seed, sizes, args.noise_rate)

    if args.pool:
        outp = Path(args.pool)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        pos = sum(r["label"] for r in rows)
        print(f"Pool: wrote {len(rows)} rows ({pos} pos / {len(rows)-pos} neg) -> {outp}")
        return

    splits = split_by_seed(rows, args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    def dist(items, k):
        d = {}
        for r in items:
            d[r[k]] = d.get(r[k], 0) + 1
        return dict(sorted(d.items()))
    print(f"Variant {args.variant} — total {len(rows)}")
    for name, items in splits.items():
        pos = sum(r["label"] for r in items)
        print(f"  {name}: {len(items)} ({pos} pos / {len(items)-pos} neg)")
    print("hardening:", dist(rows, "hardening"))
    print("subcategory:", dist(rows, "subcategory"))
    manifest = out.parent / "domains.json"
    doms = json.loads(manifest.read_text()) if manifest.exists() else []
    if out.name not in doms:
        doms.append(out.name); manifest.write_text(json.dumps(sorted(doms)))


if __name__ == "__main__":
    main()
