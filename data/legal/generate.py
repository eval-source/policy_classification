"""
Legal-domain synthetic generator (mirrors data/it/generate.py structure).

POSITIVE (binding/negotiated instruments): nda_confidentiality, contract, indemnity_liability,
privilege_litigation, compliance_obligation.
NEGATIVE: eula_tos_privacy (public boilerplate), legal_news, casual_legal_reference, legal_dept_logistics.

Hard families: counterfactual pairs (boilerplate/template/news vs binding clause — share seed_id),
intent_only positives (obligation stated, clause not quoted), near_boundary negatives (legal
news/commentary/opinion), casual register, and typo noise.

Run:  python data/legal/generate.py --variant v1 --seed 0 --out data/legal
"""
import argparse
import json
import random
import re
import string
from pathlib import Path

PARTIES = ["Acme Corp", "Globex", "Initech", "Umbrella Inc", "Hooli", "Stark Industries", "Wonka Ltd"]
PEOPLE = ["Alice", "Bob", "Priya", "Diego", "Mei", "Sam", "Jordan", "Chen", "Fatima", "Lars"]
FORMATS = ["doc", "email", "contract", "slack", "memo"]
COUNSEL = ["outside counsel", "our GC", "the legal team", "Wilson & Pratt LLP"]


def wrap(rng, fmt, body, subject="update"):
    who = rng.choice(PEOPLE)
    if fmt == "slack":
        return f"{who.lower()}: {body}"
    if fmt == "email":
        return f"Subject: {subject}\nHi team,\n\n{body}\n\nRegards,\n{who}"
    if fmt == "memo":
        return f"MEMORANDUM — {subject}\n{body}"
    return body  # doc / contract


# --------------------------- positives ---------------------------

def pos_nda(rng, fmt):
    p, yrs = rng.choice(PARTIES), rng.choice(["three (3)", "five (5)", "two (2)"])
    t = rng.choice([
        f"Confidentiality. Each party shall hold the other's Confidential Information in strict "
        f"confidence and shall not disclose it to any third party for a period of {yrs} years.",
        f"This Mutual Non-Disclosure Agreement is entered into by and between {p} and the Recipient. "
        f"Recipient agrees not to use Confidential Information except to evaluate the proposed transaction.",
        f"Section 7. Non-Disclosure. The Receiving Party shall protect Confidential Information using "
        f"no less than reasonable care and shall return or destroy it upon termination.",
    ])
    return wrap(rng, fmt, t, subject="NDA"), "nda_confidentiality"


def pos_contract(rng, fmt):
    p = rng.choice(PARTIES)
    t = rng.choice([
        f"Per our executed MSA with {p}, the SOW pricing is locked for 12 months and net-30 payment "
        f"terms apply; renewal is automatic absent 60 days' written notice.",
        f"This Statement of Work, governed by the Master Services Agreement dated Jan 3, between {p} "
        f"(\"Client\") and Vendor, sets fees of $48,000 payable in equal monthly installments.",
        f"Data Processing Agreement: Processor shall process Personal Data only on documented "
        f"instructions from the Controller and shall assist with data-subject requests under Art. 28.",
    ])
    return wrap(rng, fmt, t, subject="MSA/SOW"), "contract"


def pos_indemnity(rng, fmt):
    p = rng.choice(PARTIES)
    t = rng.choice([
        f"Indemnification. The Supplier shall indemnify, defend, and hold harmless the Client against "
        f"any and all losses, damages, and reasonable attorneys' fees arising from breach of this Agreement.",
        f"Limitation of Liability. In no event shall {p}'s aggregate liability exceed the fees paid in "
        f"the twelve (12) months preceding the claim, except for indemnification obligations.",
        f"Warranty. Vendor warrants that the Services will conform to the Documentation for ninety (90) "
        f"days; Client's sole remedy is re-performance or refund.",
    ])
    return wrap(rng, fmt, t, subject="liability clause"), "indemnity_liability"


def pos_privilege(rng, fmt):
    p = rng.choice(PARTIES)
    t = rng.choice([
        f"PRIVILEGED & CONFIDENTIAL — ATTORNEY-CLIENT. Memo from {rng.choice(COUNSEL)} re: our exposure "
        f"in the pending {p} litigation; do not forward outside the legal hold.",
        f"Litigation update: {p} served us with a complaint alleging breach; counsel recommends we "
        f"preserve all documents and assert the work-product privilege over the internal assessment.",
        f"Settlement discussions with {p} are protected under FRE 408; do not reference the offered "
        f"figure ($250k) outside privileged channels.",
    ])
    return wrap(rng, fmt, t, subject="PRIVILEGED"), "privilege_litigation"


def pos_obligation(rng, fmt):
    p = rng.choice(PARTIES)
    t = rng.choice([
        f"Reminder: we are contractually obligated under the {p} DPA to notify the controller within "
        f"24 hours of any personal-data breach — this is a binding term, not best practice.",
        f"Under the MSA we must maintain $5M cyber-insurance and provide a SOC 2 report annually; "
        f"these are enforceable obligations with cure periods.",
        f"The {p} agreement requires us to obtain written consent before subcontracting — we are bound "
        f"to that, so loop in legal before engaging the vendor.",
    ])
    return wrap(rng, fmt, t, subject="obligation"), "compliance_obligation"


POS = [pos_nda, pos_contract, pos_indemnity, pos_privilege, pos_obligation]

# --------------------------- negatives ---------------------------

def neg_eula(rng, fmt):
    t = rng.choice([
        "By using this website you agree to our Terms of Service and acknowledge our Privacy Policy.",
        "We've updated our Privacy Policy. Your continued use of the app constitutes acceptance of the "
        "new terms. No action is required.",
        "Terms of Use: This service is provided \"as is\". See our cookie policy for details on tracking.",
    ])
    return wrap(rng, fmt, t, subject="terms update"), "eula_tos_privacy"


def neg_news(rng, fmt):
    p = rng.choice(PARTIES)
    t = rng.choice([
        f"News: {p} settled its antitrust case for $40M today; analysts expect more scrutiny of the sector.",
        "Op-ed: why indemnification clauses are reshaping SaaS contracts in 2026. Worth a read for PMs.",
        "Reuters: the new data-privacy regulation takes effect next quarter, affecting most EU vendors.",
    ])
    return wrap(rng, fmt, t, subject="legal news"), "legal_news"


def neg_casual(rng, fmt):
    t = rng.choice([
        "legally speaking we should probably double-check this, but I'm not a lawyer lol",
        "ugh the legal review is taking forever, anyone know who to ping?",
        "is it just me or does every vendor call end with 'we'll have legal look at it'",
    ])
    return wrap(rng, fmt, t, subject="re: legal"), "casual_legal_reference"


def neg_logistics(rng, fmt):
    t = rng.choice([
        "Reminder: the legal team's weekly sync moved to 3pm Thursdays in the Oak room.",
        "Legal is hiring two paralegals this quarter; reach out to Priya if you have referrals.",
        "The legal department's budget review is due Friday — please submit your contract-volume numbers.",
    ])
    return wrap(rng, fmt, t, subject="legal dept"), "legal_dept_logistics"


NEG = [neg_eula, neg_news, neg_casual, neg_logistics]

# --------------------------- counterfactual pairs ---------------------------

# MINIMAL-EDIT counterfactuals: pos and neg share an IDENTICAL clause body; only the terse FRAME
# differs, and that frame is the SOLE signal of negotiated/binding instrument (POSITIVE) vs a
# policy-EXCLUDED context — public boilerplate (ToS/privacy), commentary, sample/template, textbook
# example, or an unexecuted draft (NEGATIVE). The earlier CFs were too easy (frozen cf-spec 100)
# because the pos/neg sentences were entirely different and the cues were loud ("EXECUTED", "blank
# template — fill in [PARTY NAME]", "Law360 explains"). Sharing the binding-looking clause body and
# leaving only a subtle frame is what makes these discriminative for a strong base model.

_CLAUSES = [
    ("the Receiving Party shall hold all Confidential Information in strict confidence and not "
     "disclose it to any third party for five (5) years", "nda_confidentiality"),
    ("the Supplier shall indemnify and hold the Client harmless against all third-party claims "
     "arising from the Services", "indemnity_liability"),
    ("the Processor shall notify the Controller within 24 hours of any personal-data breach",
     "compliance_obligation"),
    ("the Vendor shall maintain $5M in cyber-insurance and deliver a SOC 2 Type II report annually",
     "contract"),
    ("neither party shall solicit the other's employees for twelve (12) months after termination",
     "contract"),
    ("the Licensee shall not reverse-engineer, decompile, or disassemble the Software", "contract"),
    ("each party's aggregate liability shall not exceed the fees paid in the prior twelve (12) months",
     "indemnity_liability"),
]


def _cap(s):
    return s[0].upper() + s[1:]


# POSITIVE frames — negotiated / executed / party-specific instrument
_BIND_FRAMES = [
    lambda c, p, k, n: f"Section {n}. {_cap(c)}.",
    lambda c, p, k, n: f"Per our executed {k} with {p}, {c}.",
    lambda c, p, k, n: f"Under the {k} we signed with {p}, {c}.",
    lambda c, p, k, n: f"{_cap(c)}. Countersigned by both parties yesterday.",
    lambda c, p, k, n: f"As agreed in the fully-executed {k}, {c}.",
]
# NEGATIVE frames — same clause body, policy-EXCLUDED context (each a defensible non-trigger).
# Deliberately SUBTLE/descriptive: most don't name a category outright, so the model must infer
# "this is describing a norm / quoting a reference / not yet executed" rather than key on a keyword.
_NONBIND_FRAMES = [
    lambda c, p, k, n: f"Most {k}s say {c}.",                       # describing the norm (commentary)
    lambda c, p, k, n: f"In a typical deal, {c}.",                  # commentary
    lambda c, p, k, n: f"For reference, the standard language: {c}.",   # quoting a reference, not the deal
    lambda c, p, k, n: f"Our public Terms of Service note that {c}.",   # boilerplate
    lambda c, p, k, n: f"Textbook example — {c}.",                  # educational
    lambda c, p, k, n: f"Draft for discussion, nothing agreed yet: {c}.",  # unexecuted
    lambda c, p, k, n: f"A blank {k} template includes: {c}.",      # template/sample
]


def _cf_pair(rng, fmt):
    c, sub = rng.choice(_CLAUSES)
    p = rng.choice(PARTIES)
    k = rng.choice(["MSA", "NDA", "SOW", "DPA", "Agreement"])
    n = rng.randint(2, 14)
    pos = wrap(rng, fmt, rng.choice(_BIND_FRAMES)(c, p, k, n), subject=k)
    neg = wrap(rng, fmt, rng.choice(_NONBIND_FRAMES)(c, p, k, n), subject="for reference")
    return pos, neg, sub


# single parameterized family; variety comes from the random clause body + frame per draw
CF = [_cf_pair]

# --------------------------- near-boundary negatives ---------------------------

def near_boundary(rng, fmt):
    t, sub = rng.choice([
        ("Analysis: the Supreme Court's latest ruling may change how warranty disclaimers are "
         "enforced. General commentary, not legal advice.", "legal_news"),
        ("Our Privacy Policy explains what data we collect and your rights — it's public boilerplate "
         "applicable to all users.", "eula_tos_privacy"),
        ("What's the difference between an MSA and an SOW? Studying for a contracts class.", "legal_news"),
        ("The legal team recommends everyone complete the annual compliance training by Friday.", "legal_dept_logistics"),
        ("Hot take: most EULAs are unreadable and nobody actually agrees to them meaningfully.", "casual_legal_reference"),
        ("Blog tutorial: here's a sample indemnification clause you might see in an SOW, with notes "
         "on what each part means. Educational only.", "legal_news"),
        ("We promise to always protect your privacy and treat your data with respect.", "eula_tos_privacy"),
        ("Quote from the court opinion: \"the parties shall bear their own costs\" — interesting "
         "precedent, sharing for awareness.", "legal_news"),
    ])
    return wrap(rng, fmt, t, subject="fyi"), sub


# --------------------------- intent-only positives ---------------------------

def intent_only(rng, fmt):
    p = rng.choice(PARTIES)
    t, sub = rng.choice([
        (f"Heads up — under our signed contract with {p} we're on the hook to indemnify them for any "
         f"data breach; that's binding, loop in legal.", "indemnity_liability"),
        (f"We executed the NDA with {p} last week, so anything they share is now under confidentiality "
         f"obligations on our side.", "nda_confidentiality"),
        ("The DPA we signed requires 24-hour breach notification — we are legally bound to that timeline.",
         "compliance_obligation"),
        (f"We're now in active litigation with {p}; legal hold is in effect, preserve all related docs.",
         "privilege_litigation"),
    ])
    return wrap(rng, fmt, t, subject="fyi"), sub


# --------------------------- casual long-form ---------------------------

CASUAL_OPENERS = ["", "PSA: ", "heads up — ", "ok so ", "ugh ", "quick one: "]
CASUAL_CLOSERS = ["", " anyway", " lol", " 🙃", " thoughts?", " back to it"]


def casual_pos(rng):
    p = rng.choice(PARTIES)
    return rng.choice([
        f"finally got the {p} contract signed after 3 rounds of redlines — the indemnification cap is "
        f"$2M and we're bound to net-30, so AP please note we can't change those terms unilaterally",
        f"so we apparently agreed in the {p} MSA to a 5-year confidentiality term?? just re-read it. "
        f"that's binding on us, fyi, don't share their roadmap deck with anyone",
        "reminder we're under a signed DPA so the 24h breach-notification clause is a hard contractual "
        "obligation, not a nice-to-have — flag any incident to legal immediately",
    ]), rng.choice(["contract", "nda_confidentiality", "compliance_obligation"])


def casual_neg(rng):
    return rng.choice([
        "lpt: actually read the Terms of Service before clicking agree, half of them claim rights to "
        "your data. (nobody does this, including me)",
        "the legal team offsite got moved again, third time this month, at this point just put it on a "
        "recurring invite please",
        "rant: every contract negotiation takes 6 weeks because legal and procurement can't agree on a "
        "meeting time, not even the terms",
        "did anyone catch the news about the big antitrust settlement? wild numbers. not our problem tho",
    ]), rng.choice(["casual_legal_reference", "legal_dept_logistics", "legal_news"])


# --------------------------- noise ---------------------------

CONTRACT_MAP = {"don't": "dont", "won't": "wont", "it's": "its", "we're": "were", "that's": "thats",
                "shouldn't": "shouldnt", "can't": "cant"}


def _charop(w, rng):
    if len(w) < 4:
        return w
    i = rng.randint(1, len(w) - 2)
    op = rng.randint(0, 2)
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
        # protect numbers, $amounts, citations, ALLCAPS legal markers
        if any(c.isdigit() for c in tok) or "$" in tok or sum(c.isupper() for c in tok) >= 3:
            out.append(tok); continue
        if tok.isalpha() and len(tok) >= 5 and rng.random() < rate:
            out.append(_charop(tok, rng)); continue
        out.append(tok)
    s = "".join(out)
    if s[:1].isupper() and rng.random() < 0.3:
        s = s[0].lower() + s[1:]
    return s


# --------------------------- build ---------------------------

def build(variant, seed, sizes, noise_rate):
    rng = random.Random(seed)
    rows = []
    sid = [0]

    def add(text, label, sub, difficulty, hardening, pair_id=None, seed_id=None, fmt="doc"):
        if seed_id is None:
            seed_id = f"lg-seed-{sid[0]}"; sid[0] += 1
        rows.append(dict(id=f"lg-{len(rows):05d}", seed_id=seed_id, domain="legal", text=text,
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
            sidv, pid = f"lg-cf-{k}", f"cf-{k}"
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

    # dedup identical texts (leakage-safe)
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
    ap.add_argument("--out", default="data/legal")
    ap.add_argument("--easy", type=int, default=None)
    ap.add_argument("--cf-pairs", type=int, default=170)
    ap.add_argument("--intent", type=int, default=90)
    ap.add_argument("--nearbound", type=int, default=110)
    ap.add_argument("--casual", type=int, default=120)
    ap.add_argument("--noise-rate", type=float, default=0.4)
    args = ap.parse_args()
    easy = args.easy if args.easy is not None else (700 if args.variant == "v0" else 480)
    sizes = dict(easy=easy, cf_pairs=args.cf_pairs, intent=args.intent,
                 nearbound=args.nearbound, casual=args.casual)
    rows = build(args.variant, args.seed, sizes, args.noise_rate)
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
