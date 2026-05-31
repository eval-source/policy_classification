"""
Fetch + label REAL IT-domain data from Hugging Face (bulk = metadata + regex; a Claude/human
judge verifies a sample separately). Writes data/real/ (gitignored — licenses + possible PII).

Sources (label per the same rubric as the synthetic generator):
  - ai4privacy/pii-masking-300k   -> pii_handling. POSITIVE iff a HIGH-sensitivity entity
       (SSN, card, passport, password, IBAN, medical, ...) is present; a lone name/email is
       low-sensitivity -> NEGATIVE/low-confidence (the "casual PII" near-boundary).
  - Tobi-Bueck/customer-support-tickets -> support. NEGATIVE (helpdesk/support) by default;
       security/breach-tagged tickets -> POSITIVE (vuln_incident, mid-confidence, verify);
       a regex secret in the body overrides -> POSITIVE (secret_credential).
  - AlicanKiraz0/All-CVE-Records-Training-Dataset -> security_news. NEGATIVE: a public
       third-party CVE advisory is a reference, not OUR disclosure (rubric call; verify).
  - code_search_net -> public_snippet. NEGATIVE (public code); regex secret overrides.

Every row carries label_source {regex|metadata}, confidence {high|med|low}, source="real",
hf_dataset. Splits are leakage-safe by row; the test split also yields a realistic
positive-rare view and a balanced view.

NOTE: secret_credential / infra_config / access_control / our-security_policy have NO clean
public source (ethics/availability) -> they stay synthetic-only. This is logged, not hidden.

Run:  python data/it/fetch_real.py --limit 500 --seed 0
"""
import argparse
import json
import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "eval"))
import regex_baseline  # reuse the deterministic secret detector

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "real"

# PII rubric (v3 — label-audit fix f028): in-scope "PII handling" if HIGH-sensitivity PII, OR
# a PERSONAL RECORD = enough SENSITIVE/STRUCTURAL identifiers about a person (address, phone,
# DOB, ID/account numbers...). Crucially, bare NAMES no longer count — v2 counted named
# entities, so name-heavy NARRATIVE (comments, strategy docs) was wrongly labeled a record.
HIGH_SENS = {
    "SSN", "SOCIALNUMBER", "PASSPORT", "DRIVERLICENSE", "IDCARD", "TAXNUMBER",
    "CREDITCARDNUMBER", "CREDITCARDCVV", "IBAN", "BIC", "ACCOUNTNUMBER", "PIN",
    "PASSWORD", "MEDICAL", "BITCOINADDRESS", "ETHEREUMADDRESS", "VEHICLEVIN",
    "PHONEIMEI", "MASKEDNUMBER",
}
# substrings marking a SENSITIVE/STRUCTURAL identifier (not a bare name/date/job)
STRUCTURAL_KEYS = ("STREET", "CITY", "STATE", "ZIP", "POSTCODE", "POSTAL", "BUILDING",
                   "ADDRESS", "PHONE", "EMAIL", "DOB", "BIRTH", "ACCOUNT", "LICENSE",
                   "PASSPORT", "SSN", "IDCARD", "STUDENT", "EMPLOYEE", "CUSTOMER",
                   "NATIONAL", "TAX", "IPV", "IBAN", "CREDIT")
NAME_LABELS = {"FIRSTNAME", "LASTNAME", "SURNAME", "MIDDLENAME", "FULLNAME", "GIVENNAME", "NAME"}


def pii_label(labels):
    """Return (positive: bool, confidence) for a set of uppercased PII entity labels."""
    if labels & HIGH_SENS:
        return True, "high"
    n_struct = sum(1 for L in labels if any(k in L for k in STRUCTURAL_KEYS))
    has_name = any(L in NAME_LABELS for L in labels)
    if n_struct >= 2 or (has_name and n_struct >= 1):
        return True, "med"     # a genuine personal record (name+address, address+phone, ...)
    return False, "low"        # lone/casual identifier or name-only narrative
# A support-ticket positive needs BOTH a security-ish tag AND incident language in the body.
# (Verification showed a tag alone is too loose — it swept in outages, bug reports, and
# security-compliance *inquiries*; the incident-keyword gate drops those "request" tickets.)
SEC_TAGS = {"data breach", "security", "phishing", "malware", "ransomware", "hacking"}
INCIDENT_KW = re.compile(
    r"\b(breach|unauthori[sz]ed|credential theft|compromis|exfiltrat|ransomware|"
    r"phish|data leak|hacked|intrusion|malware|account takeover|sensitive data (?:was|were|access))\b",
    re.I)


def norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def truncate(text, n=2000):
    text = (text or "").strip()
    return text if len(text) <= n else text[:n]


def regex_override(text):
    label, matches = regex_baseline.predict(text)
    return (label == 1), matches


# --------------------------- per-source row producers ---------------------------

def from_pii(limit, rng):
    from datasets import load_dataset
    ds = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=True)
    out = []
    for row in ds:
        if row.get("language") not in (None, "English", "en"):
            continue
        text = truncate(row.get("source_text") or row.get("target_text"))
        if not text:
            continue
        labels = {m.get("label", "").upper() for m in (row.get("privacy_mask") or [])}
        hit_secret, _ = regex_override(text)
        if hit_secret:
            lab, sub, conf, src = 1, "secret_credential", "high", "regex"
        else:
            is_pos, conf = pii_label(labels)
            lab, sub, src = (1 if is_pos else 0), "pii_handling", "metadata"
        out.append((text, lab, sub, conf, src, "prose", "ai4privacy/pii-masking-300k"))
        if len(out) >= limit:
            break
    return out


def from_support(limit, rng):
    from datasets import load_dataset
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train", streaming=True)
    out = []
    for row in ds:
        if row.get("language") not in (None, "en"):
            continue
        text = truncate(((row.get("subject") or "") + "\n" + (row.get("body") or "")).strip())
        if not text:
            continue
        tags = {str(row.get(f"tag_{i}") or "").lower() for i in range(1, 9)}
        hit_secret, _ = regex_override(text)
        if hit_secret:
            lab, sub, conf, src = 1, "secret_credential", "high", "regex"
        elif (tags & SEC_TAGS) and INCIDENT_KW.search(text):
            lab, sub, conf, src = 1, "vuln_incident", "med", "metadata"  # genuine breach/incident
        else:
            q = str(row.get("queue") or "").lower()
            sub = "helpdesk_ticket" if any(k in q for k in ("hardware", "it support", "technical")) else "support_chatter"
            lab, conf, src = 0, "med", "metadata"
        out.append((text, lab, sub, conf, src, "ticket", "Tobi-Bueck/customer-support-tickets"))
        if len(out) >= limit:
            break
    return out


def from_cve(limit, rng):
    from datasets import load_dataset
    ds = load_dataset("AlicanKiraz0/All-CVE-Records-Training-Dataset", split="train", streaming=True)
    out = []
    for row in ds:
        text = truncate(row.get("Assistant") or row.get("User"))
        if not text:
            continue
        # third-party CVE advisory -> reference, not our disclosure -> NEGATIVE (rubric call)
        out.append((text, 0, "security_news", "med", "metadata", "advisory",
                    "AlicanKiraz0/All-CVE-Records-Training-Dataset"))
        if len(out) >= limit:
            break
    return out


CODE_CANDIDATES = [
    ("iamtarun/python_code_instructions_18k_alpaca", None, ["output", "instruction"]),
    ("Nan-Do/code-search-net-python", None, ["code", "func_code_string"]),
    ("google/code_x_glue_ct_code_to_text", "python", ["code", "original_string"]),
]


def from_code(limit, rng):
    from datasets import load_dataset
    last = None
    for rid, cfg, fields in CODE_CANDIDATES:
        try:
            ds = load_dataset(rid, cfg, split="train", streaming=True)
            out = []
            for row in ds:
                text = next((truncate(row.get(f)) for f in fields if row.get(f)), "")
                if not text:
                    continue
                hit_secret, _ = regex_override(text)
                if hit_secret:
                    lab, sub, conf, src = 1, "secret_credential", "high", "regex"
                else:
                    lab, sub, conf, src = 0, "public_snippet", "high", "metadata"
                out.append((text, lab, sub, conf, src, "code", rid))
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception as e:
            last = f"{rid}: {type(e).__name__}"
            continue
    raise RuntimeError(f"all code candidates failed (last: {last})")


def from_terraform(limit, rng):
    # real infrastructure-as-code. Sensitive values (regex secret) -> positive; clean resource
    # definitions -> negative infra_config (a real "looks-infra-y but no secrets" hard negative).
    from datasets import load_dataset
    ds = load_dataset("galcan/terraform_sec", split="train", streaming=True)
    out = []
    for row in ds:
        text = truncate(row.get("input"))
        if not text:
            continue
        hit, _ = regex_override(text)
        if hit:
            out.append((text, 1, "secret_credential", "high", "regex", "config", "galcan/terraform_sec"))
        else:
            out.append((text, 0, "infra_config", "med", "metadata", "config", "galcan/terraform_sec"))
        if len(out) >= limit:
            break
    return out


def from_security_policy(limit, rng):
    # real information-security policy documents -> in-scope security_policy (positive).
    from datasets import load_dataset
    ds = load_dataset("davidquicast/my-information-security-policy-distiset", split="train", streaming=True)
    out, seen = [], set()
    for row in ds:
        text = truncate(row.get("context"))
        if not text or text in seen:
            continue
        seen.add(text)
        out.append((text, 1, "security_policy", "med", "metadata", "doc",
                    "davidquicast/my-information-security-policy-distiset"))
        if len(out) >= limit:
            break
    return out


def from_enron(limit, rng):
    # real corporate email = realistic negative traffic + diversity. Conservative labeling:
    # negative unless a literal secret is present (NOTE: genuinely-sensitive emails may be
    # under-labeled -> a known label-noise risk for this source).
    from datasets import load_dataset
    ds = load_dataset("LLM-PBE/enron-email", split="train", streaming=True)
    out = []
    for row in ds:
        text = truncate(row.get("text") or row.get("body"))
        if not text or len(text) < 40:
            continue
        hit, _ = regex_override(text)
        if hit:
            out.append((text, 1, "secret_credential", "high", "regex", "email", "LLM-PBE/enron-email"))
        else:
            out.append((text, 0, "support_chatter", "low", "metadata", "email", "LLM-PBE/enron-email"))
        if len(out) >= limit:
            break
    return out


# NOTE: CVE dropped from the benchmark (decision 2026-05-29, finding f011): a public
# third-party CVE advisory sits ambiguously between the policy's in-scope "vulnerability
# details" and out-of-scope "security news/commentary". `from_cve` is kept for easy revisit.
SOURCES = {"pii": from_pii, "support": from_support, "code": from_code,
           "terraform": from_terraform, "security_policy": from_security_policy, "enron": from_enron}


# --------------------------- build ---------------------------

def load_synthetic_hashes():
    hashes = set()
    for sp in ("train", "val", "test"):
        fp = ROOT / "data" / "it" / f"{sp}.jsonl"
        if fp.exists():
            for line in fp.open():
                if line.strip():
                    hashes.add(norm(json.loads(line)["text"]))
    return hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="rows pulled per source (pre-dedup)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-rate", type=float, default=0.12, help="positive rate for realistic test view")
    ap.add_argument("--train-min-conf", default="med", choices=["low", "med", "high"])
    args = ap.parse_args()
    rng = random.Random(args.seed)

    syn = load_synthetic_hashes()
    seen = set(syn)  # dedup real vs synthetic AND within real
    rows = []
    counter = 0
    print("Fetching sources (failures are skipped):")
    for name, fn in SOURCES.items():
        try:
            produced = fn(args.limit, rng)
        except Exception as e:
            print(f"  {name:8} SKIPPED: {type(e).__name__}: {str(e)[:120]}")
            continue
        kept = 0
        for (text, lab, sub, conf, lsrc, fmt, hf) in produced:
            h = norm(text)
            if not h or h in seen:
                continue
            seen.add(h)
            rows.append(dict(
                id=f"real-{counter:05d}", seed_id=f"real-{name}-{counter}", domain="it",
                text=text, label=lab, subcategory=sub, difficulty="natural",
                hardening="real", pair_id=None, source="real", format=fmt, noisy=False,
                label_source=lsrc, confidence=conf, hf_dataset=hf,
            ))
            counter += 1
            kept += 1
        print(f"  {name:8} produced={len(produced):4} kept(after dedup)={kept}")

    if not rows:
        print("No rows fetched (network/gating?). Aborting.")
        return

    # leakage-safe split by seed_id (each real row is its own group; no pairs here)
    rng.shuffle(rows)
    n = len(rows)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)
    splits = {"train": rows[:n_tr], "val": rows[n_tr:n_tr + n_va], "test": rows[n_tr + n_va:]}

    # train: gate on confidence to keep label noise out of the model
    order = {"low": 0, "med": 1, "high": 2}
    thr = order[args.train_min_conf]
    splits["train"] = [r for r in splits["train"] if order[r["confidence"]] >= thr]

    OUT.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    # realistic positive-rare + balanced views of the held-out test pool
    test = splits["test"]
    pos = [r for r in test if r["label"] == 1]
    neg = [r for r in test if r["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    # realistic: keep all negs, take positives to hit pos_rate
    k_pos = max(1, int(args.pos_rate / (1 - args.pos_rate) * len(neg)))
    realistic = neg + pos[:k_pos]
    rng.shuffle(realistic)
    # balanced: equal counts
    m = min(len(pos), len(neg))
    balanced = pos[:m] + neg[:m]
    rng.shuffle(balanced)
    for name, items in [("test_realistic", realistic), ("test_balanced", balanced)]:
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    # summary
    def dist(items, key):
        d = {}
        for r in items:
            d[r[key]] = d.get(r[key], 0) + 1
        return dict(sorted(d.items()))

    print(f"\nTotal real rows kept: {n}  (deduped vs {len(syn)} synthetic)")
    for name in ("train", "val", "test", "test_realistic", "test_balanced"):
        items = splits.get(name) or {"test_realistic": realistic, "test_balanced": balanced}[name]
        p = sum(r["label"] for r in items)
        print(f"  {name:15} {len(items):4} ({p} pos / {len(items)-p} neg)")
    print("subcategory:", dist(rows, "subcategory"))
    print("label_source:", dist(rows, "label_source"))
    print("confidence:", dist(rows, "confidence"))
    print("hf_dataset:", dist(rows, "hf_dataset"))


if __name__ == "__main__":
    main()
