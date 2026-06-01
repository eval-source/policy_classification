"""
IT-domain synthetic data generator.

Variants:
  v0  — easy core only (clearly-separable pos/neg). Establishes a high-F1 baseline.
  v1  — easy core + four HARD families that attack the surface-matching shortcut v0 exposed.

LABELLING RUBRIC (applied consistently across families):
  POSITIVE (label=1) — the text DISCLOSES or EFFECTUATES sensitive IT content:
    a real secret value; an *enacted* access grant/change; an incident affecting US;
    actual PII; OUR security control/config containing sensitive data. A secret that is
    merely obfuscated (base64/spaced) is still a disclosure -> positive.
  NEGATIVE (label=0) — the text only REFERENCES the topic without disclosing/effectuating:
    a placeholder instead of a real secret; a request/proposal not yet enacted;
    third-party security news; public boilerplate/ToS; vendor security marketing claims;
    abstract/educational questions; generic security advice.

Hard families (field `hardening`):
  counterfactual — minimal-edit pairs that flip the label; the two members share a seed_id
                   so the splitter keeps them together (no "answer" leakage).
  intent_only    — in-scope disclosures with NO literal secret to regex.
  near_boundary  — sounds in-scope but is a reference, not a disclosure (-> negative).
  obfuscation    — real secrets, obscured (base64 / spaced / chunked) to defeat regex.

Coverage axes varied: format (slack/email/jira/doc/config/commit), persona, length.
Positives embed FORMAT-VALID but FAKE secrets (random values) so the regex baseline has
signal and we never ship a real credential.

Run:  python data/it/generate.py --variant v1 --seed 0 --out data/it
"""
import argparse
import base64
import json
import random
import re
import string
from pathlib import Path

# ---------------------------------------------------------------------------
# Fake-but-format-valid secret generators (deterministic given the rng)
# ---------------------------------------------------------------------------

B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
B64 = string.ascii_letters + string.digits + "+/"
HEX = "0123456789abcdef"


def _rand(rng, alphabet, n):
    return "".join(rng.choice(alphabet) for _ in range(n))


def aws_access_key(rng):
    return "AKIA" + _rand(rng, B32, 16)


def aws_secret_key(rng):
    return _rand(rng, B64, 40)


def github_token(rng):
    return "ghp_" + _rand(rng, string.ascii_letters + string.digits, 36)


def slack_token(rng):
    return f"xoxb-{rng.randint(10**11, 10**12)}-{rng.randint(10**11, 10**12)}-{_rand(rng, string.ascii_letters + string.digits, 24)}"


def jwt(rng):
    seg = lambda n: _rand(rng, string.ascii_letters + string.digits + "-_", n)
    return f"eyJ{seg(20)}.eyJ{seg(40)}.{seg(43)}"


def generic_api_key(rng):
    return _rand(rng, HEX, 32)


def password(rng):
    base = rng.choice(["Tr0ub4dor", "Hunter", "Summer", "Winter", "Dragon", "Matrix", "Falcon"])
    return f"{base}{rng.randint(10, 99)}{rng.choice('!@#$%&*')}{_rand(rng, string.ascii_lowercase, 3)}"


def pg_conn_string(rng):
    user = rng.choice(["app", "svc_api", "prod_rw", "admin", "etl"])
    host = rng.choice(["db.internal", "prod-db-1.us-east-1.rds.amazonaws.com", "10.0.2.14"])
    db = rng.choice(["main", "users", "billing", "analytics"])
    return f"postgres://{user}:{password(rng)}@{host}:5432/{db}"


def private_key_blob(rng):
    return "-----BEGIN RSA PRIVATE KEY-----\n" + _rand(rng, B64, 64) + "\n" + _rand(rng, B64, 64) + "\n-----END RSA PRIVATE KEY-----"


def ssn(rng):
    return f"{rng.randint(100, 899):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"


# ---------------------------------------------------------------------------
# Shared slot vocab + format envelope
# ---------------------------------------------------------------------------

PEOPLE = ["Alice", "Bob", "Priya", "Diego", "Mei", "Sam", "Jordan", "Chen", "Fatima", "Lars"]
SERVICES = ["payments-api", "auth-service", "data-pipeline", "billing-worker", "search-index", "notification-svc"]
ENVS = ["production", "staging", "prod", "the prod cluster", "us-east-1 prod"]
VENDORS = ["Okta", "Datadog", "PagerDuty", "Snowflake", "AWS", "GitHub", "Cloudflare", "Jira", "Slack", "1Password"]
COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Inc", "Hooli", "Stark Industries"]
FORMATS = ["slack", "email", "jira", "doc", "config", "commit"]


def wrap(rng, fmt, body, subject="update"):
    who = rng.choice(PEOPLE)
    if fmt == "slack":
        return f"{who.lower()}: {body}"
    if fmt == "email":
        return f"Subject: {subject}\nHi team,\n\n{body}\n\nThanks,\n{who}"
    if fmt == "jira":
        return f"[{rng.choice(['OPS','SEC','INFRA','IT'])}-{rng.randint(100,9999)}] {subject}\n{body}"
    if fmt == "commit":
        return f"commit {_rand(rng, HEX, 7)}\n\n{body}"
    return body  # doc / config: as-is


# ---------------------------------------------------------------------------
# EASY CORE — positive generators (label=1)
# ---------------------------------------------------------------------------

def pos_secret_credential(rng, fmt):
    _, val, noun = rng.choice([
        ("aws", aws_access_key(rng) + " / " + aws_secret_key(rng), "AWS access key + secret"),
        ("gh", github_token(rng), "GitHub personal access token"),
        ("slack", slack_token(rng), "Slack bot token"),
        ("jwt", jwt(rng), "service JWT"),
        ("api", generic_api_key(rng), "API key"),
        ("conn", pg_conn_string(rng), "database connection string"),
        ("pkey", private_key_blob(rng), "private key"),
    ])
    svc = rng.choice(SERVICES)
    t = rng.choice([
        f"Here's the {noun} for {svc}: {val}",
        f"Rotated the {noun} for {svc}, new value is {val} — update your env.",
        f"prod creds for {svc}\n{noun}: {val}",
        f"can you set the {noun}? it's {val}",
    ])
    return wrap(rng, fmt, t, subject=f"{svc} credentials"), "secret_credential"


def pos_access_control(rng, fmt):
    who, env = rng.choice(PEOPLE), rng.choice(ENVS)
    t = rng.choice([
        f"Granted {who} admin/sudo access to {env}. Added their key to the {rng.choice(SERVICES)} bastion.",
        f"I added {who} to the AdministratorAccess IAM group and gave them write on the {env} S3 buckets.",
        f"Promoted {who} to owner on the GitHub org and gave root on the {env} hosts.",
        f"Updated sudoers on {env}: {who.lower()} ALL=(ALL) NOPASSWD:ALL",
    ])
    return wrap(rng, fmt, t, subject="access change"), "access_control"


def pos_security_policy(rng, fmt):
    t = rng.choice([
        "Security policy: all production access requires MFA, SSO, and is logged; standing access is forbidden — use just-in-time elevation.",
        "Per the new infosec policy, secrets must live in the vault, never in code; keys rotate every 90 days and on offboarding.",
        "Policy: encrypt all PII at rest with AES-256, restrict prod DB access to the on-call group, and audit every privileged action.",
        "Effective immediately, SSH to production is only allowed through the bastion with hardware-key 2FA; password auth is disabled.",
    ])
    return wrap(rng, fmt, t, subject="security policy"), "security_policy"


def pos_infra_config(rng, fmt):
    svc = rng.choice(SERVICES)
    t = rng.choice([
        f"# .env for {svc}\nDATABASE_URL={pg_conn_string(rng)}\nAWS_SECRET_ACCESS_KEY={aws_secret_key(rng)}\nJWT_SIGNING_KEY={generic_api_key(rng)}",
        f"production values.yaml\n  apiKey: {generic_api_key(rng)}\n  dbPassword: {password(rng)}\n  host: prod-db-1.internal",
        f"terraform.tfvars\n  rds_master_password = \"{password(rng)}\"\n  datadog_api_key = \"{generic_api_key(rng)}\"",
        f"kubectl secret for {svc}:\n  GITHUB_TOKEN: {github_token(rng)}\n  REDIS_URL: redis://:{password(rng)}@10.0.1.7:6379",
    ])
    return wrap(rng, "config", t, subject=f"{svc} config"), "infra_config"


def pos_vuln_incident(rng, fmt):
    svc = rng.choice(SERVICES)
    t = rng.choice([
        f"INCIDENT: attacker exploited an unauthenticated RCE in {svc} (CVE-2024-{rng.randint(1000,9999)}) and pulled the prod DB; rotating all keys now.",
        f"Security incident — we found a publicly exposed S3 bucket with customer data and a leaked {rng.choice(['AWS','Slack'])} token; containment in progress.",
        f"Pen test found a SQL injection in {svc} that dumps the users table including password hashes. Sev1, patch tonight.",
        f"We were breached: phished an admin's Okta session, lateral-moved to the {rng.choice(ENVS)} hosts, exfil of ~40k records.",
    ])
    return wrap(rng, fmt, t, subject="SECURITY INCIDENT"), "vuln_incident"


def pos_pii_handling(rng, fmt):
    t = rng.choice([
        f"Attached is the customer export: full names, emails, DOBs, and SSNs (e.g. {ssn(rng)}) for the {rng.choice(['billing','support'])} cohort.",
        "We need to store the patients' medical record numbers, home addresses, and insurance IDs in the new table — what's the retention policy?",
        f"Here are the employee records with SSN {ssn(rng)} and bank routing numbers for payroll import.",
        "Dumping the full PII set (passport numbers, national IDs, phone numbers) to a CSV for the migration — where should it live?",
    ])
    return wrap(rng, fmt, t, subject="PII export"), "pii_handling"


POS_GENS = [pos_secret_credential, pos_access_control, pos_security_policy,
            pos_infra_config, pos_vuln_incident, pos_pii_handling]

# ---------------------------------------------------------------------------
# EASY CORE — negative generators (label=0)
# ---------------------------------------------------------------------------

def neg_helpdesk(rng, fmt):
    t = rng.choice([
        "My monitor won't turn on, the power light just blinks. Can someone swap it?",
        "I forgot my badge at home, can I get a temporary one for today?",
        "The printer on the 3rd floor is jammed again. Ticket please.",
        "My mouse stopped working after the update, tried new batteries already.",
    ])
    return wrap(rng, fmt, t, subject="helpdesk"), "helpdesk_ticket"


def neg_public_snippet(rng, fmt):
    t = rng.choice([
        "Here's a quick FizzBuzz in Python:\nfor i in range(1,101):\n    print('Fizz'*(i%3==0)+'Buzz'*(i%5==0) or i)",
        "To reverse a list in Python just use lst[::-1]. No imports needed.",
        "Tutorial: a basic Flask hello-world route is @app.route('/') def home(): return 'hi'",
    ])
    return wrap(rng, fmt, t, subject="how-to"), "public_snippet"


def neg_vendor_mention(rng, fmt):
    v1, v2 = rng.sample(VENDORS, 2)
    t = rng.choice([
        f"We're evaluating {v1} vs {v2} for next quarter, anyone used both?",
        f"Reminder: the {v1} renewal is coming up, finance needs the headcount.",
        f"Does anyone have a contact at {v1}? Want to ask about their roadmap.",
    ])
    return wrap(rng, fmt, t, subject="tooling"), "vendor_mention"


def neg_support_chatter(rng, fmt):
    t = rng.choice([
        "My laptop has been super slow since this morning, fans spinning constantly.",
        "Anyone else's wifi dropping every few minutes on the 5th floor?",
        "Zoom keeps freezing during calls, is it just me?",
        "The VPN is sluggish today, pages take forever to load.",
    ])
    return wrap(rng, fmt, t, subject="support"), "support_chatter"


NEG_GENS = [neg_helpdesk, neg_public_snippet, neg_vendor_mention, neg_support_chatter]


# ---------------------------------------------------------------------------
# HARD family 1 — counterfactual pairs (minimal edit flips the label)
# each returns (pos_text, neg_text, subcategory)
# ---------------------------------------------------------------------------

def cf_github_token(rng, fmt):
    svc = rng.choice(SERVICES)
    real = github_token(rng)
    pos = wrap(rng, fmt, f"GitHub token for {svc}: {real}", subject=f"{svc} token")
    neg = wrap(rng, fmt, "GitHub token for {}: ghp_{} (placeholder — fill in your own)".format(svc, "x" * 36), subject=f"{svc} token")
    return pos, neg, "secret_credential"


def cf_conn_string(rng, fmt):
    real = pg_conn_string(rng)
    pos = wrap(rng, fmt, f"Set DATABASE_URL={real}", subject="db url")
    neg = wrap(rng, fmt, "Set DATABASE_URL=postgres://USER:PASSWORD@HOST:5432/DB  # replace placeholders", subject="db url")
    return pos, neg, "secret_credential"


def cf_aws_key(rng, fmt):
    real = aws_access_key(rng)
    pos = wrap(rng, fmt, f"export AWS_ACCESS_KEY_ID={real}", subject="aws key")
    neg = wrap(rng, fmt, "export AWS_ACCESS_KEY_ID=<YOUR_ACCESS_KEY_ID>  # see the docs", subject="aws key")
    return pos, neg, "secret_credential"


def cf_grant_vs_request(rng, fmt):
    who, env = rng.choice(PEOPLE), rng.choice(ENVS)
    pos = wrap(rng, fmt, f"Done — granted {who} admin access to {env} and added them to the sudoers group.", subject="access")
    neg = wrap(rng, fmt, f"{who} is asking whether they could get admin access to {env} — should we approve? No change made yet.", subject="access")
    return pos, neg, "access_control"


def cf_incident_vs_news(rng, fmt):
    co = rng.choice(COMPANIES)
    pos = wrap(rng, fmt, f"We were breached last night — attacker exfiltrated our prod customer DB. Incident response is live.", subject="incident")
    neg = wrap(rng, fmt, f"Did you see the news? {co} was breached last night and lost their customer DB. Glad it wasn't us.", subject="news")
    return pos, neg, "vuln_incident"


def cf_policy_vs_advice(rng, fmt):
    pos = wrap(rng, fmt, "New company policy: all access to our prod database now requires MFA and just-in-time elevation, effective Monday.", subject="policy")
    neg = wrap(rng, fmt, "Friendly security tip: in general it's good practice to enable MFA and avoid standing access. Just sharing.", subject="tip")
    return pos, neg, "security_policy"


def cf_pii_vs_format(rng, fmt):
    s = ssn(rng)
    pos = wrap(rng, fmt, f"Here's the customer record you asked for: Jane Doe, DOB 1987-03-12, SSN {s}, 14 Elm St.", subject="record")
    neg = wrap(rng, fmt, "Quick FYI on formatting: a US SSN looks like XXX-XX-XXXX (three-two-four digits). No real numbers here.", subject="formatting")
    return pos, neg, "pii_handling"


def cf_docs_example(rng, fmt):
    # our real config (POS) vs the canonical AWS docs example creds (NEG).
    # NOTE: the example key matches the AWS regex on purpose — exposes that regex
    # can't tell a documented placeholder from a real secret; the model should.
    # Values are generated at runtime (no literal secret-shaped strings live in source).
    svc = rng.choice(SERVICES)
    pos = wrap(rng, fmt, f"{svc} prod config — API_KEY={generic_api_key(rng)}", subject="config")
    neg = wrap(rng, fmt, f"Straight from the AWS docs example, just the sample creds: {aws_access_key(rng)} / {aws_secret_key(rng)}", subject="docs example")
    return pos, neg, "secret_credential"


def cf_redacted(rng, fmt):
    real = github_token(rng)
    pos = wrap(rng, fmt, f"prod GitHub token for the deploy bot: {real}", subject="token")
    neg = wrap(rng, fmt, "prod GitHub token for the deploy bot: [REDACTED] — stripped it out before pasting into the ticket", subject="token")
    return pos, neg, "secret_credential"


def cf_env_ref(rng, fmt):
    pos = wrap(rng, fmt, f"DATABASE_URL=postgres://app:{password(rng)}@prod-db-1.internal:5432/main", subject="env")
    neg = wrap(rng, fmt, "DATABASE_URL=${DATABASE_URL}   # injected from the vault at runtime, no value stored in the repo", subject="env")
    return pos, neg, "secret_credential"


def cf_training_example(rng, fmt):
    real = password(rng)
    pos = wrap(rng, fmt, f"Sending you the prod DB password so you can run the migration: {real}", subject="db pw")
    neg = wrap(rng, fmt, "In the security training deck we use an obviously-fake throwaway password like 'hunter2' to explain why password reuse is bad.", subject="training")
    return pos, neg, "secret_credential"


def cf_synthetic_pii(rng, fmt):
    s = ssn(rng)
    pos = wrap(rng, fmt, f"Attached the real customer cohort export including names, emails and SSNs (e.g. {s}).", subject="export")
    neg = wrap(rng, fmt, "Generated 100 FAKE test users with random SSN-shaped values (e.g. 000-00-0000) for the staging seed fixture.", subject="test fixture")
    return pos, neg, "pii_handling"


def cf_drill(rng, fmt):
    pos = wrap(rng, fmt, "INCIDENT (real): we were breached overnight — attacker exfiltrated the prod customer DB. IR is live.", subject="incident")
    neg = wrap(rng, fmt, "Planning a tabletop incident DRILL for Thursday — a hypothetical breach scenario to practice runbooks. No real incident.", subject="drill")
    return pos, neg, "vuln_incident"


def cf_cve_advisory(rng, fmt):
    svc = rng.choice(SERVICES)
    pos = wrap(rng, fmt, f"Confirmed: our {svc} has an exploitable SQL injection that dumps the users table. Patching tonight.", subject="vuln")
    neg = wrap(rng, fmt, "Reading a public CVE advisory about a SQLi bug in a library we don't even use — we're not affected, just FYI.", subject="advisory")
    return pos, neg, "vuln_incident"


def cf_revoke_vs_propose(rng, fmt):
    who = rng.choice(PEOPLE)
    pos = wrap(rng, fmt, f"Revoked {who}'s prod admin access and rotated the keys they held. Done.", subject="access")
    neg = wrap(rng, fmt, f"Should we revoke {who}'s prod admin access? Opening it for discussion — nothing changed yet.", subject="access")
    return pos, neg, "access_control"


# --- v3: harder counterfactuals, leaning into the over-trigger axis ---
# All secret-shaped values below are generated at runtime; no literal keys live in source.

def stripe_live_key(rng):
    return "sk_live_" + _rand(rng, string.ascii_letters + string.digits, 24)


def stripe_test_key(rng):
    return "sk_test_" + _rand(rng, string.ascii_letters + string.digits, 24)


def google_api_key(rng):
    return "AIza" + _rand(rng, string.ascii_letters + string.digits + "-_", 35)


def cf_vendor_example_keys(rng, fmt):
    # real vendor secret (POS) vs a vendor test/example key framed as public docs (NEG).
    # The negative's "tell" is the surrounding context ("docs test key"), not the value.
    if rng.random() < 0.5:
        pos = wrap(rng, fmt, f"prod Stripe secret key: {stripe_live_key(rng)}", subject="stripe")
        neg = wrap(rng, fmt, f"just the Stripe docs test key from their quickstart, for the sample: {stripe_test_key(rng)}", subject="stripe docs")
    else:
        pos = wrap(rng, fmt, f"our Maps API key in prod: {google_api_key(rng)}", subject="maps key")
        neg = wrap(rng, fmt, f"the example key from Google's quickstart README, not ours: {google_api_key(rng)}", subject="readme")
    return pos, neg, "secret_credential"


def cf_redaction_leak(rng, fmt):
    # adversarial vs the '[REDACTED] => negative' shortcut: same prefix, but POS still
    # contains the FULL real secret (a disclosure regardless of the word 'redacted').
    real = github_token(rng)
    pos = wrap(rng, fmt, f"Redacted for the ticket: {real}", subject="token")
    neg = wrap(rng, fmt, "Redacted for the ticket: [REDACTED] — actually stripped the value this time", subject="token")
    return pos, neg, "secret_credential"


def cf_partial_last4(rng, fmt):
    # full PAN+CVC (POS) vs last-four only (NEG — non-sensitive, shown on receipts).
    pan = "4242 4242 4242 4242"
    pos = wrap(rng, fmt, f"customer's full card on file: {pan}, exp 04/27, CVC {rng.randint(100,999)}.", subject="card")
    neg = wrap(rng, fmt, f"customer's card ending in {rng.randint(1000,9999)} — just confirming the last four for support.", subject="card")
    return pos, neg, "pii_handling"


def cf_expired_vs_live(rng, fmt):
    # RUBRIC CALL (documented): an explicitly dead/revoked credential grants nothing -> PASS;
    # the current/live one -> TRIGGER. Tests whether the model reasons about validity, not shape.
    pos = wrap(rng, fmt, f"current prod AWS access key in use: {aws_access_key(rng)}", subject="aws key")
    neg = wrap(rng, fmt, f"the OLD AWS key we rotated out and revoked back in March — dead now, pasting for the audit trail: {aws_access_key(rng)}", subject="old key")
    return pos, neg, "secret_credential"


def cf_test_fixture(rng, fmt):
    pos = wrap(rng, fmt, f"prod deploy sets the real service token: {generic_api_key(rng)}", subject="token")
    neg = wrap(rng, fmt, "in test_auth.py we hardcode a dummy token 'test-token-123' for the unit test — not a real credential.", subject="unit test")
    return pos, neg, "secret_credential"


def cf_policy_question(rng, fmt):
    pos = wrap(rng, fmt, rng.choice([
        "Our policy requires MFA for all production access, enforced via SSO — that's the rule, effective now.",
        "Per our security policy, prod database access is restricted to the on-call group and every action is audited.",
    ]), subject="policy")
    neg = wrap(rng, fmt, rng.choice([
        "Does our policy actually require MFA for prod, or is it just recommended? Trying to find the answer.",
        "Anyone know if we even have a written policy about who can touch the prod database? Asking for a doc.",
    ]), subject="policy?")
    return pos, neg, "security_policy"


CF_GENS = [cf_github_token, cf_conn_string, cf_aws_key, cf_grant_vs_request,
           cf_incident_vs_news, cf_policy_vs_advice, cf_pii_vs_format,
           cf_docs_example, cf_redacted, cf_env_ref, cf_training_example,
           cf_synthetic_pii, cf_drill, cf_cve_advisory, cf_revoke_vs_propose,
           cf_vendor_example_keys, cf_redaction_leak, cf_partial_last4,
           cf_expired_vs_live, cf_test_fixture, cf_policy_question]


# ---------------------------------------------------------------------------
# HARD family 2 — intent-only positives (in scope, NO literal secret)
# ---------------------------------------------------------------------------

def intent_only(rng, fmt):
    who = rng.choice(PEOPLE)
    t, sub = rng.choice([
        ("The root password for the prod servers is sitting in the shared ops doc and hasn't been rotated in over a year.", "secret_credential"),
        (f"Heads up: I DM'd the production API key to {who} on Slack so they can test — let me know when to rotate it.", "secret_credential"),
        (f"I gave the new contractor read access to the customer PII table so they can run their queries.", "access_control"),
        ("The S3 bucket holding our nightly database backups is currently public-read. Anyone with the URL can download it.", "vuln_incident"),
        ("Someone accidentally committed our prod credentials to the public GitHub repo last night; we're rotating everything now.", "vuln_incident"),
        ("We're going to start storing patients' SSNs and home addresses in the new analytics warehouse for the dashboards.", "pii_handling"),
        ("FYI the prod database currently has no password set — it's open on the internal network to anyone.", "secret_credential"),
    ])
    return wrap(rng, fmt, t, subject="fyi"), sub


# ---------------------------------------------------------------------------
# HARD family 3 — near-boundary negatives (reference, not disclosure)
# ---------------------------------------------------------------------------

def near_boundary(rng, fmt):
    co, v = rng.choice(COMPANIES), rng.choice(VENDORS)
    t, sub = rng.choice([
        (f"Reuters: a critical OpenSSL CVE was disclosed today affecting millions of servers worldwide. Patches are rolling out.", "security_news"),
        (f"{co} disclosed a data breach this week impacting 2M users; analysts expect regulatory scrutiny.", "security_news"),
        (f"By using AcmeApp you agree to our Terms of Service and acknowledge our Privacy Policy and security practices.", "eula_boilerplate"),
        (f"{v}'s security page states they are SOC 2 Type II certified and encrypt all customer data at rest and in transit.", "vendor_security_claim"),
        ("How does Kubernetes RBAC actually work? I'm trying to understand the difference between Roles and RoleBindings.", "abstract_question"),
        ("Reminder for everyone: use a password manager and turn on 2FA for your personal accounts. Good hygiene!", "security_advice"),
        ("What's the difference between symmetric and asymmetric encryption? Reviewing for an interview.", "abstract_question"),
        (f"Blog post: 'Top 10 cloud security mistakes' — interesting read, mostly about misconfigured buckets in general.", "security_news"),
    ])
    return wrap(rng, fmt, t, subject="fyi"), sub


# ---------------------------------------------------------------------------
# HARD family 4 — obfuscated secrets (real secret, obscured -> defeats regex)
# ---------------------------------------------------------------------------

def _spaced(s):
    return " ".join(s)


def _chunked(s, n=4):
    return "-".join(s[i:i + n] for i in range(0, len(s), n))


def obfuscated(rng, fmt):
    kind = rng.randint(0, 4)
    if kind == 0:
        key = aws_access_key(rng) + aws_secret_key(rng)
        enc = base64.b64encode(key.encode()).decode()
        t = f"Don't paste secrets in plaintext — here it is base64'd, decode before use: {enc}"
    elif kind == 1:
        key = aws_access_key(rng)
        t = f"prod access key (mind the spaces): {_spaced(key)}"
    elif kind == 2:
        tok = github_token(rng)
        t = f"github token, chunked so chat doesn't linkify it: {_chunked(tok)}"
    elif kind == 3:
        pw = password(rng)
        t = f"the prod db password is (typed out): {_spaced(pw)}"
    else:
        key = generic_api_key(rng)
        enc = base64.b64encode(key.encode()).decode()
        t = f"api key for the service, base64: {enc} — decode and set as API_KEY"
    return wrap(rng, fmt, t, subject="credential (obfuscated)"), "secret_credential"


# ---------------------------------------------------------------------------
# HARD family 5 — casual long-form register (Slack / Reddit / forum)
# Real traffic is long, lowercase, rambling, emoji-laden. Our other families are short and
# clean, which is why the model under-triggers on real positives. These add that register.
# Negatives are the key new hard case: techie war-stories / PSAs / vents that DISCLOSE NOTHING.
# ---------------------------------------------------------------------------

CASUAL_OPENERS = ["", "PSA: ", "TIL ", "psa for new folks: ", "rant: ", "lpt: ", "ok so ",
                  "heads up — ", "minor win: ", "anyone else — ", "quick one: "]
CASUAL_CLOSERS = ["", " anyway hth", " carry on", " 🤷", " lol", " that's the post", " ok back to work",
                  " 🫠", " how's everyone's day", " (no i will not be taking questions)", " send help"]
CASUAL_TOOLS = ["the VPN", "Claude Code", "the CI pipeline", "Slack", "the office wifi", "Zoom",
                "the build", "Jira", "Okta SSO", "the staging env", "the printer", "our Notion"]


def _wrap_casual(rng, body):
    return (rng.choice(CASUAL_OPENERS) + body + rng.choice(CASUAL_CLOSERS)).strip()


def casual_negative(rng):
    tool = rng.choice(CASUAL_TOOLS)
    dur = rng.choice(["20 min", "an hour", "like 3 hrs", "all afternoon", "two whole days", "a solid hour"])
    body = rng.choice([
        f"if your Claude Code rate limit won't reset after a plan upgrade, just re-login. spent way "
        f"too long on this — the CLI caches your tier locally and keeps enforcing the OLD limit until "
        f"you force a fresh auth. /logout then /login fixed it for me.",
        f"you can cache node_modules in CI and it cut our build from {dur} to almost nothing. no idea "
        f"why this isn't the default tbh.",
        f"spent {dur} debugging a CORS error that turned out to be a missing trailing slash on the api "
        f"url. i need a vacation.",
        f"{tool} has been flaking out all week, third time today it dropped mid-call. not even mad "
        f"anymore just impressed at the consistency.",
        f"rotate your laptop password every quarter or IT will absolutely nag you. use a password "
        f"manager, your future self will thank you.",
        f"git reflog exists and it just saved my entire afternoon after i nuked a branch with a bad "
        f"rebase. reflog is the real mvp.",
        f"who keeps booking the big conf room for 1-person zoom calls 👀 asking for a friend (the "
        f"friend is me, i could not find a room).",
        f"is it just me or has {tool} been garbage all week. might just give up and reboot for the "
        f"tenth time.",
        f"finally migrated our docs to {tool} and honestly it was mostly copy-paste, took {dur}.",
        f"reminder that turning it off and on again genuinely fixes {tool} like 80% of the time. "
        f"i don't make the rules.",
        f"my standup ran 40 min again today. at this point it's not a standup it's a sit-down.",
        f"does anyone actually read the security awareness training or do we all just click through it",
    ])
    return _wrap_casual(rng, body), rng.choice(
        ["support_chatter", "public_snippet", "security_advice", "abstract_question"])


def casual_positive(rng):
    kind = rng.randint(0, 4)
    who = rng.choice(PEOPLE).lower()
    if kind == 0:
        body = (f"finally got the deploy green after a few hrs 😮‍💨 had to hardcode the staging token "
                f"for now bc the vault setup is still borked: {github_token(rng)} — will move it before "
                f"friday i promise")
        sub = "secret_credential"
    elif kind == 1:
        body = (f"i think i just pasted the prod db password into the wrong channel?? it was "
                f"{password(rng)}. how bad is this on a scale of 1 to update-my-resume, rotating now "
                f"but wanted to flag")
        sub = "secret_credential"
    elif kind == 2:
        body = (f"our customer csv export still has everyone's SSNs in plaintext, like {ssn(rng)}, found "
                f"it digging thru an old ticket. that's not supposed to be there right? who do i tell")
        sub = "pii_handling"
    elif kind == 3:
        body = (f"gave {who} root on the prod cluster real quick so they could tail logs during the "
                f"outage, figured it was faster than a ticket. will revoke after, lmk if thats not ok")
        sub = "access_control"
    else:
        body = ("welp. pushed our .env to a public github repo for like 10 whole minutes before i "
                "noticed. rotating all the aws keys now, lesson learned")
        sub = "vuln_incident"
    return _wrap_casual(rng, body), sub


# ---------------------------------------------------------------------------
# Noise injection — realistic human typos/contractions/dropped caps.
# Protects structured tokens (keys, emails, URLs, numbers) so secrets & labels stay intact.
# ---------------------------------------------------------------------------

CONTRACT = {"don't": "dont", "won't": "wont", "isn't": "isnt", "can't": "cant", "didn't": "didnt",
            "it's": "its", "i'm": "im", "that's": "thats", "you're": "youre", "we're": "were",
            "there's": "theres", "doesn't": "doesnt", "what's": "whats"}


def _charop(w, rng):
    if len(w) < 4:
        return w
    i = rng.randint(1, len(w) - 2)
    op = rng.randint(0, 2)
    if op == 0:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]   # transpose
    if op == 1:
        return w[:i] + w[i + 1:]                      # drop a char
    return w[:i] + w[i] + w[i:]                        # double a char


def _protected(tok):
    # leave secrets / emails / URLs / numbers / acronyms untouched
    return (any(c.isdigit() for c in tok) or any(c in tok for c in "@:/\\_")
            or len(tok) > 20 or sum(c.isupper() for c in tok) >= 3)


def inject_noise(text, rng, rate=0.08):
    out = []
    for tok in re.split(r"(\s+)", text):
        if not tok or tok.isspace():
            out.append(tok); continue
        low = tok.lower()
        if low in CONTRACT and rng.random() < 0.6:
            out.append(CONTRACT[low]); continue
        if _protected(tok):
            out.append(tok); continue
        if tok.isalpha() and len(tok) >= 4 and rng.random() < rate:
            out.append(_charop(tok, rng)); continue
        out.append(tok)
    s = "".join(out)
    if s[:1].isupper() and rng.random() < 0.4:
        s = s[0].lower() + s[1:]
    if s.endswith(".") and rng.random() < 0.3:
        s = s[:-1]
    return s


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(variant, seed, sizes, noise_rate=0.0):
    rng = random.Random(seed)
    rows = []
    sid = [0]  # mutable counter

    def add(text, label, subcat, difficulty, hardening, pair_id=None, seed_id=None, fmt="doc"):
        if seed_id is None:
            seed_id = f"it-seed-{sid[0]}"; sid[0] += 1
        rows.append(dict(
            id=f"it-{len(rows):05d}", seed_id=seed_id, domain="it", text=text, label=label,
            subcategory=subcat, difficulty=difficulty, hardening=hardening,
            pair_id=pair_id, source="synthetic", format=fmt, noisy=False,
        ))

    # --- easy core ---
    n_easy = sizes["easy"]
    for _ in range(n_easy // 2):
        fmt = rng.choice(FORMATS); text, sub = rng.choice(POS_GENS)(rng, fmt)
        add(text, 1, sub, "easy", "core", fmt=fmt)
    for _ in range(n_easy - n_easy // 2):
        fmt = rng.choice(FORMATS); text, sub = rng.choice(NEG_GENS)(rng, fmt)
        add(text, 0, sub, "easy", "core", fmt=fmt)

    if variant == "v1":
        # --- counterfactual pairs (both members share a seed_id) ---
        for k in range(sizes["cf_pairs"]):
            fmt = rng.choice(FORMATS)
            pos, neg, sub = rng.choice(CF_GENS)(rng, fmt)
            sidv = f"it-cf-{k}"; pid = f"cf-{k}"
            add(pos, 1, sub, "hard", "counterfactual", pair_id=pid, seed_id=sidv, fmt=fmt)
            add(neg, 0, sub, "hard", "counterfactual", pair_id=pid, seed_id=sidv, fmt=fmt)
        # --- intent-only positives ---
        for _ in range(sizes["intent"]):
            fmt = rng.choice(FORMATS); text, sub = intent_only(rng, fmt)
            add(text, 1, sub, "hard", "intent_only", fmt=fmt)
        # --- near-boundary negatives ---
        for _ in range(sizes["nearbound"]):
            fmt = rng.choice(FORMATS); text, sub = near_boundary(rng, fmt)
            add(text, 0, sub, "hard", "near_boundary", fmt=fmt)
        # --- obfuscated positives ---
        for _ in range(sizes["obf"]):
            fmt = rng.choice(FORMATS); text, sub = obfuscated(rng, fmt)
            add(text, 1, sub, "hard", "obfuscation", fmt=fmt)
        # --- casual long-form (Slack/Reddit register): ~half pos, half neg ---
        for k in range(sizes["casual"]):
            fmt = rng.choice(["slack", "reddit", "forum"])
            if k % 2 == 0:
                text, sub = casual_positive(rng); add(text, 1, sub, "hard", "casual", fmt=fmt)
            else:
                text, sub = casual_negative(rng); add(text, 0, sub, "hard", "casual", fmt=fmt)

    # dedup identical texts (keeps splits leakage-safe: no same string in train AND test)
    seen, deduped = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"]); deduped.append(r)
    rows = deduped

    # --- realistic noise on a fraction of rows (secrets/numbers protected) ---
    if noise_rate > 0:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        for j in idx[: int(noise_rate * len(rows))]:
            rows[j]["text"] = inject_noise(rows[j]["text"], rng)
            rows[j]["noisy"] = True

    rng.shuffle(rows)
    return rows


def split_by_seed(rows, seed, fracs=(0.7, 0.15, 0.15)):
    """Leakage-safe split: group by seed_id (so counterfactual pairs + variants co-locate)."""
    rng = random.Random(seed + 1)
    seed_ids = sorted({r["seed_id"] for r in rows})
    rng.shuffle(seed_ids)
    n = len(seed_ids)
    n_train, n_val = int(fracs[0] * n), int(fracs[1] * n)
    train_ids = set(seed_ids[:n_train]); val_ids = set(seed_ids[n_train:n_train + n_val])
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        splits["train" if r["seed_id"] in train_ids else "val" if r["seed_id"] in val_ids else "test"].append(r)
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v0", "v1"], default="v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/it")
    # v0 uses easy=800; v1 reduces easy core and adds hard families
    ap.add_argument("--easy", type=int, default=None)
    ap.add_argument("--cf-pairs", type=int, default=230)
    ap.add_argument("--intent", type=int, default=90)
    ap.add_argument("--nearbound", type=int, default=110)
    ap.add_argument("--obf", type=int, default=70)
    ap.add_argument("--casual", type=int, default=140)
    ap.add_argument("--noise-rate", type=float, default=0.4,
                    help="fraction of rows to perturb with realistic typos (secrets protected)")
    ap.add_argument("--pool", default=None,
                    help="emit one UNSPLIT candidate-pool file (path) instead of train/val/test "
                         "splits — for the co-evolution loop (rows keep their seed_id)")
    args = ap.parse_args()

    easy = args.easy if args.easy is not None else (800 if args.variant == "v0" else 500)
    sizes = dict(easy=easy, cf_pairs=args.cf_pairs, intent=args.intent,
                 nearbound=args.nearbound, obf=args.obf, casual=args.casual)

    rows = build(args.variant, args.seed, sizes, noise_rate=args.noise_rate)

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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")

    def dist(items, key):
        d = {}
        for r in items:
            d[r[key]] = d.get(r[key], 0) + 1
        return dict(sorted(d.items()))

    print(f"Variant {args.variant} — total rows: {len(rows)}")
    for name, items in splits.items():
        pos = sum(r["label"] for r in items)
        hard = sum(1 for r in items if r["difficulty"] == "hard")
        print(f"  {name}: {len(items)} ({pos} pos / {len(items)-pos} neg, {hard} hard)")
    print("Hardening dist:", dist(rows, "hardening"))
    print("Difficulty dist:", dist(rows, "difficulty"))
    print("Subcategory dist:", dist(rows, "subcategory"))
    print(f"Noisy rows: {sum(1 for r in rows if r['noisy'])} / {len(rows)}")

    # maintain data/domains.json
    domain = out.name
    manifest = out.parent / "domains.json"
    domains = json.loads(manifest.read_text()) if manifest.exists() else []
    if domain not in domains:
        domains.append(domain)
        manifest.write_text(json.dumps(sorted(domains)))
        print(f"Registered domain '{domain}' in {manifest}")


if __name__ == "__main__":
    main()
