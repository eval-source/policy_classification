"""Legal domain spec — binding/negotiated legal instruments and obligations.

Note: unlike IT, regex is ~useless here — binding instruments and public boilerplate share the
same legalese ("hereinafter", "the parties agree"), so there's no high-precision literal signal.
regex_patterns is intentionally empty; the discriminative work is semantic (handled by the model).
Operating point: misses are costly (a missed contract clause) → favor RECALL.
"""
from domains._base import DomainSpec

POLICY = """\
POLICY — Legal (binding / negotiated legal instruments and obligations).

A text TRIGGERS this policy if it is or discloses ANY of:
  - an NDA or confidentiality agreement
  - a contract / MSA / SOW / DPA (negotiated, party-specific terms)
  - an indemnification, liability, or warranty clause
  - privileged material or litigation / dispute language
  - a binding contractual compliance obligation

A text DOES NOT trigger if it is:
  - a website EULA, terms-of-service, or privacy-policy page (public boilerplate)
  - general legal news or commentary
  - a casual reference to "legal" (the team; "legally speaking")
  - internal logistics about the legal department (meetings, staffing)"""

FEWSHOT = [
    ("Section 4. Confidentiality. Each party shall hold the other's Confidential Information in "
     "strict confidence and not disclose it to any third party for a period of five (5) years.", 1),
    ("By using this website you agree to our Terms of Service and acknowledge our Privacy Policy.", 0),
    ("The Supplier shall indemnify and hold harmless the Client against any and all losses arising "
     "from breach of this Agreement.", 1),
    ("New op-ed: why indemnification clauses are reshaping SaaS contracts in 2026. Interesting read.", 0),
    ("Per our executed MSA with Acme, the SOW pricing is locked for 12 months and net-30 applies.", 1),
    ("legally speaking we should probably double-check, but I'm not a lawyer lol", 0),
    ("Attached is the privileged memo from outside counsel re: the pending Globex litigation — do "
     "not forward, attorney-client privileged.", 1),
    ("Reminder: the legal team's weekly sync moved to 3pm Thursdays in the Oak room.", 0),
    ("We are contractually obligated under the DPA to notify the controller within 24 hours of any "
     "personal-data breach.", 1),
    ("Here's a blank NDA template you can fill in with your own party names and dates.", 0),
]

POS_KIND = {
    "nda_confidentiality": "a binding NDA or confidentiality agreement",
    "contract": "a negotiated contract / MSA / SOW / DPA",
    "indemnity_liability": "an indemnification, liability, or warranty clause",
    "privilege_litigation": "privileged material or litigation/dispute language",
    "compliance_obligation": "a binding contractual compliance obligation",
}
POS_EXTRA = {
    "intent_only": " It states the binding obligation even without quoting the full clause.",
    "casual": " Although written casually, it references/handles a binding instrument.",
}
NEG_WHY = {
    "counterfactual": ("it is public boilerplate (EULA/ToS/privacy policy), a blank template, or "
                       "commentary about a legal topic — not a negotiated binding instrument"),
    "near_boundary": ("it is general legal news/commentary, a casual reference to 'legal', or "
                      "internal legal-department logistics — not a binding instrument"),
    "casual": "it is casual chatter mentioning legal matters with no binding obligation",
    "core": "it is routine, non-legal content",
}
NEG_WHY_SUBCAT = {
    "eula_tos_privacy": "it is public-facing boilerplate (terms/privacy policy), not a negotiated agreement",
    "legal_news": "it is legal news/commentary about the topic, not a binding instrument",
}

SPEC = DomainSpec(
    name="legal", policy=POLICY, fewshot=FEWSHOT,
    pos_kind=POS_KIND, pos_extra=POS_EXTRA, neg_why=NEG_WHY, neg_why_subcat=NEG_WHY_SUBCAT,
    regex_patterns={},  # no useful literal signal for legal (binding vs boilerplate share legalese)
    hedge="the text mentions legal matters",
    default_pos_kind="a binding legal instrument or obligation",
    default_neg_why="it is not a binding negotiated instrument",
)
