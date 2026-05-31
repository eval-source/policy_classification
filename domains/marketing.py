"""Marketing / GTM domain spec — outbound messaging, claims, and commercial terms.

Operating point: false positives are costly (crying wolf on internal comms), but unsubstantiated
claims are a regulatory risk → roughly balanced, slightly precision-favoring. Regex is weak here
(superlatives/promo tokens appear in both real claims and casual opinions) — a small set only.
"""
import re
from domains._base import DomainSpec

POLICY = """\
POLICY — Marketing / GTM (outbound messaging, claims, and commercial terms).

A text TRIGGERS this policy if it is or contains ANY of:
  - a product/efficacy or competitive claim (especially unsubstantiated)
  - pricing, discount, or promotional terms
  - brand-voice / positioning or campaign copy (outbound)
  - a press / PR statement subject to advertising regulation

A text DOES NOT trigger if it is:
  - internal campaign scheduling, calendar invites, or team logistics
  - a third-party market-research or industry-trend fact
  - a generic mention of the word "marketing"
  - someone's personal opinion about an ad"""

FEWSHOT = [
    ("New tagline for the launch: \"CloudSync is 3x faster than any competitor — guaranteed.\"", 1),
    ("Reminder: the campaign creative review is Thursday 2pm, please add your decks to the folder.", 0),
    ("Promo: 40% off all annual plans through Friday — use code LAUNCH40 at checkout.", 1),
    ("Gartner reports the martech market grew 12% last year; useful context for planning.", 0),
    ("Press release draft: \"Acme today announced the industry's first AI-native CRM, setting a new "
     "standard for sales teams worldwide.\"", 1),
    ("honestly that new competitor ad is kind of cringe, did anyone else see it?", 0),
    ("Landing page hero: \"Clinically proven to reduce churn by 50%. Join 10,000+ happy teams.\"", 1),
    ("Can we move the marketing standup to 10am? The 9am clashes with the eng sync.", 0),
    ("We're positioning this as the premium, enterprise-grade option — lean into 'trusted by Fortune 500'.", 1),
    ("Marketing is hiring a content designer this quarter; share the JD if you know anyone.", 0),
]

POS_KIND = {
    "efficacy_competitive_claim": "a product/efficacy or competitive claim (regulated marketing content)",
    "pricing_promo": "pricing, discount, or promotional terms",
    "brand_campaign_copy": "brand positioning or outbound campaign copy",
    "press_pr": "a press/PR statement subject to advertising regulation",
}
POS_EXTRA = {
    "intent_only": " It states the outbound claim/term even without the full creative.",
    "casual": " Although written casually, it is outbound claim/positioning content.",
}
NEG_WHY = {
    "counterfactual": ("it is internal campaign logistics, a third-party market fact, or a personal "
                       "opinion about an ad — not an outbound claim/term/copy we are publishing"),
    "near_boundary": ("it is a third-party market-research fact, a generic mention of 'marketing', or "
                      "a personal opinion about an ad — not regulated outbound content"),
    "casual": "it is casual chatter about marketing with no outbound claim or term",
    "core": "it is routine internal/non-marketing content",
}
NEG_WHY_SUBCAT = {
    "internal_logistics": "it is internal scheduling/logistics, not outbound content",
    "market_fact": "it is a third-party market fact, not a claim we are making",
}

# Weak regex: promo tokens + common superlative/efficacy claim markers. Low precision (they also
# appear in opinions/news) — included only as a baseline signal, not relied upon.
PATTERNS = {
    "promo_pct": re.compile(r"\b\d{1,2}%\s*off\b", re.I),
    "promo_code": re.compile(r"\b(?:use\s+code|promo\s*code)\b", re.I),
    "superlative_claim": re.compile(r"\b(?:guaranteed|clinically proven|#1|best[- ]in[- ]class|"
                                    r"\d+x faster|world'?s first|industry'?s first)\b", re.I),
}

SPEC = DomainSpec(
    name="marketing", policy=POLICY, fewshot=FEWSHOT,
    pos_kind=POS_KIND, pos_extra=POS_EXTRA, neg_why=NEG_WHY, neg_why_subcat=NEG_WHY_SUBCAT,
    regex_patterns=PATTERNS, placeholder_hints=None, suppress_in_placeholder=(),
    hedge="the text is about marketing",
    default_pos_kind="regulated outbound marketing content",
    default_neg_why="it is not outbound claim/term/copy",
)
