from __future__ import annotations

import re


TIMELINE_IMMEDIATE_PATTERNS = ("asap", "immediately", "right away", "this week", "today", "urgent")
TIMELINE_QUARTER_PATTERNS = ("next quarter", "this quarter", "q1", "q2", "q3", "q4", "30 days", "60 days", "90 days")
TIMELINE_YEAR_PATTERNS = ("this year", "next year", "12 months", "end of year", "2026", "2027")
TIMELINE_DECISION_PATTERNS = (
    "decide", "decision", "renewal", "contract", "evaluate", "evaluation",
    "considering", "switch", "switching", "migration", "migrate",
    "deadline", "cutover", "go live", "go-live",
)
TIMELINE_EXPLICIT_ANCHOR_PHRASES = (
    "end of quarter", "quarter end", "end of month", "month end",
    "end of year", "next quarter", "this quarter", "next month", "this month",
    "this week", "next week", "a few weeks", "few weeks", "a few days", "few days",
    "30 days", "60 days", "90 days", "12 months", "next year", "this year",
    "asap", "immediately", "right away", "today", "tomorrow",
)
TIMELINE_RELATIVE_ANCHOR_RE = re.compile(
    r"\b(?:\d+\s*-\s*\d+|\d+|one|two|three|four|five|six|seven|eight|nine|ten|a few|few)"
    r"(?:\s+to\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))?"
    r"\s+(?:business\s+days?|days?|weeks?|months?)\b",
    re.IGNORECASE,
)
TIMELINE_MONTH_DAY_RE = re.compile(
    r"\b(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|"
    r"aug|august|sep|sept|september|oct|october|nov|november|dec|december)\.?"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
TIMELINE_SLASH_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
TIMELINE_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TIMELINE_CONTRACT_END_PATTERNS = (
    "contract end", "contract ends", "contract expires", "expiration date",
    "expiry date", "renewal date", "renewal window", "term ends", "term expires",
    "auto renew", "auto-renew", "automatic renewal", "at renewal", "upon renewal",
    "final month of", "current contract",
)
TIMELINE_DECISION_DEADLINE_PATTERNS = (
    "notice", "notice period", "before renewal", "before the contract ends",
    "before the contract expires", "deadline", "decide", "decision", "evaluating",
    "evaluation", "considering", "switch", "switching", "migrate", "migration",
    "cutover", "go live", "go-live", "cancel by",
)
TIMELINE_CONTRACT_EVENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:at|upon)\s+(?:the\s+)?renewal\b", re.I),
    re.compile(r"\b(?:auto[- ]?renew(?:al)?|annual renewal|next renewal|renewal date|renewal window)\b", re.I),
    re.compile(r"\bfinal month of (?:my|our|the) current contract\b", re.I),
    re.compile(r"\b(?:current|existing)\s+contract\b", re.I),
)
TIMELINE_AMBIGUOUS_VENDOR_TOKENS = {"copper", "close"}
TIMELINE_AMBIGUOUS_VENDOR_PRODUCT_CONTEXT_PATTERNS = (
    "crm", "sales", "pipeline", "lead", "leads", "deal", "deals", "account",
    "contact", "contacts", "prospect", "prospects", "software", "saas",
)

BUDGET_CURRENCY_TOKEN_RE = re.compile(
    r"(?P<raw>(?:\$|usd\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?)",
    re.IGNORECASE,
)
BUDGET_ANY_AMOUNT_TOKEN_RE = re.compile(
    r"(?:\$|usd\s*|\u20ac|eur\s*|\u00a3|gbp\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?",
    re.IGNORECASE,
)
BUDGET_ANNUAL_AMOUNT_RE = re.compile(
    r"(?P<raw>(?:\$|usd\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?)"
    r"\s*(?P<period>(?:/\s*|\bper\b\s*|\ba\b\s*)?(?:yr|year)\b|annually\b|annual\b|yearly\b)",
    re.IGNORECASE,
)
BUDGET_PRICE_PER_SEAT_RE = re.compile(
    r"(?P<raw>(?:\$|usd\s*)\s?\d[\d,]*(?:\.\d+)?\s*(?:[km])?)"
    r"\s*(?:/|\bper\b)\s*(?:seat|user|license|licence)\b"
    r"(?:\s*(?:/|\bper\b)\s*(?:monthly|month|mo|annually|annual|year|yr))?",
    re.IGNORECASE,
)
BUDGET_SEAT_COUNT_RE = re.compile(
    r"\b(?P<count>\d[\d,]{0,6})\s+(?P<unit>seats?|users?|licenses?|licences?)\b",
    re.IGNORECASE,
)
BUDGET_PRICE_INCREASE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%\s+(?:price\s+)?(?:increase|higher|more|jump|hike)"
    r"|(?:price|pricing|renewal)\s+(?:increase|jump|hike)"
    r"|(?:raised|increased)\s+(?:our\s+)?(?:price|pricing|renewal|invoice))\b",
    re.IGNORECASE,
)
BUDGET_PRICE_INCREASE_DETAIL_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?%\s+(?:price\s+)?(?:increase|higher|more|jump|hike)"
    r"|(?:price|pricing|renewal)\s+(?:increase|jump|hike)[^.!,;]{0,80}"
    r"|(?:raised|increased)[^.!,;]{0,80})",
    re.IGNORECASE,
)
BUDGET_COMMERCIAL_CONTEXT_PATTERNS = (
    "pricing", "price", "priced", "cost", "costs", "costly", "expensive",
    "budget", "billing", "invoice", "overcharg", "renewal", "quote", "quoted",
    "contract", "subscription", "license", "licence", "plan", "seat", "user",
)
BUDGET_ANNUAL_CONTEXT_PATTERNS = (
    "renewal", "quote", "quoted", "contract", "subscription", "license",
    "licence", "annual", "annually", "yearly", "per year", "/year", "/yr",
)
BUDGET_MONTHLY_PERIOD_PATTERNS = (
    "monthly", "per month", "/month", "/mo", "a month",
)
BUDGET_ANNUAL_PERIOD_PATTERNS = (
    "annual", "annually", "yearly", "per year", "/year", "/yr", "a year", "a yr",
)
BUDGET_PER_UNIT_PATTERNS = (
    "per seat", "/seat", "per user", "/user", "per license", "/license",
    "per licence", "/licence", "per agent", "/agent", "per person", "/person",
    "per employee", "/employee", "per endpoint", "/endpoint", "per device", "/device",
    "per member", "/member", "per contact", "/contact",
)
BUDGET_NOISE_PATTERNS = (
    "salary", "salaries", "compensation", "bonus", "payroll", "hourly",
    "per hour", "an hour", "wage", "wages", "job offer", "interview", "intern",
    "income", "revenue", "profit", "arr", "mrr", "valuation", "mortgage",
    "rent", "tuition", "commission",
)
