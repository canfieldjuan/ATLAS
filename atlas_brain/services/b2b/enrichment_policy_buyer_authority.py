from __future__ import annotations

import re


POST_PURCHASE_REVIEW_SOURCES = frozenset({
    "g2",
    "gartner",
    "trustradius",
    "capterra",
    "software_advice",
    "peerspot",
    "sourceforge",
    "trustpilot",
})

POST_PURCHASE_USAGE_PATTERNS = (
    "we use",
    "we've used",
    "we have used",
    "been using",
    "using this",
    "using it",
    "in production",
    "implemented",
    "deployed",
    "rolled out",
    "adopted",
    "renewed",
    "customer since",
    "our team uses",
    "our company uses",
)

ROLE_TYPE_ALIASES = {
    "economicbuyer": "economic_buyer",
    "decisionmaker": "economic_buyer",
    "buyer": "economic_buyer",
    "budgetowner": "economic_buyer",
    "executive": "economic_buyer",
    "director": "economic_buyer",
    "champion": "champion",
    "manager": "champion",
    "teamlead": "champion",
    "lead": "champion",
    "evaluator": "evaluator",
    "admin": "evaluator",
    "administrator": "evaluator",
    "analyst": "evaluator",
    "architect": "evaluator",
    "enduser": "end_user",
    "user": "end_user",
    "ic": "end_user",
    "individualcontributor": "end_user",
    "unknown": "unknown",
}

ROLE_LEVEL_ALIASES = {
    "executive": "executive",
    "exec": "executive",
    "csuite": "executive",
    "cxo": "executive",
    "ceo": "executive",
    "cto": "executive",
    "cfo": "executive",
    "cio": "executive",
    "cmo": "executive",
    "coo": "executive",
    "cro": "executive",
    "president": "executive",
    "founder": "executive",
    "owner": "executive",
    "executivedirector": "executive",
    "presidentfounder": "executive",
    "ownermanagingmember": "executive",
    "ed": "executive",
    "director": "director",
    "vp": "director",
    "vicepresident": "director",
    "head": "director",
    "directeur": "director",
    "managingdirector": "director",
    "headofcustomerexperience": "director",
    "manager": "manager",
    "lead": "manager",
    "teamlead": "manager",
    "supervisor": "manager",
    "coordinator": "manager",
    "projectmanager": "manager",
    "programmanager": "manager",
    "productmanager": "manager",
    "marketingmanager": "manager",
    "digitalmarketingmanager": "manager",
    "salesmanager": "manager",
    "operationsmanager": "manager",
    "itmanager": "manager",
    "businessdevelopmentmanager": "manager",
    "clientservicemanager": "manager",
    "customersuccessmanager": "manager",
    "pmo": "manager",
    "bdm": "manager",
    "leadconsultant": "manager",
    "projectmanagement": "manager",
    "ic": "ic",
    "individualcontributor": "ic",
    "individual": "ic",
    "user": "ic",
    "product": "ic",
    "marketing": "ic",
    "digitalmarketing": "ic",
    "consultant": "ic",
    "customersupport": "ic",
    "customersuccess": "ic",
    "humanresources": "ic",
    "softwaredevelopment": "ic",
    "it": "ic",
    "devops": "ic",
    "swe": "ic",
    "fse": "ic",
    "cybersecurityanalyst": "ic",
    "chemicalengineer": "ic",
    "industrialengineer": "ic",
    "customersatisfactionandqa": "ic",
    "marketingteam": "ic",
}

EXEC_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(vp\b|vice president|director|head of|chief|cfo|ceo|coo|cio|cto|cro|cmo|founder|owner|president|executive director|managing member)\b",
    re.I,
)
CHAMPION_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(manager|lead|team lead|supervisor|coordinator|pmo|project management|bdm)\b",
    re.I,
)
EVALUATOR_REVIEWER_TITLE_PATTERN = re.compile(
    r"\b(analyst|architect|engineer|developer|administrator|admin|consultant|specialist|devops|qa|customer support|customer success|human resources|marketing|product|software development|cybersecurity|it\b|swe\b|fse\b)\b",
    re.I,
)
EXEC_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(ceo|cto|cfo|cio|coo|cmo|cro|chief|founder|owner|president)\b",
    re.I,
)
DIRECTOR_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(vp|vice president|svp|evp|director|head of)\b",
    re.I,
)
MANAGER_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(manager|team lead|lead|supervisor|coordinator)\b",
    re.I,
)
IC_ROLE_TEXT_PATTERN = re.compile(
    r"\b(i am|i'm|as|working as|as the)\s+(an?\s+|the\s+)?"
    r"(engineer|developer|administrator|admin|analyst|specialist|"
    r"consultant|marketer|designer|architect)\b",
    re.I,
)
ECONOMIC_BUYER_TEXT_PATTERNS = (
    re.compile(
        r"\b(we|i) decided to (switch|move|migrate|leave|replace|renew|buy|adopt|go with)\b",
        re.I,
    ),
    re.compile(r"\bapproved (the )?(purchase|renewal|budget)\b", re.I),
    re.compile(r"\bsigned off on (the )?(purchase|renewal|budget|migration)\b", re.I),
    re.compile(r"\bfinal decision (was|is) to\b", re.I),
)
CHAMPION_TEXT_PATTERNS = (
    re.compile(r"\b(i|we) recommended\b", re.I),
    re.compile(r"\bchampioned\b", re.I),
    re.compile(r"\bpushed for\b", re.I),
    re.compile(r"\badvocated for\b", re.I),
)
EVALUATOR_TEXT_PATTERNS = (
    re.compile(r"\bevaluating alternatives\b", re.I),
    re.compile(r"\bcomparing options\b", re.I),
    re.compile(r"\bproof of concept\b", re.I),
    re.compile(r"\bpoc\b", re.I),
    re.compile(r"\bshortlist\b", re.I),
    re.compile(r"\btrialing\b", re.I),
    re.compile(r"\bpiloting\b", re.I),
    re.compile(r"\btasked with evaluating\b", re.I),
)
END_USER_TEXT_PATTERNS = (
    re.compile(r"\bi use\b", re.I),
    re.compile(r"\bwe use\b", re.I),
    re.compile(r"\bday-to-day\b", re.I),
    re.compile(r"\bdaily use\b", re.I),
    re.compile(r"\buse it for\b", re.I),
)
MANAGER_DECISION_TITLE_PATTERN = re.compile(
    r"\b(operations manager|it manager|project manager|program manager|product manager|marketing manager|sales manager|business development manager|client service manager|customer success manager|team lead|lead consultant|pmo|bdm|security manager|risk management)\b",
    re.I,
)
COMMERCIAL_DECISION_TEXT_PATTERN = re.compile(
    r"\b(renewal|quote|quoted|pricing|price increase|budget|contract|procurement|vendor selection|selected|chose|approved|sign(?:ed)? off|purchase|buying committee|rfp|rfq|evaluate|evaluation|migration)\b",
    re.I,
)
