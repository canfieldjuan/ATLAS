from __future__ import annotations


LOW_FIDELITY_TOKEN_STOPWORDS = {
    "and", "for", "the", "with", "cloud", "software", "platform",
}

LOW_FIDELITY_COMMERCIAL_MARKERS = {
    "alternative", "alternatives", "budget", "contract", "cost", "expensive",
    "migrate", "migration", "pricing", "renewal", "replace", "replaced",
    "seat", "seats", "support", "switch", "switching",
}

LOW_FIDELITY_STRONG_COMMERCIAL_MARKERS = {
    "alternative", "alternatives", "budget", "contract", "cost", "expensive",
    "migrate", "migration", "pricing", "renewal", "replace", "replaced",
    "seat", "seats", "switch", "switching",
}

LOW_FIDELITY_TECHNICAL_PATTERNS = (
    r"\bhow (?:can|do|to)\b",
    r"\bbest practice\b",
    r"\bsetting up\b",
    r"\banswer to question\b",
    r"\bapi token\b",
    r"\bbuild pipeline\b",
    r"\bconnect(?:ing)?\b",
    r"\bcosmos db\b",
    r"\bdocker\b",
    r"\berror\b",
    r"\bfailed\b",
    r"\bintegrat(?:e|ion)\b",
    r"\bjenkins\b",
    r"\bkey vault\b",
    r"\blogin\b",
    r"\bplugin\b",
    r"\breact frontend\b",
    r"\bssl verification failed\b",
    r"\bsubscription form\b",
    r"\bvagrant\b",
    r"\bxamarin\b",
)

LOW_FIDELITY_CONSUMER_PATTERNS = (
    r"\b2fa\b",
    r"\bapp support\b",
    r"\bdownloaded\b",
    r"\bghosting email\b",
    r"\bgoogle play\b",
    r"\bhacked\b",
    r"\bminecraft\b",
    r"\bmy son\b",
    r"\boutlook app\b",
    r"\btaskbar\b",
    r"\bwindows 11\b",
    r"\bworkspace account\b",
)
