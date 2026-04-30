from __future__ import annotations


CHURN_SIGNAL_BOOL_FIELDS = (
    "intent_to_leave",
    "actively_evaluating",
    "migration_in_progress",
    "support_escalation",
    "contract_renewal_mentioned",
)

KNOWN_SEVERITY_LEVELS = {"primary", "secondary", "minor"}
KNOWN_LOCK_IN_LEVELS = {"high", "medium", "low", "unknown"}
KNOWN_SENTIMENT_DIRECTIONS = {"declining", "consistently_negative", "improving", "stable_positive", "unknown"}
KNOWN_ROLE_TYPES = {"economic_buyer", "champion", "evaluator", "end_user", "unknown"}
KNOWN_ROLE_LEVELS = {"executive", "director", "manager", "ic", "unknown"}
KNOWN_BUYING_STAGES = {"active_purchase", "evaluation", "renewal_decision", "post_purchase", "unknown"}
KNOWN_DECISION_TIMELINES = {"immediate", "within_quarter", "within_year", "unknown"}
KNOWN_CONTRACT_VALUE_SIGNALS = {"enterprise_high", "enterprise_mid", "mid_market", "smb", "unknown"}
KNOWN_REPLACEMENT_MODES = {
    "competitor_switch", "bundled_suite_consolidation", "workflow_substitution",
    "internal_tool", "none",
}
KNOWN_OPERATING_MODEL_SHIFTS = {
    "sync_to_async", "chat_to_docs", "chat_to_ticketing", "consolidation", "none",
}
KNOWN_PRODUCTIVITY_DELTA_CLAIMS = {"more_productive", "less_productive", "no_change", "unknown"}
KNOWN_ORG_PRESSURE_TYPES = {
    "procurement_mandate", "standardization_mandate", "bundle_pressure",
    "budget_freeze", "none",
}
KNOWN_CONTENT_TYPES = {"review", "community_discussion", "comment", "insider_account"}
KNOWN_ORG_HEALTH_LEVELS = {"high", "medium", "low", "unknown"}
KNOWN_LEADERSHIP_QUALITIES = {"poor", "mixed", "good", "unknown"}
KNOWN_INNOVATION_CLIMATES = {"stagnant", "declining", "healthy", "unknown"}
KNOWN_MORALE_LEVELS = {"high", "medium", "low", "unknown"}
KNOWN_DEPARTURE_TYPES = {"voluntary", "involuntary", "still_employed", "unknown"}
