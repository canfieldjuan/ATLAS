"""Built-in autonomous task handlers."""

import importlib
import logging

logger = logging.getLogger("atlas.autonomous.tasks")

# (module_name, attribute, task_name)
_BUILTIN_TASKS = [
    ("security_summary", "run", "security_summary"),
    ("device_health", "run", "device_health_check"),
    ("morning_briefing", "run", "morning_briefing"),
    ("gmail_digest", "run", "gmail_digest"),
    ("proactive_actions", "run", "proactive_actions"),
    ("departure_check", "run", "departure_check"),
    ("departure_auto_fix", "run", "departure_auto_fix"),
    ("calendar_reminder", "run", "calendar_reminder"),
    ("action_escalation", "run", "action_escalation"),
    ("preference_learning", "run", "preference_learning"),
    ("pattern_learning", "run", "pattern_learning"),
    ("anomaly_detection", "run", "anomaly_detection"),
    ("email_draft", "run", "email_draft"),
    ("email_intake", "run", "email_intake"),
    ("model_swap", "run_day", "model_swap_day"),
    ("model_swap", "run_night", "model_swap_night"),
    ("email_backfill", "run", "email_backfill"),
    ("email_auto_approve", "run", "email_auto_approve"),
    ("email_stale_check", "run", "email_stale_check"),
    ("invoice_overdue_check", "run", "invoice_overdue_check"),
    ("invoice_payment_reminders", "run", "invoice_payment_reminders"),
    ("monthly_invoice_generation", "run", "monthly_invoice_generation"),
    ("reasoning_tick", "run", "reasoning_tick"),
    ("reasoning_reflection", "run", "reasoning_reflection"),
    ("news_intake", "run", "news_intake"),
    ("market_intake", "run", "market_intake"),
    ("daily_intelligence", "run", "daily_intelligence"),
    ("article_enrichment", "run", "article_enrichment"),
]


def _safe_register(runner, module_name: str, attr: str, task_name: str) -> None:
    """Import and register a single task, logging on failure."""
    try:
        mod = importlib.import_module(f".{module_name}", package=__package__)
        runner.register_builtin(task_name, getattr(mod, attr))
    except Exception:
        logger.error("Failed to register task %s", task_name, exc_info=True)


def register_builtin_tasks(runner) -> None:
    """Register all builtin task handlers."""
    for module_name, attr, task_name in _BUILTIN_TASKS:
        _safe_register(runner, module_name, attr, task_name)
