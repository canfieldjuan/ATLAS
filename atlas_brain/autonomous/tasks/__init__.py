"""Built-in autonomous task handlers."""


def register_builtin_tasks(runner) -> None:
    """Register all builtin task handlers."""
    from .security_summary import run as security_summary_run
    from .device_health import run as device_health_run
    from .morning_briefing import run as morning_briefing_run
    from .gmail_digest import run as gmail_digest_run
    from .proactive_actions import run as proactive_actions_run
    from .departure_check import run as departure_check_run
    from .departure_auto_fix import run as departure_auto_fix_run
    from .calendar_reminder import run as calendar_reminder_run
    from .action_escalation import run as action_escalation_run
    from .preference_learning import run as preference_learning_run
    from .pattern_learning import run as pattern_learning_run
    from .anomaly_detection import run as anomaly_detection_run
    from .email_draft import run as email_draft_run
    from .email_intake import run as email_intake_run
    from .model_swap import run_day as model_swap_day_run, run_night as model_swap_night_run
    from .email_backfill import run as email_backfill_run
    from .email_auto_approve import run as email_auto_approve_run
    from .email_stale_check import run as email_stale_check_run
    from .invoice_overdue_check import run as invoice_overdue_check_run
    from .invoice_payment_reminders import run as invoice_payment_reminders_run
    from .monthly_invoice_generation import run as monthly_invoice_generation_run
    from .reasoning_tick import run as reasoning_tick_run
    from .reasoning_reflection import run as reasoning_reflection_run

    runner.register_builtin("security_summary", security_summary_run)
    runner.register_builtin("device_health_check", device_health_run)
    runner.register_builtin("morning_briefing", morning_briefing_run)
    runner.register_builtin("gmail_digest", gmail_digest_run)
    runner.register_builtin("proactive_actions", proactive_actions_run)
    runner.register_builtin("departure_check", departure_check_run)
    runner.register_builtin("departure_auto_fix", departure_auto_fix_run)
    runner.register_builtin("calendar_reminder", calendar_reminder_run)
    runner.register_builtin("action_escalation", action_escalation_run)
    runner.register_builtin("preference_learning", preference_learning_run)
    runner.register_builtin("pattern_learning", pattern_learning_run)
    runner.register_builtin("anomaly_detection", anomaly_detection_run)
    runner.register_builtin("email_draft", email_draft_run)
    runner.register_builtin("email_intake", email_intake_run)
    runner.register_builtin("model_swap_day", model_swap_day_run)
    runner.register_builtin("model_swap_night", model_swap_night_run)
    runner.register_builtin("email_backfill", email_backfill_run)
    runner.register_builtin("email_auto_approve", email_auto_approve_run)
    runner.register_builtin("email_stale_check", email_stale_check_run)
    runner.register_builtin("invoice_overdue_check", invoice_overdue_check_run)
    runner.register_builtin("invoice_payment_reminders", invoice_payment_reminders_run)
    runner.register_builtin("monthly_invoice_generation", monthly_invoice_generation_run)
    runner.register_builtin("reasoning_tick", reasoning_tick_run)
    runner.register_builtin("reasoning_reflection", reasoning_reflection_run)
