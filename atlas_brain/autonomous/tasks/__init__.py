"""Built-in autonomous task handlers."""


def register_builtin_tasks(runner) -> None:
    """Register all builtin task handlers."""
    from .security_summary import run as security_summary_run
    from .device_health import run as device_health_run
    from .morning_briefing import run as morning_briefing_run
    from .gmail_digest import run as gmail_digest_run
    from .proactive_actions import run as proactive_actions_run
    from .departure_check import run as departure_check_run

    runner.register_builtin("security_summary", security_summary_run)
    runner.register_builtin("device_health_check", device_health_run)
    runner.register_builtin("morning_briefing", morning_briefing_run)
    runner.register_builtin("gmail_digest", gmail_digest_run)
    runner.register_builtin("proactive_actions", proactive_actions_run)
    runner.register_builtin("departure_check", departure_check_run)
