"""
Edge offline skill system.

Provides local query handling for time, timers, math, and status
queries — both as a fast path when the brain is online and as a
fallback when offline.
"""

from typing import Optional

from .registry import SkillRegistry, SkillRouter

_router: Optional[SkillRouter] = None


def get_skill_router() -> SkillRouter:
    """Get or create the global skill router with all built-in skills."""
    global _router
    if _router is None:
        from ..config import settings

        registry = SkillRegistry()

        from .time_skill import TimeSkill
        from .timer_skill import TimerSkill
        from .math_skill import MathSkill
        from .status_skill import StatusSkill

        registry.register(TimeSkill(timezone=settings.skills.timezone))
        registry.register(TimerSkill(max_timers=settings.skills.max_timers))
        registry.register(MathSkill())
        registry.register(StatusSkill())

        _router = SkillRouter(registry)
    return _router


def shutdown_skills() -> None:
    """Clean up skill resources (cancel timers, etc.)."""
    if _router is None:
        return
    for skill in _router._registry.all_skills():
        if hasattr(skill, "shutdown"):
            skill.shutdown()
