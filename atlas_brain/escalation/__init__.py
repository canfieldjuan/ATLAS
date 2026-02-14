"""
Escalation system for edge security events.

Classifies security events as routine or escalation-worthy,
and triggers LLM synthesis for high-priority alerts.

Usage:
    from atlas_brain.escalation import get_escalation_evaluator

    evaluator = get_escalation_evaluator()
    result = await evaluator.evaluate(event_type, message, node_id)
"""

from typing import Optional

from .evaluator import EscalationEvaluator, EscalationResult

_evaluator: Optional[EscalationEvaluator] = None


def get_escalation_evaluator() -> EscalationEvaluator:
    """Get or create the singleton EscalationEvaluator."""
    global _evaluator
    if _evaluator is None:
        _evaluator = EscalationEvaluator()
    return _evaluator


__all__ = [
    "EscalationEvaluator",
    "EscalationResult",
    "get_escalation_evaluator",
]
