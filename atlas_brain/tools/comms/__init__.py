"""
Communications tools for email, notifications, and messaging.

Used by receptionist mode for business communications and comms mode for personal use.
"""

from .email import (
    EmailTool,
    email_tool,
    EstimateEmailTool,
    estimate_email_tool,
    ProposalEmailTool,
    proposal_email_tool,
)
from .notify import NotifyTool, notify_tool

__all__ = [
    # Email
    "EmailTool",
    "email_tool",
    "EstimateEmailTool",
    "estimate_email_tool",
    "ProposalEmailTool",
    "proposal_email_tool",
    # Notifications
    "NotifyTool",
    "notify_tool",
]
