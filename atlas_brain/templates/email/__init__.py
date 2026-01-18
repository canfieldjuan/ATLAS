"""Email templates for Effingham Office Maids."""

from .estimate_confirmation import (
    BUSINESS_NAME,
    BUSINESS_PHONE,
    BUSINESS_EMAIL,
    BUSINESS_WEBSITE,
    format_business_email,
    format_residential_email,
)

from .proposal import (
    format_business_proposal,
    format_residential_proposal,
)

__all__ = [
    "BUSINESS_NAME",
    "BUSINESS_PHONE",
    "BUSINESS_EMAIL",
    "BUSINESS_WEBSITE",
    "format_business_email",
    "format_residential_email",
    "format_business_proposal",
    "format_residential_proposal",
]
