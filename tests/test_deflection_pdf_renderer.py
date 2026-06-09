from __future__ import annotations

import pytest

from atlas_brain.deflection_pdf_renderer import (
    render_deflection_full_report_pdf,
)


def test_render_deflection_full_report_pdf_from_artifact_markdown() -> None:
    pdf_bytes = render_deflection_full_report_pdf(
        {
            "markdown": (
                "# Support Ticket Deflection Report\n\n"
                "## Support Tax Confirmation\n\n"
                "- Customers ask why duplicate-looking invoices appear.\n\n"
                "| Rank | Customer question | Tickets |\n"
                "|---:|---|---:|\n"
                "| 1 | Why did I get two invoices? | 4 |\n"
            ),
        }
    )

    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


def test_render_deflection_full_report_pdf_requires_markdown() -> None:
    with pytest.raises(ValueError, match="artifact markdown is required"):
        render_deflection_full_report_pdf({"summary": {"generated": 0}})
