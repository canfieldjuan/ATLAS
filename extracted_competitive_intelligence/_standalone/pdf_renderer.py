"""PDF renderer port for standalone competitive intelligence."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, runtime_checkable


class PDFRendererNotConfigured(RuntimeError):
    pass


@runtime_checkable
class PDFRenderer(Protocol):
    def render_report_pdf(
        self,
        *,
        report_type: str,
        vendor_filter: str | None = None,
        category_filter: str | None = None,
        report_date: date | str | None = None,
        executive_summary: str | None = None,
        intelligence_data: dict[str, Any] | list[Any] | None = None,
        data_density: dict[str, Any] | None = None,
    ) -> bytes:
        ...

    def render_vendor_full_report_pdf(
        self,
        vendor_name: str,
        report_data: dict[str, Any],
        briefing_data: dict[str, Any] | None = None,
    ) -> bytes:
        ...


_renderer: PDFRenderer | None = None


def configure_pdf_renderer(renderer: PDFRenderer | None) -> None:
    global _renderer
    _renderer = renderer


def get_pdf_renderer() -> PDFRenderer:
    if _renderer is None:
        raise PDFRendererNotConfigured(
            "Standalone PDF renderer adapter is not configured"
        )
    return _renderer


def render_report_pdf(
    *,
    report_type: str,
    vendor_filter: str | None = None,
    category_filter: str | None = None,
    report_date: date | str | None = None,
    executive_summary: str | None = None,
    intelligence_data: dict[str, Any] | list[Any] | None = None,
    data_density: dict[str, Any] | None = None,
) -> bytes:
    return get_pdf_renderer().render_report_pdf(
        report_type=report_type,
        vendor_filter=vendor_filter,
        category_filter=category_filter,
        report_date=report_date,
        executive_summary=executive_summary,
        intelligence_data=intelligence_data,
        data_density=data_density,
    )


def render_vendor_full_report_pdf(
    vendor_name: str,
    report_data: dict[str, Any],
    briefing_data: dict[str, Any] | None = None,
) -> bytes:
    return get_pdf_renderer().render_vendor_full_report_pdf(
        vendor_name,
        report_data,
        briefing_data,
    )
