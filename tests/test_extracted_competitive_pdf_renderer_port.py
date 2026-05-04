from __future__ import annotations

import importlib
import sys
from datetime import date
from typing import Any

import pytest

from extracted_competitive_intelligence._standalone.pdf_renderer import (
    PDFRenderer,
    PDFRendererNotConfigured,
    configure_pdf_renderer,
    get_pdf_renderer,
    render_report_pdf,
    render_vendor_full_report_pdf,
)


class PDFRendererAdapter:
    def __init__(self) -> None:
        self.report_calls: list[dict[str, Any]] = []
        self.vendor_calls: list[dict[str, Any]] = []

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
        self.report_calls.append({
            "report_type": report_type,
            "vendor_filter": vendor_filter,
            "category_filter": category_filter,
            "report_date": report_date,
            "executive_summary": executive_summary,
            "intelligence_data": intelligence_data,
            "data_density": data_density,
        })
        return b"%PDF-report-adapter"

    def render_vendor_full_report_pdf(
        self,
        vendor_name: str,
        report_data: dict[str, Any],
        briefing_data: dict[str, Any] | None = None,
    ) -> bytes:
        self.vendor_calls.append({
            "vendor_name": vendor_name,
            "report_data": report_data,
            "briefing_data": briefing_data,
        })
        return b"%PDF-vendor-adapter"


def teardown_function() -> None:
    configure_pdf_renderer(None)


def test_standalone_pdf_renderer_fails_closed_until_configured() -> None:
    configure_pdf_renderer(None)

    with pytest.raises(PDFRendererNotConfigured):
        get_pdf_renderer()


def test_standalone_pdf_renderer_returns_configured_adapter() -> None:
    adapter = PDFRendererAdapter()

    configure_pdf_renderer(adapter)

    assert isinstance(adapter, PDFRenderer)
    assert get_pdf_renderer() is adapter


def test_standalone_pdf_renderer_delegates_public_render_functions() -> None:
    adapter = PDFRendererAdapter()
    configure_pdf_renderer(adapter)

    report_pdf = render_report_pdf(
        report_type="vendor_scorecard",
        vendor_filter="Acme",
        report_date="2026-05-04",
        executive_summary="summary",
        intelligence_data={"score": 91},
        data_density={"reviews": 12},
    )
    vendor_pdf = render_vendor_full_report_pdf(
        "Acme",
        {"weekly_churn_feed": []},
        {"account_pressure": []},
    )

    assert report_pdf == b"%PDF-report-adapter"
    assert vendor_pdf == b"%PDF-vendor-adapter"
    assert adapter.report_calls == [{
        "report_type": "vendor_scorecard",
        "vendor_filter": "Acme",
        "category_filter": None,
        "report_date": "2026-05-04",
        "executive_summary": "summary",
        "intelligence_data": {"score": 91},
        "data_density": {"reviews": 12},
    }]
    assert adapter.vendor_calls == [{
        "vendor_name": "Acme",
        "report_data": {"weekly_churn_feed": []},
        "briefing_data": {"account_pressure": []},
    }]


def test_service_module_uses_standalone_port_without_atlas(monkeypatch) -> None:
    module_name = "extracted_competitive_intelligence.services.b2b.pdf_renderer"
    atlas_module_name = "atlas_brain.services.b2b.pdf_renderer"
    monkeypatch.setenv("EXTRACTED_COMP_INTEL_STANDALONE", "1")
    sys.modules.pop(module_name, None)
    sys.modules.pop(atlas_module_name, None)

    module = importlib.import_module(module_name)

    try:
        assert module.PDFRenderer.__module__ == (
            "extracted_competitive_intelligence._standalone.pdf_renderer"
        )
        with pytest.raises(module.PDFRendererNotConfigured):
            module.get_pdf_renderer()

        adapter = PDFRendererAdapter()
        module.configure_pdf_renderer(adapter)
        pdf_bytes = module.render_vendor_full_report_pdf(
            "Acme",
            {"weekly_churn_feed": []},
            {"account_pressure": []},
        )
        assert pdf_bytes == b"%PDF-vendor-adapter"
        assert atlas_module_name not in sys.modules
    finally:
        module.configure_pdf_renderer(None)
        sys.modules.pop(module_name, None)
