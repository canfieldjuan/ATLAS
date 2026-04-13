import atlas_brain.services.b2b.pdf_renderer as pdf_renderer_module
from atlas_brain.services.b2b.pdf_renderer import (
    _battle_card_priority_segments,
    _battle_card_render_view,
    _render_battle_card,
    _render_churn_feed,
    _render_vendor_scorecard,
    _section_coverage_rows,
    render_report_pdf,
    render_vendor_full_report_pdf,
)


class _CapturePdf:
    def __init__(self) -> None:
        self.section_titles: list[str] = []
        self.body_lines: list[str] = []
        self.metric_rows: list[tuple[str, str]] = []
        self.tables: list[tuple[list[str], list[list[str]]]] = []

    def alias_nb_pages(self) -> None:
        return None

    def add_page(self) -> None:
        return None

    def set_font(self, *args, **kwargs) -> None:
        return None

    def set_text_color(self, *args, **kwargs) -> None:
        return None

    def cell(self, *args, **kwargs) -> None:
        return None

    def ln(self, *args, **kwargs) -> None:
        return None

    def multi_cell(self, *args, **kwargs) -> None:
        if len(args) >= 3:
            self.body_lines.append(str(args[2]))

    def section_title(self, title: str) -> None:
        self.section_titles.append(title)

    def body_text(self, text: str) -> None:
        self.body_lines.append(text)

    def metric_row(self, label: str, value: str, color=None) -> None:
        self.metric_rows.append((label, value))

    def quote_block(self, text: str) -> None:
        self.body_lines.append(text)

    def simple_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        col_widths: list[float] | None = None,
    ) -> None:
        self.tables.append((headers, rows))

    def output(self, *args, **kwargs) -> bytes:
        if args:
            target = args[0]
            write = getattr(target, "write", None)
            if callable(write):
                write(b"%PDF-stub")
        return b"%PDF-stub"


class TestBattleCardRenderView:
    def test_prefers_contract_sections_over_stale_flat_fields(self):
        card = {
            "causal_narrative": {
                "primary_wedge": "support_erosion",
                "trigger": "Old flat trigger",
            },
            "timing_intelligence": {
                "active_eval_signals": 0,
            },
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "trigger": "Contract price hike",
                    },
                    "timing_intelligence": {
                        "active_eval_signals": {
                            "value": 6,
                            "source_id": "accounts:summary:active_eval_signal_count",
                        },
                    },
                },
            },
            "synthesis_wedge": "price_squeeze",
            "evidence_window_days": 17,
            "evidence_window_is_thin": True,
            "reasoning_source": "b2b_reasoning_synthesis",
        }

        view = _battle_card_render_view(card)

        assert view["causal_narrative"]["primary_wedge"] == "price_squeeze"
        assert view["causal_narrative"]["trigger"] == "Contract price hike"
        assert view["timing_intelligence"]["active_eval_signals"]["value"] == 6
        assert view["synthesis_wedge_label"] == "Price Squeeze"
        assert view["evidence_window_days"] == 17
        assert view["evidence_window_is_thin"] is True
        assert view["reasoning_source"] == "b2b_reasoning_synthesis"
        assert "vendor_core_reasoning" not in view
        assert "displacement_reasoning" not in view

    def test_does_not_backfill_missing_contract_section_from_flat_field(self):
        card = {
            "timing_intelligence": {
                "best_timing_window": "Legacy timing mirror",
            },
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "causal_narrative": {
                        "primary_wedge": "price_squeeze",
                        "trigger": "Contract price hike",
                    },
                },
            },
        }

        view = _battle_card_render_view(card)

        assert view["causal_narrative"]["trigger"] == "Contract price hike"
        assert "timing_intelligence" not in view

    def test_preserves_deterministic_battle_card_sections(self):
        card = {
            "vendor_weaknesses": [{"area": "pricing", "count": 8}],
            "competitor_differentiators": [{"competitor": "Freshdesk", "mentions": 10}],
            "weakness_analysis": [
                {
                    "weakness": "Pricing complexity",
                    "evidence": "Pricing complaints accelerated in renewal reviews.",
                    "customer_quote": "Our bill kept growing after add-ons.",
                    "winning_position": "Lead with simpler packaging.",
                },
            ],
            "competitive_landscape": {
                "vulnerability_window": "Renewal scrutiny is elevated after recent pricing changes.",
                "top_alternatives": ["Freshdesk (12 mentions in evaluation sets)"],
                "displacement_triggers": ["Renewals after recent packaging changes"],
            },
        }

        view = _battle_card_render_view(card)

        assert view["weakness_analysis"][0]["weakness"] == "Pricing complexity"
        assert view["competitive_landscape"]["top_alternatives"] == [
            "Freshdesk (12 mentions in evaluation sets)",
        ]

    def test_falls_back_to_nested_evidence_window_metadata(self):
        card = {
            "evidence_window": {
                "days": 15,
                "label": "Recent 15-day review window",
            },
        }

        view = _battle_card_render_view(card)

        assert view["evidence_window_days"] == 15
        assert view["evidence_window_label"] == "Recent 15-day review window"

    def test_priority_segments_include_sample_size_for_render(self):
        card = {
            "reasoning_contracts": {
                "vendor_core_reasoning": {
                    "segment_playbook": {
                        "priority_segments": [
                            {
                                "segment": "Mid-Market operations teams",
                                "estimated_reach": {
                                    "value": 22,
                                    "source_id": "segment:reach:size:mid_market",
                                },
                                "sample_size": 22,
                                "best_opening_angle": "TCO comparison",
                                "why_vulnerable": "Budget pressure",
                            },
                        ],
                    },
                },
            },
        }

        view = _battle_card_render_view(card)
        segments = _battle_card_priority_segments(view)

        assert segments == [
            {
                "segment": "Mid-Market operations teams",
                "reach": 22,
                "sample_size": 22,
                "best_opening_angle": "TCO comparison",
                "why_vulnerable": "Budget pressure",
                "disqualifier": "",
            },
        ]

    def test_render_battle_card_includes_account_pressure_block(self):
        pdf = _CapturePdf()

        _render_battle_card(
            pdf,
            {
                "vendor": "Salesforce",
                "churn_pressure_score": 71,
                "confidence": "medium",
                "total_reviews": 18,
                "account_pressure_summary": "Detected concentrated enterprise churn pressure.",
                "priority_account_names": ["Concentrix", "Slack"],
                "account_reasoning_preview": {
                    "disclaimer": "Early account signal only.",
                },
            },
            exec_summary=None,
        )

        assert "Account Pressure" in pdf.section_titles
        assert "Detected concentrated enterprise churn pressure." in pdf.body_lines
        assert "Priority accounts: Concentrix, Slack" in pdf.body_lines
        assert "Note: Early account signal only." in pdf.body_lines


class TestPreviewAccountPressureRendering:
    def test_render_churn_feed_includes_preview_account_pressure_lines(self):
        pdf = _CapturePdf()

        _render_churn_feed(
            pdf,
            [
                {
                    "vendor": "HubSpot",
                    "churn_pressure_score": 62,
                    "risk_level": "high",
                    "account_pressure_summary": "Expansion churn pressure is concentrated in large teams.",
                    "priority_account_names": ["Acme"],
                    "account_pressure_disclaimer": "Preview only.",
                },
            ],
        )

        assert "Account pressure: Expansion churn pressure is concentrated in large teams." in pdf.body_lines
        assert "Priority accounts: Acme" in pdf.body_lines
        assert "Note: Preview only." in pdf.body_lines

    def test_render_vendor_scorecard_includes_account_pressure_section(self):
        pdf = _CapturePdf()

        _render_vendor_scorecard(
            pdf,
            {
                "churn_pressure_score": 64,
                "account_pressure_summary": "Named account pressure is emerging in enterprise support teams.",
                "priority_account_names": ["Concentrix"],
                "account_reasoning_preview": {
                    "disclaimer": "Early account signal only.",
                },
            },
            exec_summary=None,
        )

        assert "Account Pressure" in pdf.section_titles
        assert "Named account pressure is emerging in enterprise support teams." in pdf.body_lines
        assert "Priority accounts: Concentrix" in pdf.body_lines
        assert "Note: Early account signal only." in pdf.body_lines

    def test_render_vendor_full_report_pdf_includes_account_pressure_section(
        self,
        monkeypatch,
    ):
        pdf = _CapturePdf()
        monkeypatch.setattr(pdf_renderer_module, "IntelligenceReportPDF", lambda: pdf)

        pdf_bytes = render_vendor_full_report_pdf(
            "Salesforce",
            {
                "weekly_churn_feed": [
                    {
                        "vendor": "Salesforce",
                        "churn_pressure_score": 71,
                    },
                ],
                "vendor_scorecards": [],
                "displacement_map": [],
            },
            briefing_data={
                "account_pressure_summary": "Sparse but real account pressure is visible.",
                "priority_account_names": ["Concentrix"],
                "account_reasoning_preview": {
                    "disclaimer": "Preview only until canonical account reasoning expands.",
                },
            },
        )

        assert pdf_bytes == b"%PDF-stub"
        assert "Account Pressure" in pdf.section_titles
        assert "Sparse but real account pressure is visible." in pdf.body_lines
        assert "Priority accounts: Concentrix" in pdf.body_lines
        assert "Note: Preview only until canonical account reasoning expands." in pdf.body_lines


class TestReportPdfEvidenceCoverage:
    def test_section_coverage_rows_summarize_witness_partial_and_thin_sections(self):
        rows = _section_coverage_rows({
            "key_insights": ["Strong expansion motion"],
            "key_insights_reference_ids": {"witness_ids": ["wit-1"]},
            "objection_handlers": ["Missing security proof points"],
            "objection_handlers_reference_ids": {"metric_ids": ["metric-1"]},
            "recommended_plays": ["Refresh proof points"],
        })

        assert rows == [
            ("Witness-backed Sections", "1"),
            ("Partial Evidence", "Objection Handlers"),
            ("Thin Evidence", "Recommended Plays"),
        ]

    def test_render_report_pdf_includes_section_coverage_in_export_path(self):
        pdf_bytes = render_report_pdf(
            report_type="exploratory_overview",
            vendor_filter="Zendesk",
            report_date="2026-04-10",
            executive_summary="Summary",
            intelligence_data={
                "key_insights": ["Strong expansion motion"],
                "key_insights_reference_ids": {"witness_ids": ["wit-1"]},
                "recommended_plays": ["Refresh proof points"],
            },
            data_density={"reviews": 12},
        )

        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes[:5] == b"%PDF-"
        assert len(pdf_bytes) > 500
