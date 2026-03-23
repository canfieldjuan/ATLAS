from atlas_brain.services.b2b.pdf_renderer import (
    _battle_card_priority_segments,
    _battle_card_render_view,
)


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
