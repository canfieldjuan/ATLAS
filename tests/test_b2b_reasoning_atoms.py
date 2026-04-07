from types import SimpleNamespace

from atlas_brain.autonomous.tasks._b2b_reasoning_atoms import (
    build_reasoning_atoms,
    build_reasoning_delta,
    build_scope_manifest,
)


def _packet():
    return SimpleNamespace(
        aggregates=[SimpleNamespace(label="total_reviews", value=24, source_id="metric:total_reviews")],
        metric_ledger=[{"_sid": "metric:switch_volume"}],
        witness_pack=[
            {
                "_sid": "witness:1",
                "witness_id": "witness:1",
                "review_id": "review-1",
                "reviewed_at": "2026-04-01T00:00:00+00:00",
                "source": "g2",
            },
            {
                "_sid": "witness:2",
                "witness_id": "witness:2",
                "review_id": "review-2",
                "reviewed_at": "2026-03-15T00:00:00+00:00",
                "source": "trustpilot",
            },
        ],
        section_packets={
            "anchor_examples": {
                "named_accounts": ["witness:1"],
                "counterevidence": ["witness:2"],
            },
            "timing_packet": {"witness_ids": ["witness:1"]},
            "displacement_packet": {"witness_ids": ["witness:1", "witness:2"]},
            "retention_packet": {"witness_ids": ["witness:2"]},
            "_witness_governance": {
                "filtered_generic_candidates": 2,
                "thin_specific_witness_pool": False,
            },
        },
        contradiction_rows=[{"summary": "Some teams still cite support strength", "_sid": "witness:2"}],
        coverage_gaps=[{"gap": "thin_account_signals", "citations": ["witness:2"]}],
    )


def _contracts():
    return {
        "vendor_core_reasoning": {
            "causal_narrative": {
                "primary_wedge": "price_squeeze",
                "causal_chain": "Pricing pressure is clustering in admin-heavy teams.",
                "why_now": "Renewals are surfacing cost scrutiny.",
                "confidence": "high",
                "citations": ["witness:1", "metric:total_reviews"],
                "what_would_weaken_thesis": ["Support sentiment rebounds"],
            },
            "timing_intelligence": {
                "best_timing_window": "Q2 renewal cycle",
                "confidence": "medium",
                "citations": ["witness:1"],
                "immediate_triggers": [
                    {
                        "type": "renewal",
                        "trigger": "Contract renewal in 60 days",
                        "urgency": "high",
                        "action": "Reach procurement and admins now",
                        "citations": ["witness:1"],
                    },
                ],
            },
            "confidence_posture": {
                "limits": ["Thin retention sample"],
                "citations": ["witness:2"],
            },
        },
        "displacement_reasoning": {
            "migration_proof": {
                "confidence": "medium",
                "evaluation_vs_switching": "Evaluations are turning into real migrations.",
                "top_destination": {
                    "value": "HubSpot",
                    "source_id": "witness:1",
                },
                "primary_switch_driver": {
                    "value": "Admin overhead",
                    "source_id": "witness:1",
                },
                "switch_volume": {
                    "value": 12,
                    "source_id": "metric:switch_volume",
                },
                "named_examples": [
                    {
                        "company": "Acme",
                        "evidence": "We moved off because admin overhead kept growing.",
                        "quotable": True,
                        "citations": ["witness:1"],
                    },
                ],
            },
            "competitive_reframes": {
                "confidence": "medium",
                "reframes": [
                    {
                        "incumbent_claim": "Cheapest at scale",
                        "reframe": "Admin sprawl makes total cost worse.",
                        "why_buyers_believe_it": "Ops teams cite extra admin work.",
                        "best_segment": "Mid-market ops",
                        "proof_point": {
                            "field": "switch_volume",
                            "value": 12,
                            "source_id": "metric:switch_volume",
                            "interpretation": "Migration volume is rising",
                        },
                        "citations": ["witness:1", "metric:switch_volume"],
                    },
                ],
                "switch_triggers": [
                    {
                        "type": "migration",
                        "description": "Budget review plus admin complaints",
                        "urgency": "medium",
                        "recommended_action": "Push total-cost comparison",
                        "citations": ["witness:1"],
                    },
                ],
            },
        },
        "account_reasoning": {
            "top_accounts": [
                {
                    "company": "Acme",
                    "role_type": "ops",
                    "buying_stage": "renewal_decision",
                    "urgency": 8,
                    "competitor_context": "Comparing HubSpot",
                    "contract_end": "2026-06-30",
                    "decision_timeline": "within_quarter",
                    "primary_pain": "Admin overhead",
                    "quote": "Admin work keeps growing.",
                    "trust_tier": "high",
                    "citations": ["witness:1"],
                },
            ],
        },
        "evidence_governance": {
            "contradictions": [
                {
                    "summary": "A minority still praise support responsiveness.",
                    "citations": ["witness:2"],
                },
            ],
            "coverage_gaps": [
                {"gap": "thin_account_signals", "citations": ["witness:2"]},
            ],
        },
    }


def test_build_scope_manifest_tracks_witness_mix_and_drop_reasons():
    manifest = build_scope_manifest(_packet())

    assert manifest["selection_strategy"] == "vendor_facet_packet_v1"
    assert manifest["reviews_considered_total"] == 24
    assert manifest["reviews_in_scope"] == 2
    assert manifest["witnesses_in_scope"] == 2
    assert manifest["witness_mix"]["anchor_examples"] == 2
    assert "filtered_generic_candidates:2" in manifest["reasons_dropped"]
    assert "thin_account_signals" in manifest["reasons_dropped"]


def test_build_reasoning_atoms_exposes_lineage():
    atoms = build_reasoning_atoms(_contracts(), _packet())

    assert atoms["schema_version"] == "v1"
    assert atoms["theses"]
    assert atoms["timing_windows"]
    assert atoms["proof_points"]
    assert atoms["account_signals"]
    assert atoms["counterevidence"]
    assert atoms["coverage_limits"]
    first_thesis = atoms["theses"][0]
    assert first_thesis["metric_ids"] == ["metric:total_reviews"]
    assert first_thesis["witness_ids"] == ["witness:1"]
    assert first_thesis["source_ids"] == ["metric:total_reviews", "witness:1"]
    assert first_thesis["evidence_count"] == 2
    assert first_thesis["last_supported_at"].startswith("2026-04-01")


def test_build_reasoning_delta_detects_core_changes():
    current = {
        "reasoning_contracts": _contracts(),
        "reasoning_atoms": build_reasoning_atoms(_contracts(), _packet()),
    }
    previous_contracts = _contracts()
    previous_contracts["vendor_core_reasoning"]["causal_narrative"]["primary_wedge"] = "support_collapse"
    previous_contracts["displacement_reasoning"]["migration_proof"]["top_destination"] = {
        "value": "Pipedrive",
        "source_id": "witness:2",
    }
    previous_atoms = build_reasoning_atoms(previous_contracts, _packet())
    previous_atoms["account_signals"] = []
    previous = {
        "reasoning_contracts": previous_contracts,
        "reasoning_atoms": previous_atoms,
    }

    delta = build_reasoning_delta(
        current,
        previous,
        current_as_of_date="2026-04-06",
        previous_as_of_date="2026-04-05",
    )

    assert delta["changed"] is True
    assert delta["wedge_changed"] is True
    assert delta["top_destination_changed"] is True
    assert delta["new_account_signals"] == ["Acme"]
    assert delta["previous_as_of_date"] == "2026-04-05"
