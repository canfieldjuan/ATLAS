import sys
from pathlib import Path
from unittest.mock import MagicMock


sys.modules.setdefault("asyncpg", MagicMock())

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from backfill_derived_fields import _matches_filter, _price_state


def test_price_state_reads_current_pricing_artifacts():
    enrichment = {
        "contract_context": {"price_complaint": True},
        "urgency_indicators": {"price_pressure_language": True},
        "evidence_spans": [
            {"signal_type": "pricing_backlash"},
            {"signal_type": "positive_anchor"},
        ],
    }

    assert _price_state(enrichment) == {
        "price_complaint": True,
        "price_pressure_language": True,
        "has_pricing_backlash": True,
    }


def test_matches_filter_accepts_positive_pricing_false_positive_fix():
    old_values = {
        "price_complaint": True,
        "price_pressure_language": True,
        "has_pricing_backlash": True,
    }
    new_enrichment = {
        "contract_context": {"price_complaint": False},
        "urgency_indicators": {"price_pressure_language": False},
        "evidence_spans": [{"signal_type": "positive_anchor"}],
    }

    assert _matches_filter("positive_pricing_false_positive", old_values, new_enrichment) is True


def test_matches_filter_rejects_rows_still_showing_real_price_pressure():
    old_values = {
        "price_complaint": True,
        "price_pressure_language": True,
        "has_pricing_backlash": True,
    }
    new_enrichment = {
        "contract_context": {"price_complaint": True},
        "urgency_indicators": {"price_pressure_language": True},
        "evidence_spans": [{"signal_type": "pricing_backlash"}],
    }

    assert _matches_filter("positive_pricing_false_positive", old_values, new_enrichment) is False
