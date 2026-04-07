import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
    "pytrends", "pytrends.request",
):
    sys.modules.setdefault(_mod, MagicMock())

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cleanup_accounts_in_motion_pollution import (
    _fetch_company_signal_rows,
    _polluted_company_signal_reason,
)


def test_polluted_company_signal_reason_rejects_descriptor_company_names():
    row = {
        "company_name": "midsized ERP software",
        "vendor_name": "Salesforce",
        "source": "reddit",
        "content_type": "insider_account",
        "confidence_score": 0.21,
    }
    assert _polluted_company_signal_reason(row) == "ineligible_company_name"


def test_polluted_company_signal_reason_rejects_low_confidence_reddit_insiders():
    row = {
        "company_name": "Acme Corp",
        "vendor_name": "Salesforce",
        "source": "reddit",
        "content_type": "insider_account",
        "confidence_score": 0.21,
    }
    assert _polluted_company_signal_reason(row) == "low_confidence_low_trust_source"


def test_polluted_company_signal_reason_rejects_deprecated_sources():
    row = {
        "company_name": "Acme Corp",
        "vendor_name": "Salesforce",
        "source": "trustpilot",
        "content_type": "review",
        "confidence_score": 0.54,
    }
    assert _polluted_company_signal_reason(row) == "deprecated_source"


def test_polluted_company_signal_reason_keeps_valid_named_accounts():
    row = {
        "company_name": "Acme Pharmaceutical",
        "vendor_name": "ClickUp",
        "source": "g2",
        "content_type": "review",
        "confidence_score": 0.62,
        "actively_evaluating": True,
    }
    assert _polluted_company_signal_reason(row) is None


def test_polluted_company_signal_reason_rejects_missing_signal_evidence():
    row = {
        "company_name": "Infohob",
        "vendor_name": "Trello",
        "source": "peerspot",
        "content_type": "review",
        "confidence_score": 0.54,
        "intent_to_leave": False,
        "actively_evaluating": False,
        "contract_renewal_mentioned": False,
        "indicator_cancel": False,
        "indicator_migration": False,
        "indicator_evaluation": False,
        "indicator_switch": False,
    }
    assert _polluted_company_signal_reason(row) == "missing_signal_evidence"


@pytest.mark.asyncio
async def test_fetch_company_signal_rows_lowercases_vendor_filter():
    pool = MagicMock()
    pool.fetch = AsyncMock(return_value=[])

    await _fetch_company_signal_rows(
        pool,
        vendors=["Trello"],
        window_days=30,
    )

    assert pool.fetch.await_args.args[1] == 30
    assert pool.fetch.await_args.args[2] == ["trello"]
