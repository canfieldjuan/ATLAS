import sys
from pathlib import Path
from unittest.mock import MagicMock


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

from cleanup_accounts_in_motion_pollution import _polluted_company_signal_reason


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
    }
    assert _polluted_company_signal_reason(row) is None
