"""Focused webhook list contract tests for the B2B dashboard API."""

import importlib
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type('UndefinedTableError', (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault('asyncpg', _asyncpg_mock)
sys.modules.setdefault('asyncpg.exceptions', _asyncpg_exceptions)

for _mod in (
    'torch',
    'torchaudio',
    'transformers',
    'accelerate',
    'bitsandbytes',
    'PIL',
    'PIL.Image',
    'numpy',
    'cv2',
    'sounddevice',
    'soundfile',
    'playwright',
    'playwright.async_api',
    'playwright_stealth',
    'curl_cffi',
    'curl_cffi.requests',
    'pytrends',
    'pytrends.request',
):
    sys.modules.setdefault(_mod, MagicMock())

b2b_dashboard = importlib.import_module('atlas_brain.api.b2b_dashboard')


@pytest.mark.asyncio
async def test_list_webhooks_exposes_latest_test_summary():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    failed_at = datetime.now(timezone.utc) - timedelta(hours=2)
    tested_at = datetime.now(timezone.utc) - timedelta(hours=3)
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                'url': 'https://hooks.example.com/churn',
                'event_types': ['churn_alert', 'signal_update'],
                'channel': 'generic',
                'enabled': True,
                'description': 'PagerDuty bridge',
                'created_at': created_at,
                'updated_at': created_at,
                'recent_deliveries': 12,
                'recent_successes': 11,
                'latest_failure_event_type': 'signal_update',
                'latest_failure_status_code': 500,
                'latest_failure_error': 'downstream timeout',
                'latest_failure_at': failed_at,
                'latest_test_success': False,
                'latest_test_status_code': 504,
                'latest_test_error': 'test timeout',
                'latest_test_at': tested_at,
            }
        ]
    )
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.list_webhooks(user=user)

    assert result['count'] == 1
    webhook = result['webhooks'][0]
    assert webhook['recent_success_rate_7d'] == 0.917
    assert webhook['latest_failure_event_type'] == 'signal_update'
    assert webhook['latest_test_success'] is False
    assert webhook['latest_test_status_code'] == 504
    assert webhook['latest_test_error'] == 'test timeout'
    assert webhook['latest_test_at'] == tested_at.isoformat()


@pytest.mark.asyncio
async def test_list_webhooks_tolerates_missing_latest_test_summary_fields():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    failed_at = datetime.now(timezone.utc) - timedelta(hours=2)
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'f6ce3a70-c7d1-4c84-a418-a9c3555d6a14',
                'url': 'https://hooks.example.com/churn',
                'event_types': ['churn_alert'],
                'channel': 'generic',
                'enabled': True,
                'description': 'Generic webhook',
                'created_at': created_at,
                'updated_at': created_at,
                'recent_deliveries': 3,
                'recent_successes': 2,
                'latest_failure_event_type': 'churn_alert',
                'latest_failure_status_code': 500,
                'latest_failure_error': 'timeout',
                'latest_failure_at': failed_at,
            }
        ]
    )
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.list_webhooks(user=user)

    assert result['count'] == 1
    webhook = result['webhooks'][0]
    assert webhook['recent_success_rate_7d'] == 0.667
    assert webhook['latest_failure_at'] == failed_at.isoformat()
    assert webhook['latest_test_success'] is None
    assert webhook['latest_test_status_code'] is None
    assert webhook['latest_test_error'] is None
    assert webhook['latest_test_at'] is None
