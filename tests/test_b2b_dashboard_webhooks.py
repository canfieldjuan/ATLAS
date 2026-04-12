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
webhook_dispatcher = importlib.import_module('atlas_brain.services.b2b.webhook_dispatcher')


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


@pytest.mark.asyncio
async def test_list_webhook_deliveries_exposes_payload_context_and_account_focus():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': '3df7f790-6afc-4e0f-b40e-a78f77e60dd2',
                'event_type': 'churn_alert',
                'payload': {
                    'vendor': 'Acme Rival',
                    'data': {
                        'company_name': 'Acme Bank',
                        'signal_type': 'competitive_displacement',
                        'company_signal_id': '22222222-2222-2222-2222-222222222222',
                        'review_id': 'review-123',
                    },
                },
                'status_code': 202,
                'duration_ms': 180,
                'attempt': 1,
                'success': True,
                'error': None,
                'delivered_at': datetime(2026, 4, 11, 1, 0, tzinfo=timezone.utc),
            }
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')
    focus = {
        'vendor': 'Acme Rival',
        'company': 'Acme Bank',
        'report_date': '2026-04-10',
        'watch_vendor': 'Acme Rival',
        'category': 'Switch Risk',
        'track_mode': 'competitor',
    }

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with patch.object(
            b2b_dashboard,
            '_resolve_webhook_activity_account_focus',
            AsyncMock(return_value=focus),
        ) as resolve_focus:
            result = await b2b_dashboard.list_webhook_deliveries('3df7f790-6afc-4e0f-b40e-a78f77e60dd2', success=None, event_type=None, start_date=None, end_date=None, limit=50, user=user)

    assert result['count'] == 1
    delivery = result['deliveries'][0]
    assert delivery['vendor_name'] == 'Acme Rival'
    assert delivery['company_name'] == 'Acme Bank'
    assert delivery['signal_type'] == 'competitive_displacement'
    assert delivery['account_review_focus'] == focus
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['signal_id'] == '22222222-2222-2222-2222-222222222222'
    assert kwargs['review_id'] == 'review-123'


@pytest.mark.asyncio
async def test_list_crm_push_log_exposes_account_review_focus():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'aa5659f4-dfe6-4c12-8fd6-e6eb5579cc44',
                'signal_type': 'company_signal',
                'signal_id': '22222222-2222-2222-2222-222222222222',
                'vendor_name': 'Acme Rival',
                'company_name': 'Acme Bank',
                'crm_record_id': 'crm-42',
                'crm_record_type': 'deal',
                'status': 'success',
                'error': None,
                'pushed_at': datetime(2026, 4, 11, 2, 0, tzinfo=timezone.utc),
            }
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')
    focus = {
        'vendor': 'Acme Rival',
        'company': 'Acme Bank',
        'report_date': '2026-04-10',
        'watch_vendor': 'Acme Rival',
        'category': 'Switch Risk',
        'track_mode': 'competitor',
    }

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with patch.object(
            b2b_dashboard,
            '_resolve_webhook_activity_account_focus',
            AsyncMock(return_value=focus),
        ) as resolve_focus:
            result = await b2b_dashboard.list_crm_push_log('aa5659f4-dfe6-4c12-8fd6-e6eb5579cc44', limit=50, user=user)

    assert result['count'] == 1
    push = result['pushes'][0]
    assert push['signal_id'] == '22222222-2222-2222-2222-222222222222'
    assert push['account_review_focus'] == focus
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['signal_id'] == '22222222-2222-2222-2222-222222222222'


@pytest.mark.asyncio
async def test_resolve_webhook_activity_account_focus_prefers_review_id_match():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                'id': '22222222-2222-2222-2222-222222222222',
                'company_name': 'Acme Bank',
                'vendor_name': 'Acme Rival',
                'review_id': 'review-123',
            },
            {
                'vendor_name': 'Acme Rival',
                'track_mode': 'competitor',
            },
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')
    report_row = {
        'report_date': '2026-04-10',
        'intelligence_data': {
            'accounts': [
                {
                    'company': 'Acme Bank',
                    'vendor': 'Acme Rival',
                    'category': 'Switch Risk',
                    'urgency': 8.1,
                    'source_reviews': ['review-123'],
                },
                {
                    'company': 'Acme Bank',
                    'vendor': 'Acme Rival',
                    'category': 'Noise',
                    'urgency': 7.5,
                    'source_reviews': ['review-other'],
                },
            ],
        },
    }

    with patch.object(
        b2b_dashboard,
        '_fetch_latest_accounts_in_motion_report',
        AsyncMock(return_value=report_row),
    ):
        focus = await b2b_dashboard._resolve_webhook_activity_account_focus(
            pool,
            user,
            vendor_name='Acme Rival',
            company_name='Acme Bank',
            signal_id='22222222-2222-2222-2222-222222222222',
        )

    assert focus == {
        'vendor': 'Acme Rival',
        'company': 'Acme Bank',
        'report_date': '2026-04-10',
        'watch_vendor': 'Acme Rival',
        'category': 'Switch Risk',
        'track_mode': 'competitor',
    }


@pytest.mark.asyncio
async def test_log_crm_push_persists_signal_id_from_envelope():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)

    await webhook_dispatcher._log_crm_push(
        pool,
        '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
        'signal_update',
        {
            'vendor': 'Acme Rival',
            'data': {
                'company_name': 'Acme Bank',
                'company_signal_id': '22222222-2222-2222-2222-222222222222',
            },
        },
        crm_record_id='crm-42',
    )

    sql = pool.execute.call_args[0][0]
    args = pool.execute.call_args[0][1:]
    assert 'signal_id' in sql
    assert args[1] == 'company_signal'
    assert args[2] == '22222222-2222-2222-2222-222222222222'
    assert args[3] == 'Acme Rival'
    assert args[4] == 'Acme Bank'
