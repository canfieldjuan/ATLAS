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
                'latest_failure_signal_id': '22222222-2222-2222-2222-222222222222',
                'latest_failure_review_id': '33333333-3333-4333-8333-333333333334',
                'latest_failure_report_id': None,
                'latest_test_success': False,
                'latest_test_status_code': 504,
                'latest_test_error': 'test timeout',
                'latest_test_at': tested_at,
                'latest_test_signal_id': None,
                'latest_test_review_id': None,
                'latest_test_report_id': '44444444-4444-4444-8444-444444444444',
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
    assert webhook['latest_failure_signal_id'] == '22222222-2222-2222-2222-222222222222'
    assert webhook['latest_failure_review_id'] == '33333333-3333-4333-8333-333333333334'
    assert webhook['latest_failure_report_id'] is None
    assert webhook['latest_test_signal_id'] is None
    assert webhook['latest_test_review_id'] is None
    assert webhook['latest_test_report_id'] == '44444444-4444-4444-8444-444444444444'


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
async def test_create_webhook_rejects_high_intent_push_for_generic_channel():
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.CreateWebhookBody(
        url='https://hooks.example.com/churn',
        secret='a' * 24,
        event_types=['high_intent_push'],
        channel='generic',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=MagicMock()):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.create_webhook(body, user=user)

    assert exc_info.value.status_code == 400
    assert 'require a CRM channel' in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_webhook_allows_high_intent_push_for_crm_channel():
    created_at = datetime.now(timezone.utc)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            'account_id': 'account-1',
            'url': 'https://hooks.example.com/crm',
            'event_types': ['high_intent_push', 'report_generated'],
            'channel': 'crm_hubspot',
            'enabled': True,
            'description': 'CRM escalation',
            'created_at': created_at,
        }
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.CreateWebhookBody(
        url='https://hooks.example.com/crm',
        secret='a' * 24,
        event_types=['high_intent_push', 'report_generated'],
        channel='crm_hubspot',
        auth_header='Bearer pat-123',
        description='CRM escalation',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.create_webhook(body, user=user)

    assert result['event_types'] == ['high_intent_push', 'report_generated']
    assert result['channel'] == 'crm_hubspot'


@pytest.mark.asyncio
async def test_create_webhook_trims_body_text_before_validation_and_persistence():
    created_at = datetime.now(timezone.utc)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            'account_id': 'account-1',
            'url': 'https://hooks.example.com/crm',
            'event_types': ['high_intent_push', 'report_generated'],
            'channel': 'crm_hubspot',
            'enabled': True,
            'description': 'CRM escalation',
            'created_at': created_at,
        }
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.CreateWebhookBody(
        url='  https://hooks.example.com/crm  ',
        secret='  ' + ('a' * 24) + '  ',
        event_types=['  high_intent_push  ', '   ', ' report_generated '],
        channel='  crm_hubspot  ',
        auth_header='  Bearer pat-123  ',
        description='  CRM escalation  ',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.create_webhook(body, user=user)

    assert result['event_types'] == ['high_intent_push', 'report_generated']
    assert result['channel'] == 'crm_hubspot'
    pool.fetchrow.assert_awaited_once()
    args = pool.fetchrow.await_args.args
    assert args[1] == 'account-1'
    assert args[2] == 'https://hooks.example.com/crm'
    assert args[3] == 'a' * 24
    assert args[4] == ['high_intent_push', 'report_generated']
    assert args[5] == 'crm_hubspot'
    assert args[6] == 'Bearer pat-123'
    assert args[7] == 'CRM escalation'


@pytest.mark.asyncio
async def test_update_webhook_blank_url_is_treated_as_no_update():
    pool = MagicMock()
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(url='   ')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.update_webhook(
                '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                body,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'No fields to update'
    pool.fetchval.assert_not_called()
    pool.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_create_webhook_validates_before_db_touch():
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.CreateWebhookBody(
        url='https://hooks.example.com/churn',
        secret='a' * 24,
        event_types=['high_intent_push'],
        channel='generic',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', side_effect=AssertionError('DB pool should not be acquired')):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.create_webhook(body, user=user)

    assert exc_info.value.status_code == 400
    assert 'require a CRM channel' in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_webhook_rejects_invalid_uuid_before_db_touch():
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', side_effect=AssertionError('DB pool should not be acquired')):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.get_webhook('   ', user=user)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'webhook_id must be a valid UUID'


@pytest.mark.asyncio
async def test_update_webhook_rejects_invalid_uuid_before_db_touch():
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(enabled=True)

    with patch.object(b2b_dashboard, '_pool_or_503', side_effect=AssertionError('DB pool should not be acquired')):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.update_webhook('   ', body, user=user)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'webhook_id must be a valid UUID'


@pytest.mark.asyncio
async def test_update_webhook_rejects_high_intent_push_for_generic_channel():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value='generic')
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(event_types=['high_intent_push'])

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.update_webhook(
                '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                body,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert 'require a CRM channel' in exc_info.value.detail
    pool.fetchval.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_webhook_deliveries_rejects_invalid_uuid_before_db_touch():
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', side_effect=AssertionError('DB pool should not be acquired')):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.list_webhook_deliveries(
                '   ',
                success=None,
                event_type=None,
                start_date=None,
                end_date=None,
                limit=50,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'webhook_id must be a valid UUID'


@pytest.mark.asyncio
async def test_test_webhook_rejects_invalid_uuid_before_db_touch():
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', side_effect=AssertionError('DB pool should not be acquired')):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.test_webhook('   ', user=user)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'webhook_id must be a valid UUID'


@pytest.mark.asyncio
async def test_list_crm_push_log_rejects_invalid_uuid_before_db_touch():
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', side_effect=AssertionError('DB pool should not be acquired')):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.list_crm_push_log('   ', limit=50, user=user)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == 'webhook_id must be a valid UUID'


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
                        'review_id': '33333333-3333-4333-8333-333333333334',
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
            '_fetch_company_signal_focus_context',
            AsyncMock(return_value={
                'signal_id': '22222222-2222-2222-2222-222222222222',
                'company_name': 'Acme Bank',
                'vendor_name': 'Acme Rival',
                'review_id': '33333333-3333-4333-8333-333333333334',
            }),
        ):
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
    assert delivery['signal_id'] == '22222222-2222-2222-2222-222222222222'
    assert delivery['signal_type'] == 'competitive_displacement'
    assert delivery['review_id'] == '33333333-3333-4333-8333-333333333334'
    assert delivery['account_review_focus'] == focus
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['signal_id'] == '22222222-2222-2222-2222-222222222222'
    assert kwargs['review_id'] == '33333333-3333-4333-8333-333333333334'


@pytest.mark.asyncio
async def test_list_webhook_deliveries_prefers_persisted_activity_refs():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': '71c27653-5df3-4d19-9b24-a1a15f61efbd',
                'event_type': 'report_generated',
                'payload': {
                    'vendor': 'Payload Vendor',
                    'data': {
                        'report_id': '11111111-1111-4111-8111-111111111111',
                    },
                },
                'signal_id': None,
                'review_id': '33333333-3333-4333-8333-333333333334',
                'report_id': '44444444-4444-4444-8444-444444444444',
                'vendor_name': 'Acme Rival',
                'company_name': 'Acme Bank',
                'status_code': 202,
                'duration_ms': 95,
                'attempt': 1,
                'success': True,
                'error': None,
                'delivered_at': datetime(2026, 4, 11, 4, 5, tzinfo=timezone.utc),
            }
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with patch.object(
            b2b_dashboard,
            '_fetch_webhook_activity_report_context',
            AsyncMock(return_value={
                'report_id': '44444444-4444-4444-8444-444444444444',
                'report_type': 'battle_card',
                'vendor_name': 'Acme Rival',
                'report_title': 'Acme Rival Battle Card',
            }),
        ) as fetch_report_context:
            with patch.object(
                b2b_dashboard,
                '_resolve_webhook_activity_account_focus',
                AsyncMock(return_value=None),
            ) as resolve_focus:
                result = await b2b_dashboard.list_webhook_deliveries(
                    '71c27653-5df3-4d19-9b24-a1a15f61efbd',
                    success=None,
                    event_type=None,
                    start_date=None,
                    end_date=None,
                    limit=50,
                    user=user,
                )

    delivery = result['deliveries'][0]
    assert delivery['vendor_name'] == 'Acme Rival'
    assert delivery['company_name'] == 'Acme Bank'
    assert delivery['review_id'] == '33333333-3333-4333-8333-333333333334'
    assert delivery['report_id'] == '44444444-4444-4444-8444-444444444444'
    assert delivery['report_type'] == 'battle_card'
    assert delivery['report_title'] == 'Acme Rival Battle Card'
    fetch_report_context.assert_awaited_once()
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['vendor_name'] == 'Acme Rival'
    assert kwargs['company_name'] == 'Acme Bank'
    assert kwargs['review_id'] == '33333333-3333-4333-8333-333333333334'


@pytest.mark.asyncio
async def test_list_crm_push_log_exposes_review_id_from_signal_context():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'cc5659f4-dfe6-4c12-8fd6-e6eb5579cc46',
                'signal_type': 'company_signal',
                'signal_id': '22222222-2222-2222-2222-222222222222',
                'vendor_name': 'Acme Rival',
                'company_name': 'Acme Bank',
                'crm_record_id': 'crm-43',
                'crm_record_type': 'deal',
                'status': 'success',
                'error': None,
                'pushed_at': datetime(2026, 4, 11, 2, 30, tzinfo=timezone.utc),
            }
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with patch.object(
            b2b_dashboard,
            '_fetch_company_signal_focus_context',
            AsyncMock(return_value={
                'signal_id': '22222222-2222-2222-2222-222222222222',
                'company_name': 'Acme Bank',
                'vendor_name': 'Acme Rival',
                'review_id': '33333333-3333-4333-8333-333333333334',
            }),
        ) as fetch_signal_context:
            with patch.object(
                b2b_dashboard,
                '_resolve_webhook_activity_account_focus',
                AsyncMock(return_value=None),
            ) as resolve_focus:
                result = await b2b_dashboard.list_crm_push_log(
                    'aa5659f4-dfe6-4c12-8fd6-e6eb5579cc44',
                    limit=50,
                    user=user,
                )

    push = result['pushes'][0]
    assert push['review_id'] == '33333333-3333-4333-8333-333333333334'
    fetch_signal_context.assert_awaited_once()
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['review_id'] == '33333333-3333-4333-8333-333333333334'


@pytest.mark.asyncio
async def test_list_crm_push_log_exposes_direct_review_id_without_signal_context():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'dd5659f4-dfe6-4c12-8fd6-e6eb5579cc47',
                'signal_type': 'high_intent_push',
                'signal_id': None,
                'review_id': '33333333-3333-4333-8333-333333333334',
                'vendor_name': 'Acme Rival',
                'company_name': 'Acme Bank',
                'crm_record_id': 'crm-44',
                'crm_record_type': 'deal',
                'status': 'success',
                'error': None,
                'pushed_at': datetime(2026, 4, 11, 2, 45, tzinfo=timezone.utc),
            }
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with patch.object(
            b2b_dashboard,
            '_resolve_webhook_activity_account_focus',
            AsyncMock(return_value=None),
        ) as resolve_focus:
            result = await b2b_dashboard.list_crm_push_log(
                'aa5659f4-dfe6-4c12-8fd6-e6eb5579cc44',
                limit=50,
                user=user,
            )

    push = result['pushes'][0]
    assert push['signal_id'] is None
    assert push['review_id'] == '33333333-3333-4333-8333-333333333334'
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['signal_id'] is None
    assert kwargs['review_id'] == '33333333-3333-4333-8333-333333333334'


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
            '_fetch_company_signal_focus_context',
            AsyncMock(return_value={
                'signal_id': '22222222-2222-2222-2222-222222222222',
                'company_name': 'Acme Bank',
                'vendor_name': 'Acme Rival',
                'review_id': '33333333-3333-4333-8333-333333333334',
            }),
        ):
            with patch.object(
                b2b_dashboard,
                '_resolve_webhook_activity_account_focus',
                AsyncMock(return_value=focus),
            ) as resolve_focus:
                result = await b2b_dashboard.list_crm_push_log('aa5659f4-dfe6-4c12-8fd6-e6eb5579cc44', limit=50, user=user)

    assert result['count'] == 1
    push = result['pushes'][0]
    assert push['signal_id'] == '22222222-2222-2222-2222-222222222222'
    assert push['review_id'] == '33333333-3333-4333-8333-333333333334'
    assert push['account_review_focus'] == focus
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['signal_id'] == '22222222-2222-2222-2222-222222222222'
    assert kwargs['review_id'] == '33333333-3333-4333-8333-333333333334'


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


def test_build_report_generated_payload_skips_failed_status():
    payload = webhook_dispatcher._build_report_generated_payload(
        report_id='report-1',
        report_type='battle_card',
        vendor_name='Acme Rival',
        status='failed',
    )

    assert payload is None


def test_build_report_generated_payload_formats_vendor_comparison_title():
    payload = webhook_dispatcher._build_report_generated_payload(
        report_id='report-1',
        report_type='vendor_comparison',
        vendor_name='Zendesk',
        category_filter='Freshdesk',
        status='published',
    )

    assert payload == {
        'artifact_type': 'vendor_comparison',
        'report_id': 'report-1',
        'report_type': 'vendor_comparison',
        'report_title': 'Zendesk vs Freshdesk',
        'status': 'published',
        'category_filter': 'Freshdesk',
    }


@pytest.mark.asyncio
async def test_log_delivery_attempt_persists_activity_refs():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)

    await webhook_dispatcher._log_delivery_attempt(
        pool,
        '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
        'report_generated',
        {
            'vendor': 'Acme Rival',
            'data': {
                'company_name': 'Acme Bank',
                'report_id': '44444444-4444-4444-8444-444444444444',
                'review_id': '33333333-3333-4333-8333-333333333334',
            },
        },
        status_code=202,
        response_body='ok',
        duration_ms=85,
        attempt=1,
        success=True,
        error_msg=None,
    )

    sql = pool.execute.call_args[0][0]
    args = pool.execute.call_args[0][1:]
    assert 'signal_id' in sql
    assert 'review_id' in sql
    assert 'report_id' in sql
    assert 'vendor_name' in sql
    assert args[1] == 'report_generated'
    assert args[3] == '44444444-4444-4444-8444-444444444444'
    assert args[4] == '33333333-3333-4333-8333-333333333334'
    assert args[5] == '44444444-4444-4444-8444-444444444444'
    assert args[6] == 'Acme Rival'
    assert args[7] == 'Acme Bank'


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
    assert 'review_id' in sql
    assert args[1] == 'company_signal'
    assert args[2] == '22222222-2222-2222-2222-222222222222'
    assert args[3] is None
    assert args[4] == 'Acme Rival'
    assert args[5] == 'Acme Bank'


@pytest.mark.asyncio
async def test_log_crm_push_persists_review_id_from_envelope():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)

    await webhook_dispatcher._log_crm_push(
        pool,
        '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
        'high_intent_push',
        {
            'vendor': 'Acme Rival',
            'data': {
                'company_name': 'Acme Bank',
                'review_id': '33333333-3333-4333-8333-333333333334',
            },
        },
        crm_record_id='crm-43',
    )

    sql = pool.execute.call_args[0][0]
    args = pool.execute.call_args[0][1:]
    assert 'review_id' in sql
    assert args[1] == 'high_intent_push'
    assert args[2] is None
    assert args[3] == '33333333-3333-4333-8333-333333333334'
    assert args[4] == 'Acme Rival'
    assert args[5] == 'Acme Bank'
    assert args[7] == 'deal'


@pytest.mark.asyncio
async def test_list_webhook_deliveries_exposes_report_context():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': '71c27653-5df3-4d19-9b24-a1a15f61efbc',
                'event_type': 'report_generated',
                'payload': {
                    'vendor': 'Acme Rival',
                    'data': {
                        'report_id': '33333333-3333-4333-8333-333333333333',
                    },
                },
                'status_code': 202,
                'duration_ms': 95,
                'attempt': 1,
                'success': True,
                'error': None,
                'delivered_at': datetime(2026, 4, 11, 4, 0, tzinfo=timezone.utc),
            }
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with patch.object(
            b2b_dashboard,
            '_fetch_webhook_activity_report_context',
            AsyncMock(return_value={
                'report_id': '33333333-3333-4333-8333-333333333333',
                'report_type': 'battle_card',
                'vendor_name': 'Acme Rival',
                'report_title': 'Acme Rival Battle Card',
            }),
        ) as fetch_report_context:
            with patch.object(
                b2b_dashboard,
                '_resolve_webhook_activity_account_focus',
                AsyncMock(return_value=None),
            ):
                result = await b2b_dashboard.list_webhook_deliveries(
                    '71c27653-5df3-4d19-9b24-a1a15f61efbc',
                    success=None,
                    event_type=None,
                    start_date=None,
                    end_date=None,
                    limit=50,
                    user=user,
                )

    delivery = result['deliveries'][0]
    assert delivery['report_id'] == '33333333-3333-4333-8333-333333333333'
    assert delivery['report_type'] == 'battle_card'
    assert delivery['report_title'] == 'Acme Rival Battle Card'
    assert delivery['vendor_name'] == 'Acme Rival'
    fetch_report_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_crm_push_log_exposes_report_context_for_report_generated():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'bb5659f4-dfe6-4c12-8fd6-e6eb5579cc45',
                'signal_type': 'report_generated',
                'signal_id': '33333333-3333-4333-8333-333333333333',
                'vendor_name': None,
                'company_name': None,
                'crm_record_id': 'note-42',
                'crm_record_type': 'note',
                'status': 'success',
                'error': None,
                'pushed_at': datetime(2026, 4, 11, 5, 0, tzinfo=timezone.utc),
            }
        ]
    )
    user = MagicMock(account_id='11111111-1111-1111-1111-111111111111')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with patch.object(
            b2b_dashboard,
            '_fetch_webhook_activity_report_context',
            AsyncMock(return_value={
                'report_id': '33333333-3333-4333-8333-333333333333',
                'report_type': 'battle_card',
                'vendor_name': 'Acme Rival',
                'report_title': 'Acme Rival Battle Card',
            }),
        ) as fetch_report_context:
            with patch.object(
                b2b_dashboard,
                '_resolve_webhook_activity_account_focus',
                AsyncMock(return_value=None),
            ) as resolve_focus:
                result = await b2b_dashboard.list_crm_push_log(
                    'aa5659f4-dfe6-4c12-8fd6-e6eb5579cc44',
                    limit=50,
                    user=user,
                )

    push = result['pushes'][0]
    assert push['review_id'] is None
    assert push['report_id'] == '33333333-3333-4333-8333-333333333333'
    assert push['report_type'] == 'battle_card'
    assert push['report_title'] == 'Acme Rival Battle Card'
    assert push['vendor_name'] == 'Acme Rival'
    fetch_report_context.assert_awaited_once()
    resolve_focus.assert_awaited_once()
    _, kwargs = resolve_focus.await_args
    assert kwargs['signal_id'] is None


@pytest.mark.asyncio
async def test_log_crm_push_persists_report_id_for_report_generated():
    pool = MagicMock()
    pool.execute = AsyncMock(return_value=None)

    await webhook_dispatcher._log_crm_push(
        pool,
        '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
        'report_generated',
        {
            'vendor': 'Acme Rival',
            'data': {
                'report_id': '33333333-3333-4333-8333-333333333333',
                'report_type': 'battle_card',
            },
        },
        crm_record_id='crm-note-42',
    )

    sql = pool.execute.call_args[0][0]
    args = pool.execute.call_args[0][1:]
    assert 'signal_id' in sql
    assert 'review_id' in sql
    assert args[1] == 'report_generated'
    assert args[2] == '33333333-3333-4333-8333-333333333333'
    assert args[3] is None
    assert args[4] == 'Acme Rival'
    assert args[5] is None
    assert args[7] == 'note'
