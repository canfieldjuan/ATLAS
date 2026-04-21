"""Focused webhook list contract tests for the B2B dashboard API."""

import importlib
import json
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
async def test_create_webhook_normalizes_event_types_and_trims_crm_auth_header():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            'account_id': 'account-1',
            'url': 'https://hooks.example.com/churn',
            'event_types': ['churn_alert', 'high_intent_push', 'signal_update'],
            'channel': 'crm_hubspot',
            'enabled': True,
            'description': 'PagerDuty bridge',
            'created_at': created_at,
        }
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.CreateWebhookBody(
        url='  https://hooks.example.com/churn  ',
        secret='  xxxxxxxxxxxxxxxx  ',
        event_types=[' churn_alert ', 'high_intent_push', 'signal_update', 'churn_alert'],
        channel='CRM_HUBSPOT',
        auth_header='  Bearer token  ',
        description='  PagerDuty bridge  ',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.create_webhook(body=body, user=user)

    _, inserted_account_id, inserted_url, inserted_secret, inserted_event_types, inserted_channel, inserted_auth_header, inserted_description = pool.fetchrow.await_args.args
    assert inserted_account_id == 'account-1'
    assert inserted_url == 'https://hooks.example.com/churn'
    assert inserted_secret == 'x' * b2b_dashboard.MIN_WEBHOOK_SECRET_LENGTH
    assert inserted_event_types == ['churn_alert', 'high_intent_push', 'signal_update']
    assert inserted_channel == 'crm_hubspot'
    assert inserted_auth_header == 'Bearer token'
    assert inserted_description == 'PagerDuty bridge'
    assert result['event_types'] == ['churn_alert', 'high_intent_push', 'signal_update']
    assert result['channel'] == 'crm_hubspot'


@pytest.mark.asyncio
async def test_create_webhook_rejects_blank_crm_auth_header():
    pool = MagicMock()
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.CreateWebhookBody(
        url='https://hooks.example.com/churn',
        secret='x' * b2b_dashboard.MIN_WEBHOOK_SECRET_LENGTH,
        event_types=['churn_alert'],
        channel='crm_hubspot',
        auth_header='   ',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.create_webhook(body=body, user=user)

    assert exc_info.value.status_code == 400
    assert 'auth_header is required' in exc_info.value.detail


@pytest.mark.asyncio
async def test_create_webhook_normalizes_blank_description_to_none():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            'id': '8bbf17ec-39a6-4df5-bc34-790183c53977',
            'account_id': 'account-1',
            'url': 'https://hooks.example.com/churn',
            'event_types': ['churn_alert'],
            'channel': 'generic',
            'enabled': True,
            'description': None,
            'created_at': created_at,
        }
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.CreateWebhookBody(
        url='https://hooks.example.com/churn',
        secret='x' * b2b_dashboard.MIN_WEBHOOK_SECRET_LENGTH,
        event_types=['churn_alert'],
        channel='generic',
        description='   ',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.create_webhook(body=body, user=user)

    assert pool.fetchrow.await_args.args[7] is None
    assert result['description'] is None


@pytest.mark.asyncio
async def test_create_webhook_rejects_non_global_destination_when_disabled():
    pool = MagicMock()
    user = MagicMock(account_id="account-1")
    body = b2b_dashboard.CreateWebhookBody(
        url="http://127.0.0.1:8080/hook",
        secret="x" * b2b_dashboard.MIN_WEBHOOK_SECRET_LENGTH,
        event_types=["churn_alert"],
        channel="generic",
    )

    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool), patch.object(
        b2b_dashboard.settings.b2b_webhook,
        "allow_non_global_destinations",
        False,
    ):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.create_webhook(body=body, user=user)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Webhook destination cannot target non-global address: 127.0.0.1"
    pool.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_create_webhook_rejects_embedded_credentials_in_url():
    pool = MagicMock()
    user = MagicMock(account_id="account-1")
    body = b2b_dashboard.CreateWebhookBody(
        url="https://user:pass@hooks.example.com/churn",
        secret="x" * b2b_dashboard.MIN_WEBHOOK_SECRET_LENGTH,
        event_types=["churn_alert"],
        channel="generic",
    )

    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.create_webhook(body=body, user=user)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Webhook URL must not include embedded credentials"
    pool.fetchrow.assert_not_called()


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
async def test_list_webhooks_excludes_synthetic_tests_from_recent_metrics_and_failure_summary():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    tested_at = datetime.now(timezone.utc) - timedelta(hours=3)
    pool = MagicMock()
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': '911f9b32-cc42-4f22-84a5-c4ae77d6e54d',
                'url': 'https://hooks.example.com/churn',
                'event_types': ['churn_alert', 'signal_update'],
                'channel': 'generic',
                'enabled': True,
                'description': 'PagerDuty bridge',
                'created_at': created_at,
                'updated_at': created_at,
                'recent_deliveries': 0,
                'recent_successes': 0,
                'latest_failure_event_type': None,
                'latest_failure_status_code': None,
                'latest_failure_error': None,
                'latest_failure_at': None,
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

    query = pool.fetch.await_args.args[0]
    assert "LEFT JOIN LATERAL (" in query
    assert "recent_stats.recent_deliveries" in query
    assert "latest_failure.event_type AS latest_failure_event_type" in query
    assert "latest_test.success AS latest_test_success" in query
    assert "AND dl.event_type <> 'test'" in query
    assert "AND dl.event_type = 'test'" in query
    assert "ORDER BY dl.delivered_at DESC, dl.id DESC" in query
    webhook = result['webhooks'][0]
    assert webhook['recent_deliveries_7d'] == 0
    assert webhook['recent_success_rate_7d'] is None
    assert webhook['latest_failure_event_type'] is None
    assert webhook['latest_test_success'] is False


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
async def test_update_webhook_normalizes_event_types_and_trimmed_fields():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                'id': 'f6ce3a70-c7d1-4c84-a418-a9c3555d6a14',
                'channel': 'generic',
            },
            {
            'id': 'f6ce3a70-c7d1-4c84-a418-a9c3555d6a14',
            'url': 'https://hooks.example.com/updated',
            'event_types': ['signal_update', 'churn_alert'],
            'channel': 'generic',
            'enabled': True,
            'description': 'Updated webhook',
            'created_at': created_at,
            'updated_at': updated_at,
            },
        ]
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(
        url='  https://hooks.example.com/updated  ',
        event_types=[' signal_update ', 'churn_alert', 'signal_update'],
        description='  Updated webhook  ',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.update_webhook(
            webhook_id='f6ce3a70-c7d1-4c84-a418-a9c3555d6a14',
            body=body,
            user=user,
        )

    existing_query, existing_webhook_id, existing_account_id = pool.fetchrow.await_args_list[0].args
    query, *query_params = pool.fetchrow.await_args_list[1].args
    assert 'SELECT id, COALESCE(channel, \'generic\') AS channel' in existing_query
    assert existing_webhook_id == b2b_dashboard._uuid.UUID('f6ce3a70-c7d1-4c84-a418-a9c3555d6a14')
    assert existing_account_id == 'account-1'
    assert 'event_types = $1' in query
    assert 'url = $2' in query
    assert 'description = $3' in query
    assert query_params[0] == ['signal_update', 'churn_alert']
    assert query_params[1] == 'https://hooks.example.com/updated'
    assert query_params[2] == 'Updated webhook'
    assert result['event_types'] == ['signal_update', 'churn_alert']
    assert result['url'] == 'https://hooks.example.com/updated'


@pytest.mark.asyncio
async def test_update_webhook_rotates_secret_and_trims_crm_auth_header():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                'channel': 'crm_hubspot',
            },
            {
                'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                'url': 'https://hooks.example.com/churn',
                'event_types': ['churn_alert', 'signal_update'],
                'channel': 'crm_hubspot',
                'enabled': True,
                'description': 'PagerDuty bridge',
                'created_at': created_at,
                'updated_at': updated_at,
            },
        ]
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(
        secret='  atlas_secret_1234567890  ',
        auth_header='  Bearer rotated-token  ',
    )

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.update_webhook(
            webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            body=body,
            user=user,
        )

    query, *query_params = pool.fetchrow.await_args_list[1].args
    assert 'secret = $1' in query
    assert 'auth_header = $2' in query
    assert query_params[0] == 'atlas_secret_1234567890'
    assert query_params[1] == 'Bearer rotated-token'
    assert result['channel'] == 'crm_hubspot'


@pytest.mark.asyncio
async def test_update_webhook_rejects_blank_crm_auth_header():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            'channel': 'crm_hubspot',
        }
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(auth_header='   ')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.update_webhook(
                webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                body=body,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert 'auth_header is required' in exc_info.value.detail


@pytest.mark.asyncio
async def test_update_webhook_rejects_short_secret():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            'id': '2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            'channel': 'generic',
        }
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(secret='short')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.update_webhook(
                webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                body=body,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        f'secret must be at least {b2b_dashboard.MIN_WEBHOOK_SECRET_LENGTH} characters'
    )


@pytest.mark.asyncio
async def test_update_webhook_clears_blank_description():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                'id': '6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf',
                'channel': 'generic',
            },
            {
                'id': '6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf',
                'url': 'https://hooks.example.com/churn',
                'event_types': ['churn_alert'],
                'channel': 'generic',
                'enabled': True,
                'description': None,
                'created_at': created_at,
                'updated_at': updated_at,
            },
        ]
    )
    user = MagicMock(account_id='account-1')
    body = b2b_dashboard.UpdateWebhookBody(description='   ')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.update_webhook(
            webhook_id='6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf',
            body=body,
            user=user,
        )

    query, *query_params = pool.fetchrow.await_args_list[1].args
    assert 'description = $1' in query
    assert query_params[0] is None
    assert result['description'] is None


@pytest.mark.asyncio
async def test_update_webhook_allows_non_global_destination_when_enabled():
    created_at = datetime.now(timezone.utc) - timedelta(days=1)
    updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        side_effect=[
            {
                "id": "6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf",
                "channel": "generic",
            },
            {
                "id": "6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf",
                "url": "http://127.0.0.1:8080/hook",
                "event_types": ["churn_alert"],
                "channel": "generic",
                "enabled": True,
                "description": "Internal test hook",
                "created_at": created_at,
                "updated_at": updated_at,
            },
        ]
    )
    user = MagicMock(account_id="account-1")
    body = b2b_dashboard.UpdateWebhookBody(url="http://127.0.0.1:8080/hook")

    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool), patch.object(
        b2b_dashboard.settings.b2b_webhook,
        "allow_non_global_destinations",
        True,
    ):
        result = await b2b_dashboard.update_webhook(
            webhook_id="6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf",
            body=body,
            user=user,
        )

    query, *query_params = pool.fetchrow.await_args_list[1].args
    assert "url = $1" in query
    assert query_params[0] == "http://127.0.0.1:8080/hook"
    assert result["url"] == "http://127.0.0.1:8080/hook"


@pytest.mark.asyncio
async def test_update_webhook_rejects_embedded_credentials_in_url():
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            "id": "6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf",
            "channel": "generic",
        }
    )
    user = MagicMock(account_id="account-1")
    body = b2b_dashboard.UpdateWebhookBody(url="https://user:pass@hooks.example.com/churn")

    with patch.object(b2b_dashboard, "_pool_or_503", return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.update_webhook(
                webhook_id="6dce7414-13bc-4d1d-a183-e1f2ff2b9dbf",
                body=body,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Webhook URL must not include embedded credentials"
    assert pool.fetchrow.await_count == 1


@pytest.mark.asyncio
async def test_webhook_delivery_summary_excludes_synthetic_tests_from_health_metrics():
    delivered_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    pool = MagicMock()
    pool.fetchrow = AsyncMock(
        return_value={
            'active_subscriptions': 2,
            'total_deliveries': 3,
            'successful': 2,
            'failed': 1,
            'avg_success_duration_ms': 212.4,
            'p95_success_duration_ms': 480.1,
            'last_delivery_at': delivered_at,
        }
    )
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.webhook_delivery_summary(days=7, user=user)

    query = pool.fetchrow.await_args.args[0]
    assert "AND dl.event_type <> 'test'" in query
    assert result['total_deliveries'] == 3
    assert result['successful'] == 2
    assert result['failed'] == 1
    assert result['success_rate'] == 0.667
    assert result['last_delivery_at'] == delivered_at.isoformat()


@pytest.mark.asyncio
async def test_list_webhook_deliveries_exposes_vendor_context_from_payload():
    delivered_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'delivery-1',
                'event_type': 'signal_update',
                'status_code': 202,
                'duration_ms': 185,
                'attempt': 1,
                'success': True,
                'error': None,
                'delivered_at': delivered_at,
                'payload': json.dumps({
                    'vendor': 'Acme Rival',
                    'data': {
                        'company_name': 'Acme Bank',
                        'signal_type': 'competitive_displacement',
                    },
                }),
            }
        ]
    )
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.list_webhook_deliveries(
            webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            success=None,
            event_type=None,
            start_date=None,
            end_date=None,
            limit=50,
            user=user,
        )

    assert result['count'] == 1
    delivery = result['deliveries'][0]
    assert delivery['event_type'] == 'signal_update'
    assert delivery['vendor_name'] == 'Acme Rival'
    assert delivery['company_name'] == 'Acme Bank'
    assert delivery['signal_type'] == 'competitive_displacement'
    assert delivery['delivered_at'] == delivered_at.isoformat()
    query, *_ = pool.fetch.await_args.args
    assert "ORDER BY delivered_at DESC, id DESC" in query


@pytest.mark.asyncio
async def test_list_webhook_deliveries_hides_synthetic_test_vendor_context():
    delivered_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'delivery-test-1',
                'event_type': 'test',
                'status_code': 202,
                'duration_ms': 120,
                'attempt': 1,
                'success': True,
                'error': None,
                'delivered_at': delivered_at,
                'payload': json.dumps({
                    'vendor': 'test_vendor',
                    'data': {
                        'company_name': 'Synthetic Account',
                        'signal_type': 'manual_test',
                    },
                }),
            }
        ]
    )
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.list_webhook_deliveries(
            webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            success=None,
            event_type=None,
            start_date=None,
            end_date=None,
            limit=50,
            user=user,
        )

    assert result['count'] == 1
    delivery = result['deliveries'][0]
    assert delivery['event_type'] == 'test'
    assert delivery['vendor_name'] is None
    assert delivery['company_name'] is None
    assert delivery['signal_type'] is None


@pytest.mark.asyncio
async def test_list_webhook_deliveries_accepts_test_event_type_filter():
    delivered_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'delivery-test-1',
                'event_type': 'test',
                'status_code': 202,
                'duration_ms': 120,
                'attempt': 1,
                'success': True,
                'error': None,
                'delivered_at': delivered_at,
                'payload': '{}',
            }
        ]
    )
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.list_webhook_deliveries(
            webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            success=None,
            event_type='test',
            start_date=None,
            end_date=None,
            limit=50,
            user=user,
        )

    assert result['count'] == 1
    assert result['deliveries'][0]['event_type'] == 'test'


@pytest.mark.asyncio
async def test_list_webhook_deliveries_rejects_invalid_event_type_filter():
    pool = MagicMock()
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.list_webhook_deliveries(
                webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                success=None,
                event_type='bogus',
                start_date=None,
                end_date=None,
                limit=50,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert "event_type must be one of" in exc_info.value.detail
    pool.fetchval.assert_not_called()
    pool.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_list_crm_push_log_treats_pushed_rows_as_success_filter():
    delivered_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    pool.fetch = AsyncMock(
        return_value=[
            {
                'id': 'push-1',
                'signal_type': 'company_signal',
                'signal_id': None,
                'vendor_name': 'Acme Rival',
                'company_name': 'Acme Bank',
                'crm_record_id': 'deal-1',
                'crm_record_type': 'deal',
                'status': 'pushed',
                'error': None,
                'pushed_at': delivered_at,
            }
        ]
    )
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        result = await b2b_dashboard.list_crm_push_log(
            webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
            status='success',
            limit=25,
            user=user,
        )

    assert result['count'] == 1
    push = result['pushes'][0]
    assert push['status'] == 'pushed'
    query, wid, statuses, limit = pool.fetch.await_args.args
    assert "status = ANY($2::text[])" in query
    assert "ORDER BY pushed_at DESC, id DESC" in query
    assert statuses == ['success', 'pushed']
    assert limit == 25


@pytest.mark.asyncio
async def test_list_crm_push_log_rejects_invalid_status_filter():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=1)
    user = MagicMock(account_id='account-1')

    with patch.object(b2b_dashboard, '_pool_or_503', return_value=pool):
        with pytest.raises(b2b_dashboard.HTTPException) as exc_info:
            await b2b_dashboard.list_crm_push_log(
                webhook_id='2ea3fd03-7fd9-4b72-8f24-117667f723e9',
                status='pending',
                limit=25,
                user=user,
            )

    assert exc_info.value.status_code == 400
    assert "status must be 'success' or 'failed'" in exc_info.value.detail
