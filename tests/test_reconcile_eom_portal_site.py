"""Contract tests for one-service Atlas -> EOM Site reconciliation."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import reconcile_eom_portal_site as reconcile  # noqa: E402

SERVICE_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CONTACT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
TOKEN = "a" * 64
ENV = {"ATLAS_TOOLS_EOM_PORTAL_TOKEN": "secret-token"}
BASE_URL = "https://portal.test"


def _service(**over):
    row = {
        "service_id": SERVICE_ID,
        "contact_id": CONTACT_ID,
        "contact_metadata": json.dumps({"portal_customer_id": 7}),
        "rate": "247.50",
        "rate_label": "Per Visit",
    }
    row.update(over)
    return row


def _customers(**site_over):
    site = {
        "id": 31,
        "customerId": 7,
        "active": True,
        "rate": 200.0,
        "rateType": "per_visit",
        "updateToken": TOKEN,
    }
    site.update(site_over)
    return [
        {
            "id": 7,
            "active": True,
            "atlasContactId": str(CONTACT_ID),
            "sites": [site],
        }
    ]


class Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class Client:
    def __init__(self, customers, patch_status=200, payload=None, on_get=None):
        self.customers = customers
        self.patch_status = patch_status
        self.payload = payload
        self.on_get = on_get
        self.patches = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def get(self, url, headers):
        assert headers == {"Authorization": "Bearer secret-token"}
        if self.on_get:
            self.on_get()
        return Response(200, {"success": True, "customers": self.customers})

    def patch(self, url, headers, json):
        self.patches.append((url, headers, json))
        payload = self.payload
        if payload is None:
            payload = {
                "success": True,
                "location": {"id": 31, "rate": 247.5, "rateType": "per_visit"},
            }
        return Response(self.patch_status, payload)


class Pool:
    def __init__(self, row):
        self.row = row
        self.initialized = 0
        self.calls = []
        self.transactions = 0

    async def initialize(self):
        self.initialized += 1

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return self.row

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        yield self


def _main_args(*extra):
    return [
        "--service-id",
        str(SERVICE_ID),
        "--site-id",
        "31",
        "--base-url",
        "https://portal.test/",
        *extra,
    ]


def _run(service, client, *extra, pool=None):
    return reconcile.main(
        _main_args(*extra),
        pool=pool or Pool(service),
        client_factory=lambda: client,
        environ=ENV,
    )


def _plan(service=None, customers=None, base_url=BASE_URL):
    return reconcile.build_plan(
        service or _service(), customers or _customers(), 31, base_url
    )


def test_real_cli_preview_is_stable_and_write_free(capsys):
    service, customers = _service(), _customers()
    first_client, second_client = Client(customers), Client(deepcopy(customers))
    first = _run(service, first_client)
    first_output = capsys.readouterr().out
    second = _run(deepcopy(service), second_client)
    second_output = capsys.readouterr().out

    assert first == second == 0
    assert first_output == second_output
    assert "Preview only; no portal data changed." in first_output
    assert "secret-token" not in first_output
    assert first_client.patches == second_client.patches == []


def test_service_read_is_exact_and_tenant_scoped():
    pool = Pool(_service())
    row = asyncio.run(reconcile.load_service(pool, SERVICE_ID))
    query, args = pool.calls[0]

    assert row["service_id"] == SERVICE_ID
    assert "s.id = $1" in query
    assert "s.business_context_id = $2" in query
    assert "c.business_context_id = $2" in query
    assert "s.status = 'active'" in query and "c.status = 'active'" in query
    assert args == (SERVICE_ID, "effingham_maids")
    with pytest.raises(SystemExit, match="not found"):
        asyncio.run(reconcile.load_service(Pool(None), SERVICE_ID))


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Per Visit", "per_visit"),
        ("Per Hour", "hourly"),
        ("Per Month", "monthly"),
    ],
)
def test_only_exact_supported_rate_labels_map(label, expected):
    plan = _plan(_service(rate_label=label))
    assert plan["desired"] == {"rate": 247.5, "rateType": expected}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_stamp", "stamped portal Customer ID"),
        ("wrong_owner", "owned by another Customer"),
        ("inconsistent_site", "inconsistent Customer ownership"),
        ("wrong_contact", "another Atlas Contact"),
        ("inactive_site", "not found uniquely"),
        ("inactive_customer", "not found uniquely"),
        ("bad_token", "valid update token"),
        ("unsupported_label", "Unsupported Atlas rate label"),
    ],
)
def test_identity_and_mapping_boundaries_fail_closed(case, message):
    service, customers = _service(), _customers()
    if case == "missing_stamp":
        service["contact_metadata"] = "{}"
    elif case == "wrong_owner":
        customers[0]["id"] = 8
        customers[0]["sites"][0]["customerId"] = 8
    elif case == "inconsistent_site":
        customers[0]["sites"][0]["customerId"] = 8
    elif case == "wrong_contact":
        customers[0]["atlasContactId"] = str(SERVICE_ID)
    elif case == "inactive_site":
        customers[0]["sites"][0]["active"] = False
    elif case == "inactive_customer":
        customers[0].pop("active")
    elif case == "bad_token":
        customers[0]["sites"][0]["updateToken"] = "stale"
    else:
        service["rate_label"] = "Monthly"

    client = Client(customers)
    with pytest.raises(SystemExit, match=message):
        _run(service, client)
    assert client.patches == []


def test_apply_requires_matching_fresh_hash_before_patch():
    customers = _customers()
    service, pool = _service(rate="247.50"), Pool(_service(rate="247.50"))
    stale = reconcile.plan_hash(_plan(service, customers))

    def change_source():
        pool.row = _service(rate="300.00")

    client = Client(customers, on_get=change_source)
    with pytest.raises(SystemExit, match="Plan hash mismatch"):
        _run(service, client, "--apply", "--plan-hash", stale, pool=pool)
    assert client.patches == []
    assert pool.transactions == 1
    assert "FOR SHARE OF s, c" in pool.calls[0][0]


def test_apply_sends_only_guarded_site_economics(capsys):
    service, customers = _service(), _customers()
    digest = reconcile.plan_hash(_plan(service, customers))
    client = Client(customers)
    pool = Pool(service)

    result = _run(
        service, client, "--apply", "--plan-hash", digest, pool=pool
    )

    assert result == 0
    assert client.patches == [
        (
            "https://portal.test/api/admin/locations/31",
            {"Authorization": "Bearer secret-token"},
            {
                "expectedUpdateToken": TOKEN,
                "rate": 247.5,
                "rateType": "per_visit",
            },
        )
    ]
    assert pool.transactions == 1
    assert "FOR SHARE OF s, c" in pool.calls[0][0]
    assert "Updated portal Site 31." in capsys.readouterr().out


def test_unchanged_apply_is_noop():
    service, customers = _service(), _customers(rate=247.5)
    plan = _plan(service, customers)
    client = Client(customers)

    assert plan["action"] == "noop"
    assert (
        _run(service, client, "--apply", "--plan-hash", reconcile.plan_hash(plan))
        == 0
    )
    assert client.patches == []


@pytest.mark.parametrize("status", [409, 500])
def test_http_failure_is_nonzero_and_never_retried(status):
    service, customers = _service(), _customers()
    plan = _plan(service, customers)
    client = Client(customers, patch_status=status)

    with pytest.raises(SystemExit, match=f"HTTP {status}"):
        _run(service, client, "--apply", "--plan-hash", reconcile.plan_hash(plan))
    assert len(client.patches) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"success": True, "location": {"id": True, "rate": 247.5, "rateType": "per_visit"}},
        {"success": True, "location": {"id": 31, "rate": 1.0, "rateType": "per_visit"}},
        {"success": False, "location": {"id": 31, "rate": 247.5, "rateType": "per_visit"}},
        [{"unexpected": "shape"}],
    ],
)
def test_invalid_success_confirmation_never_prints_success(payload, capsys):
    service, customers = _service(), _customers()
    plan = _plan(service, customers)
    client = Client(customers, payload=payload)
    with pytest.raises(SystemExit, match="invalid confirmation"):
        _run(service, client, "--apply", "--plan-hash", reconcile.plan_hash(plan))
    assert len(client.patches) == 1
    assert "Updated portal Site" not in capsys.readouterr().out


def test_confirmation_rejects_bool_collision_with_site_one():
    plan = _plan()
    plan["portal"]["siteId"] = 1
    client = Client(
        _customers(),
        payload={
            "success": True,
            "location": {"id": True, "rate": 247.5, "rateType": "per_visit"},
        },
    )
    with pytest.raises(SystemExit, match="invalid confirmation"):
        reconcile.apply_plan(client, "secret-token", plan)


def test_plan_hash_is_bound_to_portal_origin():
    assert reconcile.plan_hash(_plan()) != reconcile.plan_hash(
        _plan(base_url="https://other-portal.test")
    )


def test_apply_without_hash_is_rejected_before_external_io():
    pool, client = Pool(_service()), Client(_customers())
    with pytest.raises(SystemExit) as exc:
        reconcile.main(
            _main_args("--apply"),
            pool=pool,
            client_factory=lambda: client,
            environ=ENV,
        )
    assert exc.value.code == 2
    assert pool.initialized == 0 and client.patches == []
