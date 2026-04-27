import sys
from types import SimpleNamespace

from starlette.requests import Request
from unittest.mock import AsyncMock, MagicMock

import pytest

_asyncpg_mock = MagicMock()
_asyncpg_exceptions = MagicMock()
_asyncpg_exceptions.UndefinedTableError = type("UndefinedTableError", (Exception,), {})
_asyncpg_mock.exceptions = _asyncpg_exceptions
sys.modules.setdefault("asyncpg", _asyncpg_mock)
sys.modules.setdefault("asyncpg.exceptions", _asyncpg_exceptions)


def _request():
    return Request({
        "type": "http",
        "method": "POST",
        "path": "/test",
        "headers": [],
    })


class FakePool:
    def __init__(self, *, fetchrow_map=None, fetch_map=None):
        self.fetchrow_map = fetchrow_map or {}
        self.fetch_map = fetch_map or {}
        self.calls = []

    async def fetchrow(self, query, *params):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        for needle, value in self.fetchrow_map.items():
            if needle in normalized:
                return value(*params) if callable(value) else value
        raise AssertionError(f"Unexpected fetchrow query: {normalized}")

    async def fetch(self, query, *params):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        for needle, value in self.fetch_map.items():
            if needle in normalized:
                return value(*params) if callable(value) else value
        raise AssertionError(f"Unexpected fetch query: {normalized}")


@pytest.mark.asyncio
async def test_compute_prediction_uses_exact_signal_adapter_for_review_gate(monkeypatch):
    from atlas_brain.api import b2b_win_loss as mod
    from atlas_brain.autonomous.tasks import _b2b_shared as shared_mod

    pool = FakePool(
        fetchrow_map={
            "AS edges,": {
                "edges": 3,
                "pain_points": 0,
                "buyer_profiles": 0,
                "product_profiles": 0,
                "outcome_sequences": 0,
            },
        },
        fetch_map={
            "FROM b2b_displacement_edges": [
                {
                    "to_vendor": "Freshdesk",
                    "mention_count": 8,
                    "primary_driver": "support",
                    "signal_strength": "strong",
                    "confidence_score": 0.8,
                    "key_quote": None,
                },
                {
                    "to_vendor": "Intercom",
                    "mention_count": 5,
                    "primary_driver": "pricing",
                    "signal_strength": "moderate",
                    "confidence_score": 0.6,
                    "key_quote": None,
                },
            ],
            "FROM b2b_vendor_witnesses": [],
        },
    )
    signal_row = {
        "vendor_name": "Zendesk",
        "product_category": "CRM",
        "total_reviews": 12,
        "negative_reviews": 4,
        "churn_intent_count": 3,
        "avg_urgency_score": 8.0,
        "decision_maker_churn_rate": 0.2,
        "nps_proxy": -0.1,
        "price_complaint_rate": 0.3,
        "archetype": "price_squeeze",
        "confidence_score": 0.8,
    }
    read_signal = AsyncMock(return_value=signal_row)

    monkeypatch.setattr(shared_mod, "read_vendor_signal_detail_exact", read_signal)
    monkeypatch.setattr(
        mod,
        "_load_calibrated_weights",
        AsyncMock(return_value=(dict(mod.WEIGHTS), "static", None)),
    )
    monkeypatch.setattr(mod, "call_llm_with_skill", lambda *args, **kwargs: None)

    result = await mod._compute_prediction(pool, "Zendesk")

    read_signal.assert_awaited_once_with(pool, vendor_name="Zendesk")
    assert result.is_gated is False
    assert result.data_coverage["reviews"] == 12
    churn_factor = next(f for f in result.factors if f.name == "Churn Severity")
    assert churn_factor.gated is False
    assert churn_factor.data_points == 12
    assert "3/12 reviews show churn intent" in churn_factor.evidence
    assert all("b2b_churn_signals" not in query for query, _ in pool.calls)



@pytest.mark.asyncio
async def test_predict_win_loss_rejects_blank_vendor_before_db_touch(monkeypatch):
    from atlas_brain.api import b2b_win_loss as mod

    resolve_vendor = AsyncMock()

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "resolve_vendor_name", resolve_vendor)
    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.predict_win_loss(
            _request(),
            mod.WinLossRequest(vendor_name="   "),
            user=SimpleNamespace(account_id="acct-1"),
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == "vendor_name is required"
    resolve_vendor.assert_not_awaited()


@pytest.mark.asyncio
async def test_predict_win_loss_trims_vendor_before_resolution_and_db(monkeypatch):
    from atlas_brain.api import b2b_win_loss as mod

    pool = object()
    resolve_vendor = AsyncMock(return_value="Zendesk")
    compute_prediction = AsyncMock(
        return_value=mod.WinLossResponse(
            vendor_name="Zendesk",
            win_probability=0.62,
            confidence="medium",
            verdict="moderate",
        )
    )
    persist_prediction = AsyncMock(return_value="pred-1")
    get_pool = MagicMock(return_value=pool)

    monkeypatch.setattr(mod, "resolve_vendor_name", resolve_vendor)
    monkeypatch.setattr(mod, "_compute_prediction", compute_prediction)
    monkeypatch.setattr(mod, "_persist_prediction", persist_prediction)
    monkeypatch.setattr(mod, "get_db_pool", get_pool)

    response = await mod.predict_win_loss(
        _request(),
        mod.WinLossRequest(vendor_name="  Zendesk  ", company_size="smb", industry="saas"),
        user=SimpleNamespace(account_id="acct-1"),
    )

    resolve_vendor.assert_awaited_once_with("Zendesk")
    get_pool.assert_called_once_with()
    compute_prediction.assert_awaited_once_with(pool, "Zendesk", "smb", "saas")
    persist_prediction.assert_awaited_once()
    assert response.prediction_id == "pred-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vendor_a", "vendor_b", "field_name"),
    [
        ("   ", "Freshdesk", "vendor_a"),
        ("Zendesk", "   ", "vendor_b"),
    ],
)
async def test_compare_win_loss_rejects_blank_vendors_before_db_touch(
    monkeypatch, vendor_a, vendor_b, field_name
):
    from atlas_brain.api import b2b_win_loss as mod

    resolve_vendor = AsyncMock()

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "resolve_vendor_name", resolve_vendor)
    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.compare_win_loss(
            _request(),
            mod.WinLossCompareRequest(vendor_a=vendor_a, vendor_b=vendor_b),
            user=SimpleNamespace(account_id="acct-1"),
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == f"{field_name} is required"
    resolve_vendor.assert_not_awaited()


@pytest.mark.asyncio
async def test_compare_win_loss_rejects_same_vendor_before_db_touch(monkeypatch):
    from atlas_brain.api import b2b_win_loss as mod

    resolve_vendor = AsyncMock(side_effect=["Zendesk", "zendesk"])

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "resolve_vendor_name", resolve_vendor)
    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.compare_win_loss(
            _request(),
            mod.WinLossCompareRequest(vendor_a=" Zendesk ", vendor_b="zendesk"),
            user=SimpleNamespace(account_id="acct-1"),
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Cannot compare a vendor against itself"


@pytest.mark.asyncio
@pytest.mark.parametrize("route_name", ["get_prediction", "export_prediction_csv"])
async def test_prediction_routes_validate_ids_before_db_touch(monkeypatch, route_name):
    from atlas_brain.api import b2b_win_loss as mod

    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await getattr(mod, route_name)(" not-a-uuid ", user=SimpleNamespace(account_id="acct-1"))

    assert exc.value.status_code == 400
    assert exc.value.detail == "Invalid prediction ID"


# -- Step 1 hardening: nullable win_probability across persistence/compare/export --


@pytest.mark.asyncio
async def test_persist_prediction_stores_null_win_probability_when_gated():
    from atlas_brain.api import b2b_win_loss as mod

    captured = {}

    class _Pool:
        async def fetchrow(self, query, *params):
            captured["query"] = " ".join(str(query).split())
            captured["params"] = params
            return {"id": "pred-1"}

    gated = mod.WinLossResponse(
        vendor_name="Zendesk",
        win_probability=None,
        confidence="insufficient",
        verdict="Insufficient data.",
        is_gated=True,
    )
    req = mod.WinLossRequest(vendor_name="Zendesk")

    pid = await mod._persist_prediction(_Pool(), "acct-1", req, gated)

    assert pid == "pred-1"
    # win_probability is param index 4 (0-indexed) per the INSERT column order
    # (account_id, vendor_name, company_size, industry, win_probability, ...).
    assert captured["params"][4] is None


@pytest.mark.asyncio
async def test_compare_win_loss_suppresses_easier_target_when_one_side_gated(monkeypatch):
    from atlas_brain.api import b2b_win_loss as mod

    pool = object()
    monkeypatch.setattr(mod, "get_db_pool", MagicMock(return_value=pool))
    monkeypatch.setattr(
        mod,
        "resolve_vendor_name",
        AsyncMock(side_effect=["Zendesk", "Freshdesk"]),
    )
    monkeypatch.setattr(
        mod,
        "_compute_prediction",
        AsyncMock(side_effect=[
            mod.WinLossResponse(
                vendor_name="Zendesk",
                win_probability=0.62,
                confidence="medium",
                verdict="moderate",
                is_gated=False,
            ),
            mod.WinLossResponse(
                vendor_name="Freshdesk",
                win_probability=None,
                confidence="insufficient",
                verdict="Insufficient data.",
                is_gated=True,
            ),
        ]),
    )
    monkeypatch.setattr(mod, "_persist_prediction", AsyncMock(return_value=None))

    response = await mod.compare_win_loss(
        _request(),
        mod.WinLossCompareRequest(vendor_a="Zendesk", vendor_b="Freshdesk"),
        user=SimpleNamespace(account_id="acct-1"),
    )

    assert response.is_gated is True
    assert response.easier_target == "insufficient_data"
    assert response.probability_delta == 0
    assert response.gated_reason is not None
    assert "Freshdesk" in response.gated_reason


@pytest.mark.asyncio
async def test_compare_win_loss_suppresses_easier_target_when_both_sides_gated(monkeypatch):
    from atlas_brain.api import b2b_win_loss as mod

    pool = object()
    monkeypatch.setattr(mod, "get_db_pool", MagicMock(return_value=pool))
    monkeypatch.setattr(
        mod,
        "resolve_vendor_name",
        AsyncMock(side_effect=["Zendesk", "Freshdesk"]),
    )
    monkeypatch.setattr(
        mod,
        "_compute_prediction",
        AsyncMock(side_effect=[
            mod.WinLossResponse(
                vendor_name="Zendesk",
                win_probability=None,
                confidence="insufficient",
                verdict="Insufficient data.",
                is_gated=True,
            ),
            mod.WinLossResponse(
                vendor_name="Freshdesk",
                win_probability=None,
                confidence="insufficient",
                verdict="Insufficient data.",
                is_gated=True,
            ),
        ]),
    )
    monkeypatch.setattr(mod, "_persist_prediction", AsyncMock(return_value=None))

    response = await mod.compare_win_loss(
        _request(),
        mod.WinLossCompareRequest(vendor_a="Zendesk", vendor_b="Freshdesk"),
        user=SimpleNamespace(account_id="acct-1"),
    )

    assert response.is_gated is True
    assert response.easier_target == "insufficient_data"
    assert response.probability_delta == 0
    assert "Zendesk" in (response.gated_reason or "")
    assert "Freshdesk" in (response.gated_reason or "")


@pytest.mark.asyncio
async def test_compare_win_loss_treats_null_probability_as_gated_even_if_flag_false(monkeypatch):
    # Defense in depth: a malformed response (is_gated=False but
    # win_probability=None) must not reach the subtraction step. Treat the
    # probability contract as authoritative, not just the gate flag.
    from atlas_brain.api import b2b_win_loss as mod

    pool = object()
    monkeypatch.setattr(mod, "get_db_pool", MagicMock(return_value=pool))
    monkeypatch.setattr(
        mod,
        "resolve_vendor_name",
        AsyncMock(side_effect=["Zendesk", "Freshdesk"]),
    )
    monkeypatch.setattr(
        mod,
        "_compute_prediction",
        AsyncMock(side_effect=[
            mod.WinLossResponse(
                vendor_name="Zendesk",
                win_probability=0.62,
                confidence="medium",
                verdict="moderate",
                is_gated=False,
            ),
            mod.WinLossResponse(
                vendor_name="Freshdesk",
                win_probability=None,
                confidence="medium",
                verdict="claims sufficient but probability missing",
                is_gated=False,
            ),
        ]),
    )
    monkeypatch.setattr(mod, "_persist_prediction", AsyncMock(return_value=None))

    response = await mod.compare_win_loss(
        _request(),
        mod.WinLossCompareRequest(vendor_a="Zendesk", vendor_b="Freshdesk"),
        user=SimpleNamespace(account_id="acct-1"),
    )

    assert response.is_gated is True
    assert response.easier_target == "insufficient_data"
    assert response.probability_delta == 0
    assert "Freshdesk" in (response.gated_reason or "")


@pytest.mark.asyncio
async def test_list_recent_predictions_handles_null_win_probability(monkeypatch):
    from datetime import datetime, timezone
    from atlas_brain.api import b2b_win_loss as mod

    rows = [
        {
            "id": "pred-1",
            "vendor_name": "Zendesk",
            "win_probability": 0.62,
            "confidence": "medium",
            "is_gated": False,
            "created_at": datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
        },
        {
            "id": "pred-2",
            "vendor_name": "Freshdesk",
            "win_probability": None,
            "confidence": "insufficient",
            "is_gated": True,
            "created_at": datetime(2026, 4, 27, 12, 5, tzinfo=timezone.utc),
        },
    ]

    class _Pool:
        async def fetch(self, query, *params):
            return rows

    monkeypatch.setattr(mod, "get_db_pool", MagicMock(return_value=_Pool()))

    result = await mod.list_recent_predictions(
        limit=10, user=SimpleNamespace(account_id="acct-1")
    )

    assert result["count"] == 2
    assert result["predictions"][0].win_probability == 0.62
    assert result["predictions"][1].win_probability is None
    assert result["predictions"][1].is_gated is True


@pytest.mark.asyncio
async def test_get_prediction_handles_null_win_probability(monkeypatch):
    from datetime import datetime, timezone
    from uuid import uuid4
    from atlas_brain.api import b2b_win_loss as mod

    pid = uuid4()
    row = {
        "id": pid,
        "vendor_name": "Freshdesk",
        "win_probability": None,
        "confidence": "insufficient",
        "verdict": "Insufficient data.",
        "is_gated": True,
        "data_gates": "[]",
        "factors": "[]",
        "switching_triggers": "[]",
        "proof_quotes": "[]",
        "objections": "[]",
        "displacement_targets": "[]",
        "segment_match": None,
        "data_coverage": "{}",
        "weights_source": "static",
        "calibration_version": None,
        "recommended_approach": None,
        "lead_with": "[]",
        "talking_points": "[]",
        "timing_advice": None,
        "risk_factors": "[]",
        "created_at": datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
    }

    class _Pool:
        async def fetchrow(self, query, *params):
            return row

    monkeypatch.setattr(mod, "get_db_pool", MagicMock(return_value=_Pool()))

    response = await mod.get_prediction(
        str(pid), user=SimpleNamespace(account_id="acct-1")
    )

    assert response.win_probability is None
    assert response.is_gated is True
    assert response.confidence == "insufficient"


@pytest.mark.asyncio
async def test_export_prediction_csv_renders_unavailable_for_null_win_probability(monkeypatch):
    from datetime import datetime, timezone
    from uuid import uuid4
    from atlas_brain.api import b2b_win_loss as mod

    pid = uuid4()
    row = {
        "id": pid,
        "vendor_name": "Freshdesk",
        "win_probability": None,
        "confidence": "insufficient",
        "verdict": "Insufficient data.",
        "is_gated": True,
        "weights_source": "static",
        "calibration_version": None,
        "company_size": None,
        "industry": None,
        "factors": "[]",
        "data_gates": "[]",
        "switching_triggers": "[]",
        "proof_quotes": "[]",
        "objections": "[]",
        "displacement_targets": "[]",
        "segment_match": None,
        "data_coverage": "{}",
        "recommended_approach": None,
        "lead_with": "[]",
        "talking_points": "[]",
        "timing_advice": None,
        "risk_factors": "[]",
        "created_at": datetime(2026, 4, 27, 12, 0, tzinfo=timezone.utc),
    }

    class _Pool:
        async def fetchrow(self, query, *params):
            return row

    monkeypatch.setattr(mod, "get_db_pool", MagicMock(return_value=_Pool()))

    response = await mod.export_prediction_csv(
        str(pid), user=SimpleNamespace(account_id="acct-1")
    )

    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode())
    body = "".join(chunks)
    assert "Win Probability" in body
    assert "Unavailable" in body
