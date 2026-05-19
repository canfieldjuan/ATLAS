from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_invoicing_draft_writer_live_write.py"


def _load_script_module():
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "check_invoicing_draft_writer_live_write",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "idempotency_key": "atlas-draft-writer-live-smoke-test-v1",
        "customer_name": "ATLAS TEST - DO NOT SEND - Draft Writer Connector",
        "business_context_id": "atlas-mcp-live-smoke",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _create_payload(module):
    return {
        "success": True,
        "created": True,
        "invoice": {
            "id": "invoice-1",
            "invoice_number": "INV-2026-May-0185",
            "status": "draft",
            "customer_name": module.DEFAULT_CUSTOMER_NAME,
            "customer_email": None,
            "business_context_id": module.DEFAULT_BUSINESS_CONTEXT_ID,
            "total_amount": 0.0,
            "metadata": {
                **module.EXPECTED_METADATA,
                "idempotency_key": "atlas-draft-writer-live-smoke-test-v1",
            },
        },
    }


def _get_payload():
    return {
        "found": True,
        "invoice": {
            "invoice_number": "INV-2026-May-0185",
            "status": "draft",
        },
    }


def _pending_payload():
    return {
        "drafts": [
            {
                "invoice_number": "INV-2026-May-0185",
                "blockers": ["no_email"],
                "warnings": ["subtotal_zero", "no_contact_id"],
                "send_safe": False,
            }
        ],
        "summary": {"blocked": 1},
    }


def test_default_idempotency_key_is_daily_and_stable() -> None:
    module = _load_script_module()

    assert module._default_idempotency_key(module.date(2026, 5, 19)) == (
        "atlas-draft-writer-live-smoke-2026-05-19-v1"
    )


def test_main_refuses_to_create_without_explicit_ack_before_config(monkeypatch, capsys) -> None:
    module = _load_script_module()

    def fail_config(_args):
        raise AssertionError("config touched")

    monkeypatch.setattr(module, "_config_from_args", fail_config)

    result = module._main([])

    captured = capsys.readouterr()
    assert result == 2
    assert "--create-blocked-draft" in captured.err


def test_live_config_from_args_requires_smoke_identifiers() -> None:
    module = _load_script_module()

    try:
        module._live_config_from_args(
            _args(idempotency_key=" ", customer_name="", business_context_id=" ")
        )
    except ValueError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError")

    assert "--idempotency-key" in message
    assert "--customer-name" in message
    assert "--business-context-id" in message


def test_create_arguments_build_blocked_zero_dollar_draft() -> None:
    module = _load_script_module()
    config = module.LiveWriteConfig(
        idempotency_key="key-1",
        customer_name="ATLAS TEST",
        business_context_id="ctx-1",
    )

    args = module._create_arguments(config)

    assert args["customer_name"] == "ATLAS TEST"
    assert args["idempotency_key"] == "key-1"
    assert args["business_context_id"] == "ctx-1"
    assert "customer_email" not in args
    assert '"unit_price": 0' in args["line_items"]


def test_pending_drafts_arguments_uses_tool_cap_for_busy_accounts() -> None:
    module = _load_script_module()
    config = module.LiveWriteConfig(
        idempotency_key="key-1",
        customer_name="ATLAS TEST",
        business_context_id="ctx-1",
    )

    args = module._pending_drafts_arguments(config)

    assert args == {
        "business_context_id": "ctx-1",
        "only_blocked": True,
        "limit": 200,
    }


def test_validate_smoke_result_accepts_blocked_draft_contract() -> None:
    module = _load_script_module()
    config = module._live_config_from_args(_args())

    errors = module._validate_smoke_result(
        _create_payload(module),
        _get_payload(),
        _pending_payload(),
        config,
    )

    assert errors == []


def test_validate_smoke_result_rejects_send_safe_or_missing_metadata() -> None:
    module = _load_script_module()
    config = module._live_config_from_args(_args())
    create_payload = _create_payload(module)
    create_payload["invoice"]["metadata"] = {}
    pending_payload = _pending_payload()
    pending_payload["drafts"][0]["blockers"] = []
    pending_payload["drafts"][0]["send_safe"] = True

    errors = module._validate_smoke_result(
        create_payload,
        _get_payload(),
        pending_payload,
        config,
    )

    assert "created invoice metadata.mcp_connector mismatch" in errors
    assert "created invoice metadata.idempotency_key mismatch" in errors
    assert "pending smoke invoice is missing no_email blocker" in errors
    assert "pending smoke invoice is not blocked from sending" in errors


def test_main_reports_success_without_printing_approval_token(monkeypatch, capsys) -> None:
    module = _load_script_module()

    async def fake_run(_oauth_config, _live_config):
        return module.LiveWriteResult(
            created=False,
            invoice_number="INV-2026-May-0185",
            invoice_id="invoice-1",
            blockers=("no_email",),
            warnings=("subtotal_zero",),
        )

    monkeypatch.setattr(module, "_run_live_write_smoke", fake_run)

    result = module._main(
        [
            "--create-blocked-draft",
            "--issuer-url",
            "https://atlas.example.com/invoicing-draft-writer",
            "--resource-url",
            "https://atlas.example.com/invoicing-draft-writer/mcp",
            "--approval-token",
            "secret-approval-token-value",
            "--idempotency-key",
            "atlas-draft-writer-live-smoke-test-v1",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Draft-writer live write smoke completed" in captured.out
    assert "action: reused" in captured.out
    assert "INV-2026-May-0185" in captured.out
    assert "secret-approval-token-value" not in captured.out
