"""Unit coverage for the EOM estimate-booking rollback drain CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "drain_eom_estimate_bookings_for_rollback.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "drain_eom_estimate_bookings_for_rollback",
    _SCRIPT,
)
assert _SPEC and _SPEC.loader
drain_cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drain_cli)


def test_build_parser_defaults_to_bounded_drain_limit() -> None:
    args = drain_cli.build_parser().parse_args([])

    assert args.limit == 100


@pytest.mark.asyncio
async def test_run_rejects_non_positive_limit_before_pool_lookup() -> None:
    args = SimpleNamespace(limit=0)

    def fail_if_pool_requested():
        raise AssertionError("invalid arguments must fail before DB lookup")

    with pytest.raises(SystemExit, match="--limit must be positive"):
        await drain_cli._run(args, pool_provider=fail_if_pool_requested)


@pytest.mark.asyncio
async def test_run_closes_pool_after_success(capsys) -> None:
    class _Pool:
        initialized = False
        closed = False

        async def initialize(self) -> None:
            self.initialized = True

        async def close(self) -> None:
            self.closed = True

    class _Service:
        def __init__(self, *, pool, config) -> None:
            self.pool = pool
            self.config = config

        async def drain_unfinished_for_rollback(self, *, limit: int) -> dict[str, object]:
            assert limit == 2
            return {"ok": True, "remaining": 0, "failures": []}

    pool = _Pool()
    config = object()
    args = SimpleNamespace(limit=2)

    code = await drain_cli._run(
        args,
        pool_provider=lambda: pool,
        service_factory=_Service,
        config=config,
    )

    assert code == 0
    assert pool.initialized is True
    assert pool.closed is True
    output = capsys.readouterr().out
    assert '"ok": true' in output
    assert "rollback drain complete" in output


@pytest.mark.asyncio
async def test_run_returns_failure_when_operations_remain(capsys) -> None:
    class _Pool:
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class _Service:
        def __init__(self, *, pool, config) -> None:
            self.pool = pool
            self.config = config

        async def drain_unfinished_for_rollback(self, *, limit: int) -> dict[str, object]:
            assert limit == 1
            return {
                "ok": False,
                "remaining": 1,
                "failures": [{"operation_id": "op-1", "status": "projecting"}],
            }

    code = await drain_cli._run(
        SimpleNamespace(limit=1),
        pool_provider=_Pool,
        service_factory=_Service,
        config=object(),
    )

    assert code == 1
    assert "rollback unsafe" in capsys.readouterr().out
