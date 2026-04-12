from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.api import vision as mod


@pytest.mark.asyncio
async def test_get_vision_events_normalizes_query_defaults(monkeypatch):
    repo = SimpleNamespace(get_recent_events=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_vision_event_repo", lambda: repo)

    result = await mod.get_vision_events()

    assert result == {"count": 0, "events": []}
    assert repo.get_recent_events.await_args.kwargs == {
        "limit": 100,
        "source_id": None,
        "node_id": None,
        "class_name": None,
        "event_type": None,
        "since": None,
    }


@pytest.mark.asyncio
async def test_get_vision_events_trims_active_filters(monkeypatch):
    repo = SimpleNamespace(get_recent_events=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_vision_event_repo", lambda: repo)

    await mod.get_vision_events(source_id="  cam-1  ", node_id="  node-a  ", class_name="  person  ", event_type="  new_track  ")

    kwargs = repo.get_recent_events.await_args.kwargs
    assert kwargs["source_id"] == "cam-1"
    assert kwargs["node_id"] == "node-a"
    assert kwargs["class_name"] == "person"
    assert kwargs["event_type"] == "new_track"


@pytest.mark.asyncio
async def test_create_alert_rule_trims_inputs_before_factory(monkeypatch):
    class Manager:
        def add_rule(self, rule):
            self.rule = rule

    manager = Manager()
    monkeypatch.setattr(mod, "get_alert_manager", lambda: manager)
    monkeypatch.setattr(mod, "create_vision_rule", lambda **kwargs: SimpleNamespace(**kwargs))

    result = await mod.create_alert_rule(
        name="  front-door  ",
        source_pattern="  *front*  ",
        class_name="  person  ",
        detection_type="  new_track  ",
        message_template="  hi  ",
        cooldown_seconds=45,
        priority=7,
    )

    assert manager.rule.name == "front-door"
    assert manager.rule.source_pattern == "*front*"
    assert manager.rule.class_name == "person"
    assert manager.rule.detection_type == "new_track"
    assert manager.rule.message_template == "hi"
    assert result["rule"] == {"name": "front-door", "source_pattern": "*front*", "class_name": "person"}


@pytest.mark.asyncio
async def test_rule_routes_reject_blank_rule_name_before_manager(monkeypatch):
    monkeypatch.setattr(mod, "get_alert_manager", lambda: (_ for _ in ()).throw(AssertionError("manager touched")))

    for fn in (mod.delete_alert_rule, mod.enable_alert_rule, mod.disable_alert_rule):
        with pytest.raises(mod.HTTPException) as exc:
            await fn("   ")
        assert exc.value.status_code == 422
        assert exc.value.detail == "rule_name is required"


@pytest.mark.asyncio
async def test_get_alerts_normalizes_defaults_and_blank_filters(monkeypatch):
    repo = SimpleNamespace(get_recent_alerts=AsyncMock(return_value=[]))
    monkeypatch.setattr(mod, "get_unified_alert_repo", lambda: repo)

    result = await mod.get_alerts(rule_name="   ", source_id="   ")

    assert result == {"count": 0, "alerts": []}
    assert repo.get_recent_alerts.await_args.kwargs == {
        "limit": 50,
        "event_type": "vision",
        "include_acknowledged": False,
        "rule_name": None,
        "source_id": None,
        "since": None,
    }


@pytest.mark.asyncio
async def test_acknowledge_routes_normalize_text_inputs(monkeypatch):
    repo = SimpleNamespace(
        acknowledge_alert=AsyncMock(return_value=True),
        acknowledge_all=AsyncMock(return_value=3),
    )
    monkeypatch.setattr(mod, "get_unified_alert_repo", lambda: repo)

    ok = await mod.acknowledge_alert("00000000-0000-0000-0000-000000000001", acknowledged_by="  ops  ")
    bulk = await mod.acknowledge_all_alerts(acknowledged_by="   ", rule_name="  front-door  ", source_id="   ")

    assert ok == {"success": True, "message": "Alert acknowledged"}
    args = repo.acknowledge_alert.await_args.args
    assert str(args[0]) == "00000000-0000-0000-0000-000000000001"
    assert args[1] == "ops"
    assert bulk == {"success": True, "acknowledged_count": 3}
    assert repo.acknowledge_all.await_args.kwargs == {
        "acknowledged_by": None,
        "event_type": "vision",
        "rule_name": "front-door",
        "source_id": None,
    }
