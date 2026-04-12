from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from atlas_brain.api import recognition as mod


@pytest.mark.asyncio
async def test_get_person_rejects_blank_person_id_before_proxy(monkeypatch):
    monkeypatch.setattr(mod, "_proxy_request", AsyncMock(side_effect=AssertionError("proxy touched")))

    with pytest.raises(mod.HTTPException) as exc:
        await mod.get_person("   ")

    assert exc.value.status_code == 422
    assert exc.value.detail == "person_id is required"


@pytest.mark.asyncio
async def test_get_recognition_events_normalizes_query_defaults(monkeypatch):
    proxy = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(mod, "_proxy_request", proxy)

    result = await mod.get_recognition_events()

    assert result == {"ok": True}
    assert proxy.await_args.kwargs["params"] == {"limit": 50}


@pytest.mark.asyncio
async def test_get_recognition_events_trims_active_person_id(monkeypatch):
    proxy = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(mod, "_proxy_request", proxy)

    await mod.get_recognition_events(person_id="  p-123  ", limit=25)

    assert proxy.await_args.kwargs["params"] == {"limit": 25, "person_id": "p-123"}


@pytest.mark.asyncio
async def test_gait_frame_routes_normalize_camera_id(monkeypatch):
    proxy = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(mod, "_proxy_request", proxy)

    await mod.add_gait_frame()
    assert proxy.await_args.kwargs["params"] == {"camera_id": "webcam_office"}

    await mod.add_gait_identify_frame(camera_id="  side-cam  ")
    assert proxy.await_args.kwargs["params"] == {"camera_id": "side-cam"}


@pytest.mark.asyncio
async def test_complete_gait_enrollment_omits_blank_direction(monkeypatch):
    proxy = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(mod, "_proxy_request", proxy)

    await mod.complete_gait_enrollment(walking_direction="   ")

    assert proxy.await_args.kwargs["params"] is None


def test_recognition_request_models_trim_required_text():
    create = mod.CreatePersonRequest(name="  Alice  ")
    enroll = mod.EnrollFaceRequest(person_id="  p-1  ", camera_id="  cam-1  ", source="  manual  ")
    identify = mod.IdentifyRequest(camera_id="  cam-2  ")
    start_gait = mod.StartGaitEnrollRequest(person_id="  p-2  ", camera_id="  cam-3  ")
    gait_identify = mod.GaitIdentifyRequest(camera_id="  cam-4  ")
    combined = mod.CombinedIdentifyRequest(camera_id="  cam-5  ")

    assert create.name == "Alice"
    assert enroll.person_id == "p-1"
    assert enroll.camera_id == "cam-1"
    assert enroll.source == "manual"
    assert identify.camera_id == "cam-2"
    assert start_gait.person_id == "p-2"
    assert start_gait.camera_id == "cam-3"
    assert gait_identify.camera_id == "cam-4"
    assert combined.camera_id == "cam-5"


def test_recognition_request_models_reject_blank_required_text():
    with pytest.raises(ValidationError):
        mod.CreatePersonRequest(name="   ")
    with pytest.raises(ValidationError):
        mod.EnrollFaceRequest(person_id="   ")
    with pytest.raises(ValidationError):
        mod.IdentifyRequest(camera_id="   ")
    with pytest.raises(ValidationError):
        mod.StartGaitEnrollRequest(person_id="   ")
    with pytest.raises(ValidationError):
        mod.GaitIdentifyRequest(camera_id="   ")
    with pytest.raises(ValidationError):
        mod.CombinedIdentifyRequest(camera_id="   ")
    with pytest.raises(ValidationError):
        mod.UpdatePersonRequest(name="   ")
