"""
Device control and management endpoints.
"""

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...capabilities import (
    ActionRequest,
    CapabilityType,
    action_dispatcher,
    capability_registry,
)
from ...capabilities.intent_parser import intent_parser

router = APIRouter()


# --- Request/Response Models ---


class DeviceInfo(BaseModel):
    """Device information response."""
    id: str
    name: str
    type: str
    supported_actions: list[str]
    state: Optional[dict[str, Any]] = None


class ActionRequestBody(BaseModel):
    """Request to execute an action on a device."""
    action: str
    params: dict[str, Any] = {}


class IntentRequestBody(BaseModel):
    """Request to parse and execute a natural language intent."""
    query: str


class ActionResponse(BaseModel):
    """Response from an action execution."""
    success: bool
    message: str
    data: dict[str, Any] = {}
    error: Optional[str] = None


# --- Endpoints ---


@router.get("/", response_model=list[DeviceInfo])
async def list_devices(type: Optional[str] = None):
    """
    List all registered devices/capabilities.

    Optionally filter by device type (light, switch, sensor, etc.)
    """
    if type:
        try:
            cap_type = CapabilityType(type)
            capabilities = capability_registry.list_by_type(cap_type)
        except ValueError:
            raise HTTPException(400, f"Invalid device type: {type}")
    else:
        capabilities = capability_registry.list_all()

    devices = []
    for cap in capabilities:
        try:
            state = await cap.get_state()
            state_dict = state.to_dict() if hasattr(state, "to_dict") else None
        except Exception:
            state_dict = None

        devices.append(DeviceInfo(
            id=cap.id,
            name=cap.name,
            type=cap.capability_type.value,
            supported_actions=cap.supported_actions,
            state=state_dict,
        ))

    return devices


@router.get("/{device_id}", response_model=DeviceInfo)
async def get_device(device_id: str):
    """Get details of a specific device."""
    capability = capability_registry.get(device_id)
    if not capability:
        raise HTTPException(404, f"Device not found: {device_id}")

    try:
        state = await capability.get_state()
        state_dict = state.to_dict() if hasattr(state, "to_dict") else None
    except Exception:
        state_dict = None

    return DeviceInfo(
        id=capability.id,
        name=capability.name,
        type=capability.capability_type.value,
        supported_actions=capability.supported_actions,
        state=state_dict,
    )


@router.get("/{device_id}/state")
async def get_device_state(device_id: str):
    """Get the current state of a device."""
    capability = capability_registry.get(device_id)
    if not capability:
        raise HTTPException(404, f"Device not found: {device_id}")

    try:
        state = await capability.get_state()
        return state.to_dict() if hasattr(state, "to_dict") else {"state": str(state)}
    except Exception as e:
        raise HTTPException(500, f"Failed to get device state: {e}")


@router.post("/{device_id}/action", response_model=ActionResponse)
async def execute_device_action(device_id: str, body: ActionRequestBody):
    """Execute an action on a specific device."""
    request = ActionRequest(
        capability_id=device_id,
        action=body.action,
        params=body.params,
    )

    result = await action_dispatcher.dispatch(request)

    if not result.success:
        raise HTTPException(400, detail=result.message)

    return ActionResponse(
        success=result.success,
        message=result.message,
        data=result.data,
        error=result.error,
    )


@router.post("/intent", response_model=ActionResponse)
async def execute_intent(body: IntentRequestBody):
    """
    Parse a natural language query and execute the inferred action.

    This endpoint bridges VLM understanding with device control.

    Example queries:
    - "turn on the living room lights"
    - "set bedroom brightness to 50%"
    - "turn off all lights"
    """
    intent = await intent_parser.parse(body.query)
    if not intent:
        raise HTTPException(
            400,
            "Could not parse intent from query. Make sure a VLM is loaded.",
        )

    if intent.confidence < 0.5:
        raise HTTPException(
            400,
            f"Low confidence ({intent.confidence:.2f}) in intent parsing. Please be more specific.",
        )

    result = await action_dispatcher.dispatch_intent(intent)

    return ActionResponse(
        success=result.success,
        message=result.message,
        data={
            "intent": intent.model_dump(),
            "result": result.data,
        },
        error=result.error,
    )
