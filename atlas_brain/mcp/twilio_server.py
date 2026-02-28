"""
Atlas Telephony MCP Server (SignalWire / Twilio).

Auto-detects provider from ATLAS_COMMS_PROVIDER (default: signalwire).
SignalWire's Python SDK subclasses Twilio's, so the REST API is nearly
identical -- the only differences are client init, recording media URLs,
and phone lookup.

Key capability: outbound call recording.
  - make_call(record=True)  -- record from call creation (recommended)
  - start_recording(call_sid) -- record mid-call (for calls already in progress)

Tools (10):
    make_call           -- initiate an outbound call (with optional recording)
    get_call            -- fetch call details / status
    list_calls          -- list recent calls with optional status filter
    hangup_call         -- end an active call
    start_recording     -- start recording an active call
    stop_recording      -- stop a specific recording
    list_recordings     -- list recordings for a call
    get_recording       -- get recording details and media URL
    send_sms            -- send an SMS message
    lookup_phone        -- look up carrier / line-type info for a phone number

Run:
    python -m atlas_brain.mcp.twilio_server          # stdio
    python -m atlas_brain.mcp.twilio_server --sse    # SSE HTTP transport

Configuration (env vars -- all prefixed ATLAS_COMMS_):
    # SignalWire (default)
    ATLAS_COMMS_PROVIDER=signalwire
    ATLAS_COMMS_SIGNALWIRE_PROJECT_ID    SignalWire Project ID
    ATLAS_COMMS_SIGNALWIRE_API_TOKEN     SignalWire API Token
    ATLAS_COMMS_SIGNALWIRE_SPACE         SignalWire Space Name (e.g. 'finetunelab')

    # Twilio (alternative)
    ATLAS_COMMS_PROVIDER=twilio
    ATLAS_COMMS_TWILIO_ACCOUNT_SID       Twilio Account SID (ACxxxxxxxx)
    ATLAS_COMMS_TWILIO_AUTH_TOKEN        Twilio Auth Token

    # Common
    ATLAS_COMMS_WEBHOOK_BASE_URL     Public base URL for webhooks
    ATLAS_COMMS_RECORD_CALLS         true/false (global default for recording)
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.twilio")


async def _run_sync(fn, *args, **kwargs) -> Any:
    """Run a synchronous Twilio/SignalWire SDK call in a thread executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


@asynccontextmanager
async def _lifespan(server):
    """Twilio MCP has no DB dependency -- lifespan is a no-op."""
    logger.info("Twilio MCP: started")
    yield


mcp = FastMCP(
    "atlas-twilio",
    instructions=(
        "Telephony MCP server for Atlas (SignalWire / Twilio). "
        "Manage outbound/inbound calls, SMS, and recordings. "
        "For outbound call recording: use make_call(record=True) at call creation, "
        "or start_recording(call_sid) to begin recording on an already-active call. "
        "Always confirm phone numbers are in E.164 format (+1XXXXXXXXXX) before calling."
    ),
    lifespan=_lifespan,
)


def _is_signalwire() -> bool:
    """Check if we're using SignalWire (vs Twilio)."""
    from atlas_comms.core.config import comms_settings
    return comms_settings.provider.lower() == "signalwire"


def _space_url() -> str:
    """Get the SignalWire space base URL."""
    from atlas_comms.core.config import comms_settings
    return f"https://{comms_settings.signalwire_space}.signalwire.com"


_cached_client = None


def _client():
    """Lazily instantiate the telephony REST client (SignalWire or Twilio).

    Caches the client instance to reuse connection pools across tool calls.
    """
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    from atlas_comms.core.config import comms_settings

    if _is_signalwire():
        try:
            from signalwire.rest import Client
        except ImportError:
            raise RuntimeError(
                "SignalWire package not installed. Run: pip install signalwire"
            )

        space = comms_settings.signalwire_space
        # SignalWire's LaML compatibility API authenticates with
        # account_sid + recording_token (NOT project_id + api_token).
        account_sid = comms_settings.signalwire_account_sid
        token = comms_settings.signalwire_recording_token
        if not account_sid or not token:
            # Fallback to project_id + api_token
            account_sid = comms_settings.signalwire_project_id
            token = comms_settings.signalwire_api_token

        if not account_sid or not token or not space:
            raise RuntimeError(
                "SignalWire credentials not configured. "
                "Set ATLAS_COMMS_SIGNALWIRE_ACCOUNT_SID, "
                "ATLAS_COMMS_SIGNALWIRE_RECORDING_TOKEN, and "
                "ATLAS_COMMS_SIGNALWIRE_SPACE."
            )
        _cached_client = Client(
            account_sid, token,
            signalwire_space_url=f"https://{space}.signalwire.com",
        )
        return _cached_client
    else:
        try:
            from twilio.rest import Client
        except ImportError:
            raise RuntimeError(
                "Twilio package not installed. Run: pip install twilio"
            )

        account_sid = comms_settings.twilio_account_sid
        auth_token = comms_settings.twilio_auth_token

        if not account_sid or not auth_token:
            raise RuntimeError(
                "Twilio credentials not configured. "
                "Set ATLAS_COMMS_TWILIO_ACCOUNT_SID and ATLAS_COMMS_TWILIO_AUTH_TOKEN."
            )
        _cached_client = Client(account_sid, auth_token)
        return _cached_client


def _comms_settings():
    from atlas_comms.core.config import comms_settings
    return comms_settings


def _recording_media_url(recording_sid: str, account_sid: str) -> str:
    """Build the correct recording download URL for the active provider."""
    if _is_signalwire():
        space = _comms_settings().signalwire_space
        return (
            f"https://{space}.signalwire.com/api/laml/2010-04-01/"
            f"Accounts/{account_sid}/Recordings/{recording_sid}.mp3"
        )
    return (
        f"https://api.twilio.com/2010-04-01/Accounts/"
        f"{account_sid}/Recordings/{recording_sid}.mp3"
    )


def _account_sid() -> str:
    """Get the account/project SID for the active provider."""
    cfg = _comms_settings()
    if _is_signalwire():
        return cfg.signalwire_account_sid or cfg.signalwire_project_id
    return cfg.twilio_account_sid


def _e164(number: str) -> str:
    """Strip whitespace/dashes and ensure leading + for E.164."""
    import re
    n = re.sub(r"[\s\-\(\)\.]", "", number.strip())
    if not n.startswith("+"):
        n = "+" + n
    return n


def _outbound_caller_id() -> str:
    """Get the business phone number (SignalWire-owned) for outbound caller ID.

    Checks all registered business contexts for phone_numbers.
    This is NOT the user's personal phone --it's the SignalWire number
    customers see when called.
    """
    try:
        from atlas_comms.core.config import EFFINGHAM_MAIDS_CONTEXT
        for number in EFFINGHAM_MAIDS_CONTEXT.phone_numbers:
            cleaned = _e164(number)
            if cleaned.startswith("+"):
                return cleaned
    except Exception as exc:
        logger.warning("Failed to resolve outbound caller ID: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Tool: make_call
# ---------------------------------------------------------------------------

@mcp.tool()
async def make_call(
    to: str,
    from_number: Optional[str] = None,
    record: bool = True,
    context_id: Optional[str] = None,
) -> str:
    """
    Call a customer and bridge them to your phone.

    How it works:
      1. SignalWire calls the customer FROM your business number
      2. Customer answers ->hears "Please hold while we connect you"
      3. Your phone rings (forward_to_number) ->you pick up
      4. Both connected --recording captures the full conversation
      5. When the call ends, the recording is transcribed and processed

    NOTE: AI does NOT talk to the customer. YOU have the conversation.

    to: Customer phone number in E.164 format (+1XXXXXXXXXX).
    from_number: Caller ID --must be a SignalWire-owned number.
                 Auto-detected from the business context phone_numbers.
    record: Record this call (default True). The recording feeds the call
            intelligence pipeline (transcription, extraction, CRM update).
    context_id: Business context (e.g. 'effingham_maids'). Used to pick
                the right caller ID and greeting.

    Returns: call SID and initial status.
    """
    if not to or not to.strip():
        return json.dumps({"success": False, "error": "'to' phone number is required"})

    try:
        client = _client()
        cfg = _comms_settings()

        # Caller ID must be a SignalWire-owned number (NOT your personal phone).
        # Priority: explicit param ->business context phone_numbers ->error.
        from_num = from_number or _outbound_caller_id() or ""
        if not from_num:
            return json.dumps({"success": False, "error": "No outbound caller ID found. Set from_number or configure phone_numbers in a business context."})

        to_e164 = _e164(to)
        from_e164 = _e164(from_num)

        webhook_url = f"{cfg.webhook_base_url}/api/v1/comms/voice/outbound"
        status_url = f"{cfg.webhook_base_url}/api/v1/comms/voice/status"

        create_kwargs: dict = {
            "to": to_e164,
            "from_": from_e164,
            "url": webhook_url,
            "status_callback": status_url,
            "status_callback_event": ["initiated", "ringing", "answered", "completed"],
        }

        should_record = record or cfg.record_calls
        if should_record:
            recording_cb = f"{cfg.webhook_base_url}/api/v1/comms/voice/recording-status"
            create_kwargs["record"] = True
            create_kwargs["recording_status_callback"] = recording_cb
            create_kwargs["recording_status_callback_event"] = "completed"

        call = await _run_sync(client.calls.create, **create_kwargs)

        return json.dumps({
            "success": True,
            "call_sid": call.sid,
            "status": call.status,
            "to": to_e164,
            "from": from_e164,
            "recording_enabled": should_record,
        })
    except Exception as exc:
        logger.exception("make_call error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: get_call
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_call(call_sid: str) -> str:
    """
    Fetch details and current status for a call.

    call_sid: Twilio Call SID (CAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx).

    Returns: from, to, status, direction, duration, start_time, end_time.
    """
    if not call_sid or not call_sid.strip():
        return json.dumps({"success": False, "error": "call_sid is required"})

    try:
        call = await _run_sync(_client().calls(call_sid).fetch)
        return json.dumps({
            "success": True,
            "call_sid": call.sid,
            "from": call.from_formatted,
            "to": call.to_formatted,
            "status": call.status,
            "direction": call.direction,
            "duration": call.duration,
            "start_time": str(call.start_time) if call.start_time else None,
            "end_time": str(call.end_time) if call.end_time else None,
            "answered_by": call.answered_by,
        }, default=str)
    except Exception as exc:
        logger.exception("get_call error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_calls
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_calls(
    status: Optional[str] = None,
    limit: int = 20,
) -> str:
    """
    List recent Twilio calls.

    status: Filter by call status. One of:
        queued, ringing, in-progress, canceled, completed,
        failed, busy, no-answer --or omit for all.
    limit: Maximum calls to return (default 20, max 100).

    Returns a list of call summaries.
    """
    try:
        kwargs: dict = {"limit": min(limit, 100)}
        if status:
            kwargs["status"] = status

        calls = await _run_sync(_client().calls.list, **kwargs)
        return json.dumps({
            "success": True,
            "calls": [
                {
                    "call_sid": c.sid,
                    "from": c.from_formatted,
                    "to": c.to_formatted,
                    "status": c.status,
                    "direction": c.direction,
                    "duration": c.duration,
                    "start_time": str(c.start_time) if c.start_time else None,
                }
                for c in calls
            ],
        }, default=str)
    except Exception as exc:
        logger.exception("list_calls error")
        return json.dumps({"success": False, "error": str(exc), "calls": []})

@mcp.tool()
async def hangup_call(call_sid: str) -> str:
    """
    End an active or ringing call.

    call_sid: Twilio Call SID.
    """
    if not call_sid or not call_sid.strip():
        return json.dumps({"success": False, "error": "call_sid is required"})

    try:
        await _run_sync(_client().calls(call_sid).update, status="completed")
        return json.dumps({"success": True, "call_sid": call_sid, "status": "completed"})
    except Exception as exc:
        logger.exception("hangup_call error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: start_recording  (THE KEY FIX for outbound recording)
# ---------------------------------------------------------------------------

@mcp.tool()
async def start_recording(
    call_sid: str,
    recording_status_callback: Optional[str] = None,
) -> str:
    """
    Start recording an active call.

    Use this when a call is already in progress and you want to begin
    recording it --for example, if make_call was called without record=True,
    or to start recording after a specific point in the call.

    call_sid: Twilio Call SID of the active call.
    recording_status_callback: URL to receive recording-complete webhook.
                                Defaults to Atlas's recording-status endpoint.

    Returns: recording SID.

    NOTE: To record outbound calls from the start, prefer make_call(record=True).
    This tool handles mid-call recording or retroactive recording starts.
    """
    if not call_sid or not call_sid.strip():
        return json.dumps({"success": False, "error": "call_sid is required"})

    try:
        cfg = _comms_settings()
        cb_url = recording_status_callback or (
            f"{cfg.webhook_base_url}/api/v1/comms/voice/recording-status"
            if cfg.webhook_base_url else ""
        )

        kwargs: dict = {}
        if cb_url:
            kwargs["recording_status_callback"] = cb_url
            kwargs["recording_status_callback_event"] = "completed"

        recording = await _run_sync(_client().calls(call_sid).recordings.create, **kwargs)
        return json.dumps({
            "success": True,
            "call_sid": call_sid,
            "recording_sid": recording.sid,
            "status": recording.status,
        })
    except Exception as exc:
        logger.exception("start_recording error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: stop_recording
# ---------------------------------------------------------------------------

@mcp.tool()
async def stop_recording(call_sid: str, recording_sid: str) -> str:
    """
    Stop a specific in-progress recording.

    call_sid: Twilio Call SID.
    recording_sid: Recording SID (RExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx).
    """
    if not call_sid or not call_sid.strip():
        return json.dumps({"success": False, "error": "call_sid is required"})
    if not recording_sid or not recording_sid.strip():
        return json.dumps({"success": False, "error": "recording_sid is required"})

    try:
        await _run_sync(_client().calls(call_sid).recordings(recording_sid).update, status="stopped")
        return json.dumps({
            "success": True,
            "call_sid": call_sid,
            "recording_sid": recording_sid,
            "status": "stopped",
        })
    except Exception as exc:
        logger.exception("stop_recording error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: list_recordings
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_recordings(call_sid: str) -> str:
    """
    List all recordings for a specific call.

    call_sid: Twilio Call SID.

    Returns: list of recordings with SID, status, duration, and media URL.
    """
    if not call_sid or not call_sid.strip():
        return json.dumps({"success": False, "error": "call_sid is required", "recordings": []})

    try:
        recordings = await _run_sync(_client().recordings.list, call_sid=call_sid)
        return json.dumps({
            "success": True,
            "call_sid": call_sid,
            "recordings": [
                {
                    "recording_sid": r.sid,
                    "status": r.status,
                    "duration": r.duration,
                    "date_created": str(r.date_created),
                    "media_url": _recording_media_url(r.sid, _account_sid()),
                }
                for r in recordings
            ],
        }, default=str)
    except Exception as exc:
        logger.exception("list_recordings error")
        return json.dumps({"success": False, "error": str(exc), "recordings": [], "call_sid": call_sid})


# ---------------------------------------------------------------------------
# Tool: get_recording
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_recording(recording_sid: str) -> str:
    """
    Get details and the media URL for a specific recording.

    recording_sid: Recording SID (RExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx).

    Returns: duration, call_sid, status, and a direct MP3 download URL.
    """
    if not recording_sid or not recording_sid.strip():
        return json.dumps({"success": False, "error": "recording_sid is required"})

    try:
        r = await _run_sync(_client().recordings(recording_sid).fetch)
        media_url = _recording_media_url(recording_sid, _account_sid())
        return json.dumps({
            "success": True,
            "recording_sid": r.sid,
            "call_sid": r.call_sid,
            "status": r.status,
            "duration": r.duration,
            "date_created": str(r.date_created),
            "media_url": media_url,
        }, default=str)
    except Exception as exc:
        logger.exception("get_recording error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: send_sms
# ---------------------------------------------------------------------------

@mcp.tool()
async def send_sms(
    to: str,
    body: str,
    from_number: Optional[str] = None,
    media_urls: Optional[str] = None,
) -> str:
    """
    Send an SMS (or MMS) message via Twilio.

    to: Destination phone number in E.164 format.
    body: Message text.
    from_number: Sending number (must be a Twilio number). Defaults to
                 ATLAS_COMMS_FORWARD_TO_NUMBER.
    media_urls: Comma-separated list of public media URLs for MMS (optional).
    """
    if not to or not to.strip():
        return json.dumps({"success": False, "error": "'to' phone number is required"})
    if not body or not body.strip():
        return json.dumps({"success": False, "error": "'body' is required"})

    try:
        cfg = _comms_settings()
        from_num = from_number or _outbound_caller_id() or cfg.forward_to_number or ""
        if not from_num:
            return json.dumps({"success": False, "error": "from_number required (or set ATLAS_COMMS_FORWARD_TO_NUMBER)"})

        params: dict = {
            "to": _e164(to),
            "from_": _e164(from_num),
            "body": body,
        }

        if media_urls:
            params["media_url"] = [u.strip() for u in media_urls.split(",") if u.strip()]

        if cfg.webhook_base_url:
            params["status_callback"] = f"{cfg.webhook_base_url}/api/v1/comms/sms/status"

        msg = await _run_sync(_client().messages.create, **params)

        # Persist outbound SMS (fail-open)
        try:
            from ..storage.repositories.sms_message import get_sms_message_repo
            sms_repo = get_sms_message_repo()
            await sms_repo.create(
                message_sid=msg.sid,
                from_number=msg.from_,
                to_number=msg.to,
                direction="outbound",
                body=body,
                status=msg.status or "queued",
                source="mcp_tool",
            )
        except Exception as persist_err:
            logger.warning("Failed to persist outbound SMS %s: %s", msg.sid, persist_err)

        return json.dumps({
            "success": True,
            "message_sid": msg.sid,
            "status": msg.status,
            "to": msg.to,
            "from": msg.from_,
        })
    except Exception as exc:
        logger.exception("send_sms error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: lookup_phone
# ---------------------------------------------------------------------------

@mcp.tool()
async def lookup_phone(phone_number: str) -> str:
    """
    Look up a phone number for carrier and line-type information.

    Returns: formatted number, line type (mobile/landline/voip), carrier,
             and location when available.

    phone_number: Number in E.164 or local format.
    """
    e164 = _e164(phone_number)

    if _is_signalwire():
        return await _lookup_signalwire(e164)
    return await _run_sync(_lookup_twilio, e164)


async def _lookup_signalwire(e164: str) -> str:
    """SignalWire phone lookup via their REST Lookup API."""
    import base64
    import httpx

    try:
        cfg = _comms_settings()
        space = cfg.signalwire_space
        # Lookup API uses account_sid + recording_token auth
        acct = cfg.signalwire_account_sid or cfg.signalwire_project_id
        token = cfg.signalwire_recording_token or cfg.signalwire_api_token

        auth = base64.b64encode(f"{acct}:{token}".encode()).decode()
        url = (
            f"https://{space}.signalwire.com/api/relay/rest"
            f"/lookup/phone_number/{e164}?include=carrier"
        )

        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(url, headers={"Authorization": f"Basic {auth}"})
            resp.raise_for_status()
            result = resp.json()

        carrier_info = result.get("carrier", {})
        data: dict = {
            "success": True,
            "phone_number": result.get("e164", e164),
            "line_type": result.get("linetype"),
            "location": result.get("location"),
        }
        if carrier_info:
            data["carrier"] = carrier_info.get("lec")
            data["carrier_city"] = carrier_info.get("city")
        return json.dumps(data, default=str)
    except Exception as exc:
        logger.exception("lookup_phone (signalwire) error")
        return json.dumps({"success": False, "error": str(exc)})


def _lookup_twilio(e164: str) -> str:
    """Twilio phone lookup via Lookup v2 API."""
    try:
        result = _client().lookups.v2.phone_numbers(e164).fetch(
            fields="caller_name,line_type_intelligence"
        )
        data: dict = {
            "success": True,
            "phone_number": result.phone_number,
            "country_code": result.country_code,
            "national_number": result.national_number,
        }
        if result.caller_name:
            data["caller_name"] = result.caller_name.get("caller_name")
        if result.line_type_intelligence:
            data["line_type"] = result.line_type_intelligence.get("type")
            data["carrier"] = result.line_type_intelligence.get("mobile_network_code")
        return json.dumps(data, default=str)
    except Exception as exc:
        logger.exception("lookup_phone (twilio) error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings
        from .auth import run_sse_with_auth

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.twilio_port
        run_sse_with_auth(mcp, settings.mcp.host, settings.mcp.twilio_port)
    else:
        mcp.run(transport="stdio")
