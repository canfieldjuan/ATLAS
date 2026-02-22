"""
Atlas Twilio MCP Server.

Provider-agnostic telephony MCP server backed by Twilio's REST API.
Designed to be used alongside the existing SignalWire infrastructure —
switch ATLAS_COMMS_PROVIDER=twilio to route calls through Twilio instead.

Key capability: outbound call recording.
The legacy TwilioProvider.make_call() omitted the `record` parameter, which
meant outbound calls were never recorded.  This server exposes both:
  - make_call(record=True)  — record from call creation (recommended)
  - start_recording(call_sid) — record mid-call (for calls already in progress)

Tools (10):
    make_call           — initiate an outbound call (with optional recording)
    get_call            — fetch call details / status
    list_calls          — list recent calls with optional status filter
    hangup_call         — end an active call
    start_recording     — start recording an active call (fixes outbound recording)
    stop_recording      — stop a specific recording
    list_recordings     — list recordings for a call
    get_recording       — get recording details and media URL
    send_sms            — send an SMS message
    lookup_phone        — look up caller ID / carrier info for a phone number

Run:
    python -m atlas_brain.mcp.twilio_server          # stdio
    python -m atlas_brain.mcp.twilio_server --sse    # SSE HTTP transport

Configuration (env vars — all prefixed ATLAS_COMMS_):
    ATLAS_COMMS_TWILIO_ACCOUNT_SID   Twilio Account SID (ACxxxxxxxx…)
    ATLAS_COMMS_TWILIO_AUTH_TOKEN    Twilio Auth Token
    ATLAS_COMMS_WEBHOOK_BASE_URL     Public base URL for webhooks
    ATLAS_COMMS_RECORD_CALLS         true/false (global default for recording)
"""

import json
import logging
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("atlas.mcp.twilio")

mcp = FastMCP(
    "atlas-twilio",
    instructions=(
        "Twilio MCP server for Atlas. "
        "Manage outbound/inbound calls, SMS, and recordings via the Twilio REST API. "
        "For outbound call recording issues: use make_call(record=True) at call creation, "
        "or start_recording(call_sid) to begin recording on an already-active call. "
        "Always confirm phone numbers are in E.164 format (+1XXXXXXXXXX) before calling."
    ),
)


def _client():
    """Lazily instantiate the Twilio REST client."""
    try:
        from twilio.rest import Client
        from atlas_comms.core.config import comms_settings

        account_sid = comms_settings.twilio_account_sid
        auth_token = comms_settings.twilio_auth_token

        if not account_sid or not auth_token:
            raise RuntimeError(
                "Twilio credentials not configured. "
                "Set ATLAS_COMMS_TWILIO_ACCOUNT_SID and ATLAS_COMMS_TWILIO_AUTH_TOKEN."
            )
        return Client(account_sid, auth_token)
    except ImportError:
        raise RuntimeError(
            "Twilio package not installed. Run: pip install twilio"
        )


def _comms_settings():
    from atlas_comms.core.config import comms_settings
    return comms_settings


def _e164(number: str) -> str:
    """Strip whitespace/dashes and ensure leading + for E.164."""
    import re
    n = re.sub(r"[\s\-\(\)]", "", number.strip())
    if not n.startswith("+"):
        n = "+" + n
    return n


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
    Initiate an outbound call via Twilio.

    to: Destination phone number in E.164 format (+1XXXXXXXXXX).
    from_number: Caller ID (must be a verified Twilio number).
                 Defaults to ATLAS_COMMS_FORWARD_TO_NUMBER if not supplied.
    record: Record this call (default True — fixes the outbound recording gap).
            When True, a recording-status webhook is sent to Atlas when the
            call ends and the recording becomes available.
    context_id: Optional business context (e.g. 'effingham_maids').

    Returns: call SID and initial status.
    """
    try:
        client = _client()
        cfg = _comms_settings()

        from_num = from_number or cfg.forward_to_number or ""
        if not from_num:
            return json.dumps({"success": False, "error": "from_number is required (or set ATLAS_COMMS_FORWARD_TO_NUMBER)"})

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

        call = client.calls.create(**create_kwargs)

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
    try:
        call = _client().calls(call_sid).fetch()
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
        failed, busy, no-answer — or omit for all.
    limit: Maximum calls to return (default 20, max 100).

    Returns a list of call summaries.
    """
    try:
        kwargs: dict = {"limit": min(limit, 100)}
        if status:
            kwargs["status"] = status

        calls = _client().calls.list(**kwargs)
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
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: hangup_call
# ---------------------------------------------------------------------------

@mcp.tool()
async def hangup_call(call_sid: str) -> str:
    """
    End an active or ringing call.

    call_sid: Twilio Call SID.
    """
    try:
        _client().calls(call_sid).update(status="completed")
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
    recording it — for example, if make_call was called without record=True,
    or to start recording after a specific point in the call.

    call_sid: Twilio Call SID of the active call.
    recording_status_callback: URL to receive recording-complete webhook.
                                Defaults to Atlas's recording-status endpoint.

    Returns: recording SID.

    NOTE: To record outbound calls from the start, prefer make_call(record=True).
    This tool handles mid-call recording or retroactive recording starts.
    """
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

        recording = _client().calls(call_sid).recordings.create(**kwargs)
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
    try:
        _client().calls(call_sid).recordings(recording_sid).update(status="stopped")
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
    try:
        recordings = _client().recordings.list(call_sid=call_sid)
        return json.dumps({
            "success": True,
            "call_sid": call_sid,
            "recordings": [
                {
                    "recording_sid": r.sid,
                    "status": r.status,
                    "duration": r.duration,
                    "date_created": str(r.date_created),
                    "media_url": f"https://api.twilio.com{r.uri.replace('.json', '.mp3')}",
                }
                for r in recordings
            ],
        }, default=str)
    except Exception as exc:
        logger.exception("list_recordings error")
        return json.dumps({"success": False, "error": str(exc)})


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
    try:
        r = _client().recordings(recording_sid).fetch()
        account_sid = _comms_settings().twilio_account_sid
        media_url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{account_sid}/Recordings/{recording_sid}.mp3"
        )
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
    try:
        cfg = _comms_settings()
        from_num = from_number or cfg.forward_to_number or ""
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

        msg = _client().messages.create(**params)
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
    Look up a phone number using Twilio Lookup API.

    Returns: formatted number, country code, carrier name, and caller name
             (if available — requires Lookup add-on).

    phone_number: Number in E.164 or local format.
    """
    try:
        result = _client().lookups.v2.phone_numbers(_e164(phone_number)).fetch(
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
        logger.exception("lookup_phone error")
        return json.dumps({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    transport = "sse" if "--sse" in sys.argv else "stdio"
    if transport == "sse":
        from ..config import settings

        mcp.settings.host = settings.mcp.host
        mcp.settings.port = settings.mcp.twilio_port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
