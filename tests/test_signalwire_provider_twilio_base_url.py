"""Regression test for the twilio/signalwire decouple.

The `signalwire` SDK was dropped (it froze twilio at 6.54.0); the SignalWire
provider now uses the plain `twilio` SDK pointed at the space's
twilio-compatible LaML endpoint. See
plans/PR-Twilio-SignalWire-Decouple.md.
"""
import pytest

from atlas_comms.core.config import comms_settings
from atlas_comms.providers.signalwire import SignalWireProvider


async def test_connect_points_twilio_client_at_signalwire_space(monkeypatch):
    monkeypatch.setattr(comms_settings, "signalwire_project_id", "PROJECTID", raising=False)
    monkeypatch.setattr(comms_settings, "signalwire_api_token", "TOKEN", raising=False)
    monkeypatch.setattr(comms_settings, "signalwire_space", "myspace", raising=False)

    provider = SignalWireProvider()
    await provider.connect()

    assert provider.is_connected
    # The twilio client's Api domain is pointed at the SignalWire space,
    # not api.twilio.com -- this is what the removed signalwire SDK did.
    assert provider._client.api.base_url == "https://myspace.signalwire.com"

    # messages/calls resolve to SignalWire's twilio-compatible LaML endpoint.
    msgs_url = provider._client.messages._version.absolute_url(
        "Accounts/PROJECTID/Messages.json"
    )
    calls_url = provider._client.calls._version.absolute_url(
        "Accounts/PROJECTID/Calls.json"
    )
    assert msgs_url == (
        "https://myspace.signalwire.com/2010-04-01/Accounts/PROJECTID/Messages.json"
    )
    assert calls_url == (
        "https://myspace.signalwire.com/2010-04-01/Accounts/PROJECTID/Calls.json"
    )


async def test_connect_requires_credentials(monkeypatch):
    monkeypatch.setattr(comms_settings, "signalwire_project_id", "", raising=False)
    monkeypatch.setattr(comms_settings, "signalwire_api_token", "", raising=False)
    monkeypatch.setattr(comms_settings, "signalwire_space", "", raising=False)

    provider = SignalWireProvider()
    with pytest.raises(ValueError):
        await provider.connect()
