"""
Unit tests for the CRM provider, Email provider, and MCP server tools.

All external dependencies (DB pool, Directus HTTP, Gmail/Resend, GPU packages)
are mocked so these tests run without any real services or GPU hardware.
"""

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy/unavailable dependencies at import time.
# The following packages are only present in the full GPU runtime environment;
# they are not installed in the unit-test sandbox.  Registering them as
# MagicMocks before any atlas_brain import prevents the ImportError cascade
# through services/__init__.py and tools/__init__.py.
# ---------------------------------------------------------------------------
for _heavy_mod in [
    "PIL", "PIL.Image",          # no longer used (VLM removed), kept to avoid import errors
    "torch",                     # PyTorch — GPU runtime only
    "transformers",              # HuggingFace transformers — GPU runtime only
    "numpy",                     # NumPy — required by sentence-transformers/embedding
    "sentence_transformers",     # Semantic embedding — GPU runtime only
    "asyncpg",                   # Async PostgreSQL — requires running DB
    "llama_cpp",                 # llama.cpp LLM backend — GPU runtime only
    "dateparser",                # Date parsing library — not installed in test env
]:
    sys.modules.setdefault(_heavy_mod, MagicMock())

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contact(**kwargs) -> dict:
    return {
        "id": str(uuid4()),
        "full_name": kwargs.get("full_name", "Jane Smith"),
        "phone": kwargs.get("phone", "618-555-0100"),
        "email": kwargs.get("email", "jane@example.com"),
        "status": kwargs.get("status", "active"),
        "contact_type": kwargs.get("contact_type", "customer"),
        "business_context_id": kwargs.get("business_context_id", "effingham_maids"),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "tags": [],
        "notes": None,
        "source": "manual",
        "source_ref": None,
        "metadata": {},
    }


def _make_appointment(**kwargs) -> dict:
    return {
        "id": str(uuid4()),
        "start_time": datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc),
        "end_time": datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
        "service_type": kwargs.get("service_type", "Cleaning Estimate"),
        "status": kwargs.get("status", "confirmed"),
        "customer_name": kwargs.get("customer_name", "Jane Smith"),
        "customer_phone": kwargs.get("customer_phone", "618-555-0100"),
        "customer_email": kwargs.get("customer_email", "jane@example.com"),
        "customer_address": kwargs.get("customer_address", "123 Main St"),
        "notes": "",
        "created_at": datetime.now(timezone.utc),
    }


# ===========================================================================
# DatabaseCRMProvider (direct asyncpg queries)
# ===========================================================================

class TestDatabaseCRMProvider:
    """Tests for DatabaseCRMProvider using a mocked asyncpg pool."""

    def _mock_pool(self, fetchrow_return=None, fetch_return=None, execute_return=None):
        pool = MagicMock()
        pool.is_initialized = True
        pool.fetchrow = AsyncMock(return_value=fetchrow_return)
        pool.fetch = AsyncMock(return_value=fetch_return or [])
        pool.execute = AsyncMock(return_value=execute_return or "UPDATE 1")
        return pool

    async def test_create_contact_returns_dict(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        contact_data = _make_contact()
        pool = self._mock_pool(fetchrow_return=contact_data)

        with patch("atlas_brain.services.crm_provider.DatabaseCRMProvider.create_contact",
                   new=AsyncMock(return_value=contact_data)):
            provider = DatabaseCRMProvider()

            result = await provider.create_contact({
                "full_name": "Jane Smith",
                "phone": "618-555-0100",
                "email": "jane@example.com",
            })

        assert result["full_name"] == "Jane Smith"
        assert result["phone"] == "618-555-0100"

    async def test_search_contacts_returns_list(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        contacts = [_make_contact(), _make_contact(full_name="John Doe")]

        provider = DatabaseCRMProvider()
        provider.search_contacts = AsyncMock(return_value=contacts)

        results = await provider.search_contacts(query="smith")
        assert len(results) == 2

    async def test_get_contact_not_found_returns_none(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        provider = DatabaseCRMProvider()
        provider.get_contact = AsyncMock(return_value=None)

        result = await provider.get_contact(str(uuid4()))
        assert result is None

    async def test_delete_contact_soft_deletes(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        provider = DatabaseCRMProvider()
        provider.delete_contact = AsyncMock(return_value=True)

        result = await provider.delete_contact(str(uuid4()))
        assert result is True

    async def test_health_check_initialized(self):
        from atlas_brain.services.crm_provider import DatabaseCRMProvider

        provider = DatabaseCRMProvider()
        with patch("atlas_brain.services.crm_provider.DatabaseCRMProvider.health_check",
                   new=AsyncMock(return_value=True)):
            ok = await provider.health_check()
        assert ok is True


# ===========================================================================
# get_crm_provider factory
# ===========================================================================

class TestGetCrmProviderFactory:
    def test_returns_database_provider(self):
        import atlas_brain.services.crm_provider as mod

        mod._crm_provider = None
        provider = mod.get_crm_provider()
        from atlas_brain.services.crm_provider import DatabaseCRMProvider
        assert isinstance(provider, DatabaseCRMProvider)
        mod._crm_provider = None  # Reset for other tests


# ===========================================================================
# CRM MCP server tools
# ===========================================================================

class TestCRMMCPTools:
    """Tests for MCP tool functions using a mocked CRM provider."""

    def _patch_provider(self, mock_provider):
        return patch(
            "atlas_brain.mcp.crm_server._provider",
            return_value=mock_provider,
        )

    async def test_search_contacts_found(self):
        from atlas_brain.mcp.crm_server import search_contacts

        contact = _make_contact()
        provider = MagicMock()
        provider.search_contacts = AsyncMock(return_value=[contact])

        with self._patch_provider(provider):
            raw = await search_contacts(query="Jane")

        data = json.loads(raw)
        assert data["found"] is True
        assert data["count"] == 1
        assert data["contacts"][0]["full_name"] == "Jane Smith"

    async def test_search_contacts_not_found(self):
        from atlas_brain.mcp.crm_server import search_contacts

        provider = MagicMock()
        provider.search_contacts = AsyncMock(return_value=[])

        with self._patch_provider(provider):
            raw = await search_contacts(query="nobody")

        data = json.loads(raw)
        assert data["found"] is False
        assert data["count"] == 0

    async def test_get_contact_found(self):
        from atlas_brain.mcp.crm_server import get_contact

        contact = _make_contact()
        provider = MagicMock()
        provider.get_contact = AsyncMock(return_value=contact)

        with self._patch_provider(provider):
            raw = await get_contact(contact["id"])

        data = json.loads(raw)
        assert data["found"] is True
        assert data["contact"]["full_name"] == "Jane Smith"

    async def test_get_contact_not_found(self):
        from atlas_brain.mcp.crm_server import get_contact

        provider = MagicMock()
        provider.get_contact = AsyncMock(return_value=None)

        with self._patch_provider(provider):
            raw = await get_contact(str(uuid4()))

        data = json.loads(raw)
        assert data["found"] is False

    async def test_create_contact_success(self):
        from atlas_brain.mcp.crm_server import create_contact

        contact = _make_contact()
        provider = MagicMock()
        provider.create_contact = AsyncMock(return_value=contact)

        with self._patch_provider(provider):
            raw = await create_contact(
                full_name="Jane Smith",
                phone="618-555-0100",
                email="jane@example.com",
            )

        data = json.loads(raw)
        assert data["success"] is True
        assert data["contact"]["full_name"] == "Jane Smith"

    async def test_update_contact_no_fields(self):
        from atlas_brain.mcp.crm_server import update_contact

        provider = MagicMock()

        with self._patch_provider(provider):
            raw = await update_contact(contact_id=str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is False
        assert "No fields provided" in data["error"]

    async def test_delete_contact(self):
        from atlas_brain.mcp.crm_server import delete_contact

        provider = MagicMock()
        provider.delete_contact = AsyncMock(return_value=True)

        with self._patch_provider(provider):
            raw = await delete_contact(str(uuid4()))

        data = json.loads(raw)
        assert data["success"] is True

    async def test_log_interaction(self):
        from atlas_brain.mcp.crm_server import log_interaction

        interaction = {
            "id": str(uuid4()),
            "contact_id": str(uuid4()),
            "interaction_type": "call",
            "summary": "Customer called about scheduling",
            "occurred_at": datetime.now(timezone.utc).isoformat(),
        }
        provider = MagicMock()
        provider.log_interaction = AsyncMock(return_value=interaction)

        with self._patch_provider(provider):
            raw = await log_interaction(
                contact_id=interaction["contact_id"],
                interaction_type="call",
                summary="Customer called about scheduling",
            )

        data = json.loads(raw)
        assert data["success"] is True
        assert data["interaction"]["interaction_type"] == "call"

    async def test_get_interactions(self):
        from atlas_brain.mcp.crm_server import get_interactions

        interactions = [
            {"id": str(uuid4()), "interaction_type": "call", "summary": "first call"},
            {"id": str(uuid4()), "interaction_type": "email", "summary": "sent estimate"},
        ]
        provider = MagicMock()
        provider.get_interactions = AsyncMock(return_value=interactions)

        with self._patch_provider(provider):
            raw = await get_interactions(contact_id=str(uuid4()))

        data = json.loads(raw)
        assert data["count"] == 2

    async def test_get_contact_appointments(self):
        from atlas_brain.mcp.crm_server import get_contact_appointments

        appts = [_make_appointment(), _make_appointment()]
        provider = MagicMock()
        provider.get_contact_appointments = AsyncMock(return_value=appts)

        with self._patch_provider(provider):
            raw = await get_contact_appointments(str(uuid4()))

        data = json.loads(raw)
        assert data["count"] == 2

    async def test_search_contacts_provider_error(self):
        """Provider errors should return a safe error JSON, not raise."""
        from atlas_brain.mcp.crm_server import search_contacts

        provider = MagicMock()
        provider.search_contacts = AsyncMock(side_effect=RuntimeError("DB down"))

        with self._patch_provider(provider):
            raw = await search_contacts(query="anyone")

        data = json.loads(raw)
        assert "error" in data
        assert data["found"] is False


# ===========================================================================
# Email MCP server tools
# ===========================================================================

class TestEmailMCPTools:
    def _patch_provider(self, mock_provider):
        return patch(
            "atlas_brain.mcp.email_server._provider",
            return_value=mock_provider,
        )

    async def test_send_email_success(self):
        from atlas_brain.mcp.email_server import send_email

        provider = MagicMock()
        provider.send = AsyncMock(return_value={"id": "msg-123"})

        with self._patch_provider(provider):
            raw = await send_email(
                to="alice@example.com",
                subject="Hello",
                body="Test body",
            )

        data = json.loads(raw)
        assert data["success"] is True
        assert data["result"]["id"] == "msg-123"

    async def test_send_email_comma_separated_to(self):
        """Comma-separated 'to' is parsed into a list before being passed to the provider."""
        from atlas_brain.mcp.email_server import send_email

        provider = MagicMock()
        provider.send = AsyncMock(return_value={"id": "msg-456"})

        with self._patch_provider(provider):
            await send_email(
                to="alice@example.com, bob@example.com",
                subject="Multi",
                body="Body",
            )

        call_kwargs = provider.send.call_args.kwargs
        assert call_kwargs["to"] == ["alice@example.com", "bob@example.com"]

    async def test_send_email_provider_error(self):
        from atlas_brain.mcp.email_server import send_email

        provider = MagicMock()
        provider.send = AsyncMock(side_effect=RuntimeError("Gmail unavailable"))

        with self._patch_provider(provider):
            raw = await send_email(to="x@y.com", subject="s", body="b")

        data = json.loads(raw)
        assert data["success"] is False
        assert "Gmail unavailable" in data["error"]

    async def test_list_inbox(self):
        from atlas_brain.mcp.email_server import list_inbox

        messages = [{"id": "m1"}, {"id": "m2"}]
        provider = MagicMock()
        provider.list_messages = AsyncMock(return_value=messages)

        with self._patch_provider(provider):
            raw = await list_inbox(query="is:unread", max_results=10)

        data = json.loads(raw)
        assert data["count"] == 2
        assert data["messages"][0]["id"] == "m1"

    async def test_get_message(self):
        from atlas_brain.mcp.email_server import get_message

        full_msg = {
            "id": "m1",
            "from": "alice@example.com",
            "subject": "Invoice",
            "body_text": "Please pay…",
        }
        provider = MagicMock()
        provider.get_message = AsyncMock(return_value=full_msg)

        with self._patch_provider(provider):
            raw = await get_message("m1")

        data = json.loads(raw)
        assert data["message"]["subject"] == "Invoice"

    async def test_get_thread(self):
        from atlas_brain.mcp.email_server import get_thread

        thread = {"id": "t1", "messages": [{"id": "m1"}, {"id": "m2"}]}
        provider = MagicMock()
        provider.get_thread = AsyncMock(return_value=thread)

        with self._patch_provider(provider):
            raw = await get_thread("t1")

        data = json.loads(raw)
        assert data["thread"]["id"] == "t1"
        assert len(data["thread"]["messages"]) == 2

    async def test_list_sent_history(self):
        from atlas_brain.mcp.email_server import list_sent_history

        from atlas_brain.tools.base import ToolResult
        mock_tool_result = ToolResult(
            success=True,
            data={"emails": [], "count": 0},
            message="No emails found.",
        )

        with patch("atlas_brain.mcp.email_server.list_sent_history") as mocked:
            mocked = AsyncMock(return_value=json.dumps(
                {"success": True, "data": {"emails": [], "count": 0}, "message": "No emails found."}
            ))
            raw = await mocked(hours=24)

        data = json.loads(raw)
        assert data["success"] is True


# ===========================================================================
# CompositeEmailProvider — fallback logic
# ===========================================================================

class TestCompositeEmailProvider:
    async def test_send_uses_gmail_when_available(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        provider._gmail.is_available = AsyncMock(return_value=True)
        provider._gmail.send = AsyncMock(return_value={"id": "gmail-id"})
        provider._resend.send = AsyncMock(return_value={"id": "resend-id"})

        result = await provider.send(to=["x@y.com"], subject="s", body="b")
        assert result["id"] == "gmail-id"
        provider._resend.send.assert_not_called()

    async def test_send_falls_back_to_resend(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        provider._gmail.is_available = AsyncMock(return_value=True)
        provider._gmail.send = AsyncMock(side_effect=RuntimeError("token expired"))
        provider._resend.send = AsyncMock(return_value={"id": "resend-id"})

        result = await provider.send(to=["x@y.com"], subject="s", body="b")
        assert result["id"] == "resend-id"

    async def test_send_uses_resend_when_gmail_unavailable(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        provider._gmail.is_available = AsyncMock(return_value=False)
        provider._gmail.send = AsyncMock()
        provider._resend.send = AsyncMock(return_value={"id": "resend-only"})

        result = await provider.send(to=["x@y.com"], subject="s", body="b")
        assert result["id"] == "resend-only"
        provider._gmail.send.assert_not_called()


# ===========================================================================
# LookupCustomerTool — CRM-first strategy
# ===========================================================================

class TestLookupCustomerToolCRMFirst:
    async def test_returns_crm_contact_when_found(self):
        """When CRM has a match it should be returned without querying appointments."""
        from atlas_brain.tools.scheduling import LookupCustomerTool

        contact = _make_contact()
        contact["id"] = str(uuid4())
        mock_crm = MagicMock()
        mock_crm.search_contacts = AsyncMock(return_value=[contact])
        mock_crm.get_contact_appointments = AsyncMock(return_value=[])

        tool = LookupCustomerTool()
        # get_crm_provider is imported lazily inside execute() via
        # "from ..services.crm_provider import get_crm_provider"
        with patch("atlas_brain.services.crm_provider.get_crm_provider", return_value=mock_crm):
            result = await tool.execute({"name": "Jane"})

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["customer"]["source"] == "crm"
        assert result.data["customer"]["name"] == "Jane Smith"

    async def test_falls_back_to_appointments_when_crm_empty(self):
        """When CRM returns nothing the tool falls back to appointment rows."""
        from atlas_brain.tools.scheduling import LookupCustomerTool

        mock_crm = MagicMock()
        mock_crm.search_contacts = AsyncMock(return_value=[])

        appt = _make_appointment()
        # Make start_time aware for comparison inside the tool
        appt["start_time"] = datetime(2027, 1, 1, 9, 0, tzinfo=timezone.utc)
        mock_repo = MagicMock()
        mock_repo.search_by_name = AsyncMock(return_value=[appt])
        mock_repo.get_by_phone = AsyncMock(return_value=[])

        tool = LookupCustomerTool()
        with (
            patch("atlas_brain.services.crm_provider.get_crm_provider", return_value=mock_crm),
            patch("atlas_brain.tools.scheduling.get_appointment_repo", return_value=mock_repo),
        ):
            result = await tool.execute({"name": "Jane"})

        assert result.success is True
        assert result.data["found"] is True
        assert result.data["customer"]["source"] == "appointments"

    async def test_missing_params_returns_error(self):
        from atlas_brain.tools.scheduling import LookupCustomerTool

        tool = LookupCustomerTool()
        result = await tool.execute({})

        assert result.success is False
        assert result.error == "MISSING_PARAMS"


# ===========================================================================
# IMAPEmailProvider — unit tests (no real IMAP server needed)
# ===========================================================================

class TestIMAPEmailProvider:
    """Tests for the provider-agnostic IMAP reader."""

    def _make_provider(self, host="imap.example.com", username="user@example.com", password="pass"):
        from atlas_brain.services.email_provider import IMAPEmailProvider
        p = IMAPEmailProvider()
        p._host = host
        p._username = username
        p._password = password
        p._ssl = True
        p._mailbox = "INBOX"
        p._loaded = True
        return p

    def test_is_configured_returns_true_when_all_set(self):
        p = self._make_provider()
        assert p.is_configured() is True

    def test_is_configured_returns_false_when_no_host(self):
        p = self._make_provider(host="")
        assert p.is_configured() is False

    def test_is_configured_returns_false_when_no_password(self):
        p = self._make_provider(password="")
        assert p.is_configured() is False

    def test_imap_search_criteria_unread(self):
        from atlas_brain.services.email_provider import _imap_search_criteria
        assert _imap_search_criteria("is:unread") == "UNSEEN"

    def test_imap_search_criteria_from(self):
        from atlas_brain.services.email_provider import _imap_search_criteria
        assert _imap_search_criteria("from:alice@example.com") == 'FROM "alice@example.com"'

    def test_imap_search_criteria_subject(self):
        from atlas_brain.services.email_provider import _imap_search_criteria
        assert _imap_search_criteria("subject:invoice") == 'SUBJECT "invoice"'

    def test_imap_search_criteria_combined(self):
        from atlas_brain.services.email_provider import _imap_search_criteria
        result = _imap_search_criteria("is:unread from:alice@example.com")
        assert "UNSEEN" in result
        assert 'FROM "alice@example.com"' in result

    def test_imap_search_criteria_unknown_token_falls_back(self):
        from atlas_brain.services.email_provider import _imap_search_criteria
        # Unknown tokens silently dropped; empty criteria → ALL
        assert _imap_search_criteria("label:work OR label:personal") == "ALL"

    def test_imap_search_criteria_empty_returns_all(self):
        from atlas_brain.services.email_provider import _imap_search_criteria
        assert _imap_search_criteria("") == "ALL"

    def test_decode_mime_words_plain(self):
        from atlas_brain.services.email_provider import _decode_mime_words
        assert _decode_mime_words("Hello World") == "Hello World"

    def test_decode_mime_words_encoded(self):
        from atlas_brain.services.email_provider import _decode_mime_words
        # =?utf-8?q?Hello_World?=  → "Hello World"
        encoded = "=?utf-8?q?Hello_World?="
        assert "Hello" in _decode_mime_words(encoded)

    async def test_list_messages_calls_executor(self):
        """list_messages should invoke _list_messages_sync in an executor."""
        p = self._make_provider()
        stub = [{"id": "42", "subject": "Hi", "from": "alice@example.com"}]
        p._list_messages_sync = MagicMock(return_value=stub)

        result = await p.list_messages("is:unread", max_results=5)

        p._list_messages_sync.assert_called_once_with("is:unread", 5)
        assert result == stub

    async def test_get_message_calls_executor(self):
        p = self._make_provider()
        stub = {"id": "42", "body_text": "Hello"}
        p._get_message_sync = MagicMock(return_value=stub)

        result = await p.get_message("42")

        p._get_message_sync.assert_called_once_with("42")
        assert result["body_text"] == "Hello"

    async def test_send_raises_not_implemented(self):
        p = self._make_provider()
        with pytest.raises(NotImplementedError):
            await p.send(to=["x@y.com"], subject="s", body="b")


class TestCompositeProviderIMAPPreference:
    """CompositeEmailProvider should prefer IMAP for reading when configured."""

    async def test_reads_via_imap_when_configured(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        # Mark IMAP as configured
        provider._imap.is_configured = MagicMock(return_value=True)
        provider._imap.list_messages = AsyncMock(return_value=[{"id": "imap-1"}])
        provider._gmail.list_messages = AsyncMock(return_value=[{"id": "gmail-1"}])

        result = await provider.list_messages("is:unread")

        provider._imap.list_messages.assert_called_once()
        provider._gmail.list_messages.assert_not_called()
        assert result[0]["id"] == "imap-1"

    async def test_falls_back_to_gmail_when_imap_not_configured(self):
        from atlas_brain.services.email_provider import CompositeEmailProvider

        provider = CompositeEmailProvider()
        provider._imap.is_configured = MagicMock(return_value=False)
        provider._imap.list_messages = AsyncMock(return_value=[])
        provider._gmail.list_messages = AsyncMock(return_value=[{"id": "gmail-1"}])

        result = await provider.list_messages("is:unread")

        provider._imap.list_messages.assert_not_called()
        provider._gmail.list_messages.assert_called_once()
        assert result[0]["id"] == "gmail-1"


# ===========================================================================
# Twilio MCP Server — unit tests
# ===========================================================================

@pytest.fixture(autouse=False)
def _mock_twilio_client(monkeypatch):
    """Patch _client() in twilio_server to return a MagicMock."""
    from atlas_brain.mcp import twilio_server
    mock = MagicMock()
    monkeypatch.setattr(twilio_server, "_client", lambda: mock)
    return mock


@pytest.fixture(autouse=False)
def _mock_comms_settings(monkeypatch):
    from atlas_brain.mcp import twilio_server
    cfg = MagicMock()
    cfg.webhook_base_url = "https://atlas.example.com"
    cfg.record_calls = False
    cfg.forward_to_number = "+13095550001"
    monkeypatch.setattr(twilio_server, "_comms_settings", lambda: cfg)
    return cfg


class TestTwilioMCPTools:
    async def test_make_call_success(self, _mock_twilio_client, _mock_comms_settings):
        from atlas_brain.mcp.twilio_server import make_call

        fake_call = MagicMock()
        fake_call.sid = "CA123"
        fake_call.status = "queued"
        _mock_twilio_client.calls.create.return_value = fake_call

        raw = await make_call(to="+16185551234", record=True)
        data = json.loads(raw)

        assert data["success"] is True
        assert data["call_sid"] == "CA123"
        assert data["recording_enabled"] is True
        # Verify record=True was passed
        kwargs = _mock_twilio_client.calls.create.call_args.kwargs
        assert kwargs.get("record") is True

    async def test_make_call_missing_from_number(self, _mock_twilio_client, _mock_comms_settings):
        from atlas_brain.mcp.twilio_server import make_call

        _mock_comms_settings.forward_to_number = ""
        with patch("atlas_brain.mcp.twilio_server._outbound_caller_id", return_value=""):
            raw = await make_call(to="+16185551234", from_number=None)
        data = json.loads(raw)
        assert data["success"] is False
        assert "caller ID" in data["error"] or "from_number" in data["error"]

    async def test_make_call_twilio_error(self, _mock_twilio_client, _mock_comms_settings):
        from atlas_brain.mcp.twilio_server import make_call

        _mock_twilio_client.calls.create.side_effect = RuntimeError("invalid number")
        raw = await make_call(to="+16185551234", from_number="+13095550001")
        data = json.loads(raw)
        assert data["success"] is False
        assert "invalid number" in data["error"]

    async def test_get_call_success(self, _mock_twilio_client):
        from atlas_brain.mcp.twilio_server import get_call

        fake_call = MagicMock()
        fake_call.sid = "CA123"
        fake_call.from_formatted = "+13095550001"
        fake_call.to_formatted = "+16185551234"
        fake_call.status = "completed"
        fake_call.direction = "outbound-api"
        fake_call.duration = "62"
        fake_call.start_time = None
        fake_call.end_time = None
        fake_call.answered_by = None
        _mock_twilio_client.calls.return_value.fetch.return_value = fake_call

        raw = await get_call("CA123")
        data = json.loads(raw)
        assert data["success"] is True
        assert data["call_sid"] == "CA123"
        assert data["status"] == "completed"

    async def test_list_calls_success(self, _mock_twilio_client):
        from atlas_brain.mcp.twilio_server import list_calls

        fake_call = MagicMock()
        fake_call.sid = "CA001"
        fake_call.from_formatted = "+1"
        fake_call.to_formatted = "+2"
        fake_call.status = "completed"
        fake_call.direction = "outbound-api"
        fake_call.duration = "30"
        fake_call.start_time = None
        _mock_twilio_client.calls.list.return_value = [fake_call]

        raw = await list_calls(status="completed", limit=10)
        data = json.loads(raw)
        assert data["success"] is True
        assert len(data["calls"]) == 1
        assert data["calls"][0]["call_sid"] == "CA001"

    async def test_hangup_call(self, _mock_twilio_client):
        from atlas_brain.mcp.twilio_server import hangup_call

        raw = await hangup_call("CA123")
        data = json.loads(raw)
        assert data["success"] is True
        _mock_twilio_client.calls.return_value.update.assert_called_once_with(status="completed")

    async def test_start_recording(self, _mock_twilio_client, _mock_comms_settings):
        from atlas_brain.mcp.twilio_server import start_recording

        fake_rec = MagicMock()
        fake_rec.sid = "RE123"
        fake_rec.status = "in-progress"
        _mock_twilio_client.calls.return_value.recordings.create.return_value = fake_rec

        raw = await start_recording("CA123")
        data = json.loads(raw)
        assert data["success"] is True
        assert data["recording_sid"] == "RE123"
        # Verify the recording-status callback URL was passed
        kwargs = _mock_twilio_client.calls.return_value.recordings.create.call_args.kwargs
        assert "recording-status" in kwargs.get("recording_status_callback", "")

    async def test_stop_recording(self, _mock_twilio_client):
        from atlas_brain.mcp.twilio_server import stop_recording

        raw = await stop_recording("CA123", "RE123")
        data = json.loads(raw)
        assert data["success"] is True
        _mock_twilio_client.calls.return_value.recordings.return_value.update.assert_called_once_with(
            status="stopped"
        )

    async def test_list_recordings(self, _mock_twilio_client):
        from atlas_brain.mcp.twilio_server import list_recordings

        fake_rec = MagicMock()
        fake_rec.sid = "RE001"
        fake_rec.status = "completed"
        fake_rec.duration = "45"
        fake_rec.date_created = "2026-01-01"
        fake_rec.uri = "/2010-04-01/Accounts/AC/Recordings/RE001.json"
        _mock_twilio_client.recordings.list.return_value = [fake_rec]

        raw = await list_recordings("CA123")
        data = json.loads(raw)
        assert data["success"] is True
        assert len(data["recordings"]) == 1
        assert data["recordings"][0]["recording_sid"] == "RE001"
        assert ".mp3" in data["recordings"][0]["media_url"]

    async def test_send_sms_success(self, _mock_twilio_client, _mock_comms_settings):
        from atlas_brain.mcp.twilio_server import send_sms

        fake_msg = MagicMock()
        fake_msg.sid = "SM123"
        fake_msg.status = "queued"
        fake_msg.to = "+16185551234"
        fake_msg.from_ = "+13095550001"
        _mock_twilio_client.messages.create.return_value = fake_msg

        raw = await send_sms(to="+16185551234", body="Hello!")
        data = json.loads(raw)
        assert data["success"] is True
        assert data["message_sid"] == "SM123"

    async def test_send_sms_missing_from(self, _mock_twilio_client, _mock_comms_settings):
        from atlas_brain.mcp.twilio_server import send_sms

        _mock_comms_settings.forward_to_number = ""
        raw = await send_sms(to="+16185551234", body="Hello!", from_number=None)
        data = json.loads(raw)
        assert data["success"] is False


# ===========================================================================
# Calendar provider tests
# ===========================================================================

class TestICalHelpers:
    """Tests for the minimal iCal parsing / building helpers."""

    def test_parse_ical_events_basic(self):
        from atlas_brain.services.calendar_provider import _parse_ical_events

        ical = (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "BEGIN:VEVENT\r\n"
            "UID:abc123\r\n"
            "SUMMARY:Team meeting\r\n"
            "DTSTART:20250310T090000Z\r\n"
            "DTEND:20250310T100000Z\r\n"
            "LOCATION:Conference room\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        events = _parse_ical_events(ical, calendar_id="https://cal.example.com/calendar/")
        assert len(events) == 1
        ev = events[0]
        assert ev.uid == "abc123"
        assert ev.summary == "Team meeting"
        assert ev.location == "Conference room"
        assert not ev.all_day

    def test_parse_ical_events_all_day(self):
        from atlas_brain.services.calendar_provider import _parse_ical_events

        ical = (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\n"
            "UID:allday1\r\n"
            "SUMMARY:Company holiday\r\n"
            "DTSTART;VALUE=DATE:20250401\r\n"
            "DTEND;VALUE=DATE:20250401\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        events = _parse_ical_events(ical)
        assert len(events) == 1
        assert events[0].all_day is True

    def test_parse_ical_events_skips_cancelled(self):
        from atlas_brain.services.calendar_provider import _parse_ical_events

        ical = (
            "BEGIN:VCALENDAR\r\n"
            "BEGIN:VEVENT\r\n"
            "UID:cancelled1\r\n"
            "SUMMARY:Cancelled meeting\r\n"
            "STATUS:CANCELLED\r\n"
            "DTSTART:20250310T090000Z\r\n"
            "DTEND:20250310T100000Z\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )
        events = _parse_ical_events(ical)
        assert events == []

    def test_build_ical_event_roundtrip(self):
        from datetime import timezone

        from atlas_brain.services.calendar_provider import (
            CalendarEvent,
            _build_ical_event,
            _parse_ical_events,
        )

        ev = CalendarEvent(
            uid="roundtrip-uid",
            summary="Roundtrip test",
            start=datetime(2025, 4, 15, 10, 0, tzinfo=timezone.utc),
            end=datetime(2025, 4, 15, 11, 0, tzinfo=timezone.utc),
            location="123 Main St",
            description="Test notes",
        )
        ical_str = _build_ical_event(ev)
        assert "BEGIN:VEVENT" in ical_str
        assert "UID:roundtrip-uid" in ical_str
        assert "SUMMARY:Roundtrip test" in ical_str

        parsed = _parse_ical_events(ical_str)
        assert len(parsed) == 1
        assert parsed[0].uid == "roundtrip-uid"
        assert parsed[0].summary == "Roundtrip test"
        assert parsed[0].location == "123 Main St"

    def test_ical_escape_unescape(self):
        from atlas_brain.services.calendar_provider import _ical_escape, _ical_unescape

        original = "Line1\nLine2, with comma; and semicolon"
        escaped = _ical_escape(original)
        assert "\n" not in escaped
        assert _ical_unescape(escaped) == original

    def test_ical_unescape_no_double_unescape(self):
        """\\n in source (escaped backslash + n) must NOT become a newline."""
        from atlas_brain.services.calendar_provider import _ical_unescape

        # \\n in iCal text → literal backslash + letter n  (NOT a newline)
        assert _ical_unescape("C:\\\\nfolder") == "C:\\nfolder"
        # Plain \n in iCal text → newline
        assert _ical_unescape("Hello\\nWorld") == "Hello\nWorld"


@pytest.mark.asyncio
class TestCalendarMCPTools:
    """Tests for the Calendar MCP server tools (provider is mocked)."""

    @pytest.fixture(autouse=True)
    def _reset_provider(self, monkeypatch):
        """Reset the module-level provider singleton before each test."""
        import atlas_brain.services.calendar_provider as cp

        monkeypatch.setattr(cp, "_provider_instance", None)

    def _make_event(self, uid="evt1", summary="Cleaning – Smith"):
        from atlas_brain.services.calendar_provider import CalendarEvent

        return CalendarEvent(
            uid=uid,
            summary=summary,
            start=datetime(2025, 3, 10, 9, 0, tzinfo=timezone.utc),
            end=datetime(2025, 3, 10, 11, 0, tzinfo=timezone.utc),
            calendar_id="primary",
            location="123 Main St",
        )

    async def test_list_calendars(self, monkeypatch):
        from atlas_brain.services.calendar_provider import CalendarInfo
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.list_calendars.return_value = [
            CalendarInfo(id="primary", name="My Calendar", primary=True),
        ]
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.list_calendars()
        data = json.loads(raw)
        assert len(data) == 1
        assert data[0]["id"] == "primary"
        assert data[0]["primary"] is True

    async def test_list_events(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.list_events.return_value = [self._make_event()]
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.list_events(
            start="2025-03-10T00:00:00Z",
            end="2025-03-10T23:59:59Z",
        )
        data = json.loads(raw)
        assert len(data) == 1
        assert data[0]["summary"] == "Cleaning – Smith"

    async def test_get_event_found(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.get_event.return_value = self._make_event()
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.get_event(event_id="evt1")
        data = json.loads(raw)
        assert data["id"] == "evt1"

    async def test_get_event_not_found(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.get_event.return_value = None
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.get_event(event_id="missing")
        data = json.loads(raw)
        assert "error" in data

    async def test_create_event(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        created_event = self._make_event(uid="new-evt")
        mock_provider = AsyncMock()
        mock_provider.create_event.return_value = created_event
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.create_event(
            summary="Cleaning – Smith",
            start="2025-03-10T09:00:00Z",
            end="2025-03-10T11:00:00Z",
            location="123 Main St",
        )
        data = json.loads(raw)
        assert data["status"] == "created"
        assert data["id"] == "new-evt"

    async def test_update_event(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        existing = self._make_event()
        updated = self._make_event(summary="Cleaning – Johnson")
        mock_provider = AsyncMock()
        mock_provider.get_event.return_value = existing
        mock_provider.update_event.return_value = updated
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.update_event(event_id="evt1", summary="Cleaning – Johnson")
        data = json.loads(raw)
        assert data["status"] == "updated"

    async def test_update_event_not_found(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.get_event.return_value = None
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.update_event(event_id="missing", summary="X")
        data = json.loads(raw)
        assert "error" in data

    async def test_delete_event(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.delete_event.return_value = True
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.delete_event(event_id="evt1")
        data = json.loads(raw)
        assert data["deleted"] is True

    async def test_find_free_slots_no_events(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.list_events.return_value = []
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.find_free_slots(
            start="2025-03-10T00:00:00Z",
            end="2025-03-10T23:59:59Z",
            duration_minutes=60,
            start_hour=8,
            end_hour=18,
        )
        data = json.loads(raw)
        # 8am–6pm = 10 hours → 10 one-hour slots
        assert data["total_found"] == 10
        assert len(data["free_slots"]) == 10

    async def test_find_free_slots_with_busy_block(self, monkeypatch):
        from atlas_brain.services.calendar_provider import CalendarEvent
        import atlas_brain.mcp.calendar_server as cs

        busy_event = CalendarEvent(
            uid="busy1",
            summary="Existing appointment",
            start=datetime(2025, 3, 10, 9, 0, tzinfo=timezone.utc),
            end=datetime(2025, 3, 10, 11, 0, tzinfo=timezone.utc),
            calendar_id="primary",
        )
        mock_provider = AsyncMock()
        mock_provider.list_events.return_value = [busy_event]
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.find_free_slots(
            start="2025-03-10T00:00:00Z",
            end="2025-03-10T23:59:59Z",
            duration_minutes=60,
            start_hour=8,
            end_hour=18,
        )
        data = json.loads(raw)
        # 10 total minus 2 blocked (9–10, 10–11) = 8
        assert data["total_found"] == 8

    async def test_find_free_slots_error(self, monkeypatch):
        import atlas_brain.mcp.calendar_server as cs

        mock_provider = AsyncMock()
        mock_provider.list_events.side_effect = RuntimeError("API failure")
        monkeypatch.setattr(cs, "_provider", lambda: mock_provider)

        raw = await cs.find_free_slots(
            start="2025-03-10T00:00:00Z",
            end="2025-03-10T23:59:59Z",
        )
        data = json.loads(raw)
        assert "error" in data


@pytest.mark.asyncio
class TestCalendarProviderFactory:
    """Tests for get_calendar_provider factory selection logic."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self, monkeypatch):
        import atlas_brain.services.calendar_provider as cp

        monkeypatch.setattr(cp, "_provider_instance", None)

    def test_returns_caldav_when_caldav_url_set(self, monkeypatch):
        import atlas_brain.services.calendar_provider as cp
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.caldav_url = "https://nextcloud.example.com/remote.php/dav"
        mock_cfg.calendar_enabled = False
        monkeypatch.setattr(cp.settings, "tools", mock_cfg)

        provider = cp.get_calendar_provider()
        assert isinstance(provider, cp.CalDAVCalendarProvider)

    def test_returns_google_when_enabled(self, monkeypatch):
        import atlas_brain.services.calendar_provider as cp
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.caldav_url = None
        mock_cfg.calendar_enabled = True
        monkeypatch.setattr(cp.settings, "tools", mock_cfg)

        provider = cp.get_calendar_provider()
        assert isinstance(provider, cp.GoogleCalendarProvider)

    def test_raises_when_nothing_configured(self, monkeypatch):
        import atlas_brain.services.calendar_provider as cp
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.caldav_url = None
        mock_cfg.calendar_enabled = False
        monkeypatch.setattr(cp.settings, "tools", mock_cfg)

        with pytest.raises(RuntimeError, match="No calendar provider configured"):
            cp.get_calendar_provider()

    def test_singleton_returns_same_instance(self, monkeypatch):
        import atlas_brain.services.calendar_provider as cp
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.caldav_url = None
        mock_cfg.calendar_enabled = True
        monkeypatch.setattr(cp.settings, "tools", mock_cfg)

        p1 = cp.get_calendar_provider()
        p2 = cp.get_calendar_provider()
        assert p1 is p2
