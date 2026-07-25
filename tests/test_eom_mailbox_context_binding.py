"""Behavioral proofs for scoped CRM IMAP authorization."""

import json
from decimal import Decimal
from fractions import Fraction
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from atlas_brain import config as config_mod
from atlas_brain.services import customer_context as context_mod
from atlas_brain.services import email_provider as email_provider_mod


TENANT = "effingham_maids"
OTHER_TENANT = "churnsignals"
TENANT_CONTACT_ID = "11111111-1111-4111-8111-111111111111"
OTHER_CONTACT_ID = "22222222-2222-4222-8222-222222222222"


def _binding(**overrides):
    values = {
        "provider": "imap",
        "imap_host": "imap.eom.example",
        "imap_port": 993,
        "imap_username": "office@eom.example",
        "imap_password": "eom-app-password",
        "imap_ssl": True,
        "imap_mailbox": "INBOX",
    }
    values.update(overrides)
    return config_mod.InboxMailboxBinding(**values)


def test_config_preserves_exact_context_keys_and_accepts_secret_free_gmail(
    monkeypatch,
):
    exact = f" {TENANT} "
    monkeypatch.setenv(
        "ATLAS_EMAIL_INBOX_CONTEXT_BINDINGS",
        json.dumps(
            {
                TENANT: _binding().model_dump(mode="json"),
                exact: _binding(
                    imap_username="spaced@eom.example"
                ).model_dump(mode="json"),
            }
        ),
    )

    config = config_mod.EmailConfig()

    assert set(config.inbox_context_bindings) == {TENANT, exact}
    assert (
        config.inbox_context_bindings[exact].imap_username
        == "spaced@eom.example"
    )
    monkeypatch.delenv("ATLAS_EMAIL_INBOX_CONTEXT_BINDINGS")

    typed = _binding()
    assert config_mod.EmailConfig(
        inbox_context_bindings={TENANT: typed}
    ).inbox_context_bindings[TENANT] == typed
    gmail = config_mod.InboxMailboxBinding(provider="gmail")
    assert gmail.provider == "gmail"
    assert gmail.imap_password.get_secret_value() == ""
    assert config_mod.EmailConfig(
        inbox_context_bindings={TENANT: gmail}
    ).inbox_context_bindings[TENANT] == gmail
    with pytest.raises(ValidationError, match="unsupported fields"):
        config_mod.InboxMailboxBinding(
            provider="gmail",
            gmail_client_id="client",
            gmail_client_secret="secret",
            gmail_refresh_token="refresh",
        )


@pytest.mark.parametrize(
    "bindings",
    [
        "not-a-map",
        {1: {"provider": "imap"}},
        {" ": {"provider": "imap"}},
        {TENANT: "not-a-binding"},
        {TENANT: {"provider": "imap", "unknown": "nested-secret"}},
        {
            TENANT: {
                "provider": "gmail",
                "gmail_refresh_token": "nested-secret",
            }
        },
        {
            TENANT: {
                "provider": "imap",
                "imap_username": "office@example.com",
                "imap_password": "nested-secret",
            }
        },
        {
            TENANT: {
                **_binding().model_dump(mode="json"),
                "imap_port": "nested-secret",
            }
        },
        {
            TENANT: {
                **_binding().model_dump(mode="json"),
                "imap_ssl": "nested-secret",
            }
        },
    ],
)
def test_invalid_binding_grammar_redacts_nested_values(bindings):
    with pytest.raises(ValidationError) as captured:
        config_mod.EmailConfig(inbox_context_bindings=bindings)

    rendered = str(captured.value)
    structured = json.dumps(captured.value.errors(), default=str)
    assert "nested-secret" not in rendered
    assert "nested-secret" not in structured
    assert "nested-secret" not in captured.value.json()


@pytest.mark.parametrize(
    "port",
    [True, "993.0", b"993", Decimal("993"), Fraction(993, 1)],
)
def test_binding_preflight_preserves_port_coercion(port):
    binding = _binding().model_dump()
    binding["imap_port"] = port
    config = config_mod.EmailConfig(
        inbox_context_bindings={TENANT: binding}
    )

    expected = 1 if port is True else 993
    assert config.inbox_context_bindings[TENANT].imap_port == expected


@pytest.mark.parametrize(
    ("use_ssl", "expected"),
    [
        (0.0, False),
        (1.0, True),
        (Decimal(0), False),
        (Decimal(1), True),
        (Fraction(0, 1), False),
        (Fraction(1, 1), True),
    ],
)
def test_binding_preflight_preserves_ssl_coercion(use_ssl, expected):
    binding = _binding().model_dump()
    binding["imap_ssl"] = use_ssl
    config = config_mod.EmailConfig(
        inbox_context_bindings={TENANT: binding}
    )

    assert config.inbox_context_bindings[TENANT].imap_ssl is expected


@pytest.mark.asyncio
async def test_scoped_provider_rebinds_without_authorization_cache(monkeypatch):
    import atlas_brain.config as config_mod

    monkeypatch.setattr(
        config_mod.settings.email,
        "inbox_context_bindings",
        {TENANT: _binding(imap_username="mailbox-a@example.com")},
    )
    first = await email_provider_mod.get_scoped_inbox_provider(TENANT)

    monkeypatch.setattr(
        config_mod.settings.email,
        "inbox_context_bindings",
        {TENANT: _binding(imap_username="mailbox-b@example.com")},
    )
    second = await email_provider_mod.get_scoped_inbox_provider(TENANT)

    assert first is not second
    assert first._username == "mailbox-a@example.com"
    assert second._username == "mailbox-b@example.com"
    with pytest.raises(email_provider_mod.UnmappedInboxContextError):
        await email_provider_mod.get_scoped_inbox_provider(OTHER_TENANT)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            '"Bergmann, Katharina (Facilities Operations)"\r\n'
            " <katharina.bergmann@example.com>",
            "katharina.bergmann@example.com",
        ),
        (
            '"Bergmann, Katharina"\r\n\t'
            "<katharina.bergmann@example.com>",
            "katharina.bergmann@example.com",
        ),
        (
            '"Bergmann, Katharina"\n '
            "<katharina.bergmann@example.com>",
            "katharina.bergmann@example.com",
        ),
        ("Alice <alice@example.com>\r\nBcc: victim@example.com", None),
        ("Alice <alice@example.com>\nBcc: victim@example.com", None),
        (
            "Alice <alice@example.com>,\r\n Bob <bob@example.com>",
            None,
        ),
        ("Friends: Alice <alice@example.com>;", None),
        ("=?utf-8?q?alice=40example.com?=", None),
    ],
)
def test_sender_admission_unfolds_only_legal_continuations(value, expected):
    assert context_mod.CustomerContextService._parse_single_author(value) == expected


def test_real_imap_envelope_preserves_folded_and_duplicate_sender_evidence():
    folded = (
        b'From: "Bergmann, Katharina (Facilities Operations)"\r\n'
        b" <katharina.bergmann@example.com>\r\n"
        b"Subject: Facilities\r\n\r\n"
    )
    envelope = email_provider_mod._parse_envelope(
        "1",
        folded,
        preserve_sender_evidence=True,
    )
    assert context_mod.CustomerContextService._strict_sender_mailbox(
        envelope
    ) == "katharina.bergmann@example.com"

    duplicate = email_provider_mod._parse_envelope(
        "2",
        (
            b"From: alice@example.com\r\n"
            b"From: attacker@example.com\r\n\r\n"
        ),
        preserve_sender_evidence=True,
    )
    assert (
        context_mod.CustomerContextService._strict_sender_mailbox(duplicate)
        is None
    )
    assert context_mod.CustomerContextService._strict_sender_mailbox(
        {"from": "alice@example.com"}
    ) is None


@pytest.mark.asyncio
async def test_unscoped_inbox_preserves_legacy_provider_contract(monkeypatch):
    class _LegacyProvider:
        def __init__(self):
            self.calls = []

        async def list_messages(self, query, max_results):
            self.calls.append((query, max_results))
            return [{"id": "legacy", "from": "decoded-only"}]

    provider = _LegacyProvider()
    monkeypatch.setattr(email_provider_mod, "_email_provider", provider)

    result = await context_mod.CustomerContextService()._get_inbox_emails(
        {"email": "jos\u00e9@example.com"},
        -7,
    )

    assert result == [{"id": "legacy", "from": "decoded-only"}]
    assert provider.calls == [("from:josé@example.com", -7)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejected",
    [
        {
            "id": "collision",
            "_atlas_from_header_values": [
                '"customer@example.com" <notcustomer@example.com>'
            ],
        },
        {
            "id": "malformed",
            "_atlas_from_header_values": [
                "Undisclosed:; customer@example.com"
            ],
        },
    ],
)
async def test_scoped_inbox_scans_past_rejected_candidates(rejected):
    class _Candidates:
        def __init__(self):
            self.calls = []

        async def list_messages(self, query, max_results):
            self.calls.append((query, max_results))
            return [
                rejected,
                {
                    "id": "exact",
                    "_atlas_from_header_values": [
                        "Customer <customer@example.com>"
                    ],
                },
            ]

    provider = _Candidates()

    result = await context_mod.CustomerContextService()._get_inbox_emails(
        {"email": "customer@example.com"},
        1,
        provider=provider,
    )

    assert result == [{"id": "exact"}]
    assert provider.calls == [('from:"customer@example.com"', 50)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sender", "expected"),
    [
        ("Alice@example.com", []),
        ("alice@EXAMPLE.COM", [{"id": "domain-case"}]),
    ],
)
async def test_scoped_inbox_preserves_local_part_case(sender, expected):
    class _Candidate:
        async def list_messages(self, **_kwargs):
            return [
                {
                    "id": "domain-case",
                    "_atlas_from_header_values": [sender],
                }
            ]

    result = await context_mod.CustomerContextService()._get_inbox_emails(
        {"email": "alice@example.com"},
        1,
        provider=_Candidate(),
    )

    assert result == expected


class _CRMProvider:
    def __init__(self):
        self.contacts = {
            TENANT_CONTACT_ID: {
                "id": TENANT_CONTACT_ID,
                "email": "customer@example.com",
                "full_name": "Customer",
                "business_context_id": TENANT,
            },
            OTHER_CONTACT_ID: {
                "id": OTHER_CONTACT_ID,
                "email": "other@example.com",
                "full_name": "Other",
                "business_context_id": OTHER_TENANT,
            },
        }

    async def get_contact(self, contact_id):
        contact = self.contacts.get(str(contact_id))
        return dict(contact) if contact else None

    async def get_interactions(
        self, _contact_id, *, limit, business_context_id
    ):
        return []

    async def get_contact_appointments(
        self, _contact_id, *, business_context_id
    ):
        return []


class _FakeIMAP:
    connections = []
    on_search = None

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.uid_calls = []
        self.__class__.connections.append(self)

    def login(self, username, password):
        self.login_args = (username, password)
        return "OK", []

    def noop(self):
        return "OK", []

    def select(self, mailbox, readonly=True):
        self.selected = (mailbox, readonly)
        return "OK", []

    def uid(self, command, *args):
        self.uid_calls.append((command, args))
        if command == "search":
            if self.__class__.on_search is not None:
                self.__class__.on_search()
            return "OK", [b"41 42"]
        if command == "fetch":
            collision = (
                b'From: "customer@example.com" <notcustomer@example.com>\r\n'
                b"Subject: Collision\r\n\r\n"
            )
            folded = (
                b'From: "Bergmann, Katharina (Facilities Operations)"\r\n'
                b" <customer@example.com>\r\n"
                b"Subject: Bound reply\r\n\r\n"
            )
            headers = {"42": collision, "41": folded}
            response = []
            for index, uid in enumerate(str(args[0]).split(","), start=1):
                response.extend(
                    [
                        (
                            f"{index} (UID {uid} RFC822.HEADER {{140}}".encode(),
                            headers[uid],
                        ),
                        b")",
                    ]
                )
            return "OK", response
        raise AssertionError(f"unexpected IMAP command: {command}")

    def logout(self):
        return "BYE", []


@pytest.mark.asyncio
async def test_real_crm_mcp_uses_only_bound_imap_and_refuses_unmapped(
    monkeypatch,
):
    import atlas_brain.config as config_mod
    import atlas_brain.mcp.crm_server as crm_server
    import atlas_brain.services.crm_provider as crm_provider_mod

    provider = _CRMProvider()
    monkeypatch.setattr(
        crm_provider_mod,
        "get_crm_provider",
        lambda: provider,
    )
    monkeypatch.setattr(email_provider_mod.imaplib, "IMAP4_SSL", _FakeIMAP)
    global_provider = AsyncMock()
    monkeypatch.setattr(
        email_provider_mod,
        "get_email_provider",
        global_provider,
    )
    monkeypatch.setattr(
        config_mod.settings.email,
        "inbox_context_bindings",
        {TENANT: _binding()},
    )
    _FakeIMAP.connections.clear()
    crm_server.set_provider_override(lambda: provider)
    previous_service = context_mod._customer_context_service
    context_mod._customer_context_service = None

    try:
        scoped = json.loads(
            await crm_server.get_customer_context(
                contact_id=TENANT_CONTACT_ID,
                business_context_id=TENANT,
                max_emails=1,
            )
        )
        assert [row["id"] for row in scoped["inbox_emails"]] == ["41"]
        assert scoped["email_sources_omitted_under_scope"] == []
        assert scoped["emails_omitted_under_scope"] is False
        assert len(_FakeIMAP.connections) == 1
        connection = _FakeIMAP.connections[0]
        assert (connection.host, connection.port) == (
            "imap.eom.example",
            993,
        )
        assert connection.login_args == (
            "office@eom.example",
            "eom-app-password",
        )
        assert connection.uid_calls[0] == (
            "search",
            (None, 'FROM "customer@example.com"'),
        )

        _FakeIMAP.on_search = lambda: provider.contacts[
            TENANT_CONTACT_ID
        ].update(email="Customer@example.com")
        raced = json.loads(
            await crm_server.get_customer_context(
                contact_id=TENANT_CONTACT_ID,
                business_context_id=TENANT,
                max_emails=5,
            )
        )
        assert raced["inbox_emails"] == []
        assert raced["emails_omitted_under_scope"] is False

        unmapped = json.loads(
            await crm_server.get_customer_context(
                contact_id=OTHER_CONTACT_ID,
                business_context_id=OTHER_TENANT,
                max_emails=5,
            )
        )
        assert unmapped["inbox_emails"] == []
        assert unmapped["email_sources_omitted_under_scope"] == [
            "inbox_emails"
        ]
        assert len(_FakeIMAP.connections) == 2
        global_provider.assert_not_called()
    finally:
        _FakeIMAP.on_search = None
        crm_server.set_provider_override(None)
        context_mod._customer_context_service = previous_service
