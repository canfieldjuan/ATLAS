"""HTTP and token-grammar proof for the private Atlas public-onboarding authority."""

from __future__ import annotations

import base64
import hashlib
import hmac
import itertools
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from atlas_brain.eom_api import funnel as funnel_mod
from atlas_brain.eom_api import funnel_auth as auth_mod
from atlas_brain.eom_api.config import EOMFunnelConfig
from atlas_brain.services.eom_public_onboarding_tokens import (
    EOMPublicOnboardingTokenError,
    authenticate_eom_public_onboarding_token,
    append_eom_public_onboarding_invitation,
    eom_public_onboarding_hmac_key_fingerprint,
    format_eom_public_onboarding_token,
    parse_eom_public_onboarding_token,
)


_SERVICE = auth_mod.generate_eom_funnel_service_token()
_PUBLIC_SECRET = "this-is-a-test-only-public-onboarding-secret-value-123456"
_PREVIOUS_PUBLIC_SECRET = "previous-test-only-public-onboarding-secret-value-654321"
_PUBLIC_URL = "https://effinghamofficemaids.com/onboarding"


class _History:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id=uuid4())


class _Sender:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def __call__(self, *, to: str, subject: str, body: str, idempotency_key: str):
        self.calls.append(
            {
                "to": to,
                "subject": subject,
                "body": body,
                "idempotency_key": idempotency_key,
            }
        )
        return {"message_id": "test-resend-message", "idempotent_replay": False}


class _CRM:
    def __init__(self) -> None:
        self.session_calls: list[dict[str, str]] = []
        self.finalize_calls: list[dict[str, object]] = []
        self.revoke_calls: list[str] = []
        self.claim_calls: list[dict[str, object]] = []
        self.draft_id = uuid4()
        self.contact_id = uuid4()
        self._draft = {
            "draft_id": str(self.draft_id),
            "contact_id": str(self.contact_id),
            "recipient_email": "customer@example.com",
            "subject": "Welcome",
            "body": "Hi Customer,\n\nWelcome aboard.",
            "status": "sending",
            "blocker": None,
            "created_at": datetime(2026, 8, 17, tzinfo=timezone.utc).isoformat(),
            "claimed_at": datetime(2026, 8, 17, tzinfo=timezone.utc).isoformat(),
            "sent_at": None,
            "revoked_at": None,
            "approved_by_name": "Juan Canfield",
        }

    async def get_eom_public_onboarding_session(
        self, *, token_id: str, signing_key_fingerprint: str
    ):
        self.session_calls.append(
            {
                "token_id": token_id,
                "signing_key_fingerprint": signing_key_fingerprint,
            }
        )
        return {
            "status": "ready",
            "contact_id": str(self.contact_id),
            "full_name": "Customer Name",
            "email": "customer@example.com",
            "phone": "2175550100",
            "address": "100 Main St",
            "city": "Effingham",
            "state": "IL",
            "zip": "62401",
            "customer_type": "residential",
        }

    async def complete_eom_public_onboarding(self, **kwargs):
        self.finalize_calls.append(kwargs)
        return {
            "status": "completed",
            "contact_id": str(self.contact_id),
            "tracker_customer_id": kwargs["tracker_customer_id"],
            "tracker_site_id": kwargs["tracker_site_id"],
            "handoff_id": str(uuid4()),
            "idempotent": False,
        }

    async def revoke_eom_public_onboarding_token(self, *, draft_id: str):
        self.revoke_calls.append(draft_id)
        return {
            "token_id": str(uuid4()),
            "contact_id": str(self.contact_id),
            "status": "revoked",
            "idempotent": False,
        }

    async def claim_eom_onboarding_draft(
        self,
        *,
        draft_id: str,
        actor_id: int,
        actor_name: str,
        public_onboarding_base_url: str | None = None,
        public_onboarding_hmac_secret: str | None = None,
    ):
        self.claim_calls.append(
            {
                "draft_id": draft_id,
                "actor_id": actor_id,
                "actor_name": actor_name,
                "public_onboarding_base_url": public_onboarding_base_url,
                "public_onboarding_hmac_secret": public_onboarding_hmac_secret,
            }
        )
        result = {"claimed": True, "draft": dict(self._draft)}
        if public_onboarding_base_url is not None:
            assert public_onboarding_hmac_secret == _PUBLIC_SECRET
            token = format_eom_public_onboarding_token(
                token_id=uuid4(), secret=public_onboarding_hmac_secret
            )
            result["public_onboarding_link"] = (
                f"{public_onboarding_base_url}#token={token}"
            )
        return result

    async def confirm_eom_onboarding_draft_sent(self, *, draft_id: str):
        assert draft_id == str(self.draft_id)
        confirmed = dict(self._draft)
        confirmed["status"] = "sent"
        confirmed["sent_at"] = datetime(2026, 8, 17, 1, tzinfo=timezone.utc).isoformat()
        return confirmed


def _config(
    *,
    enabled: bool = True,
    issuance_enabled: bool | None = None,
    previous_secret: str | None = None,
) -> EOMFunnelConfig:
    values: dict[str, object] = {
        "api_enabled": True,
        "service_token_sha256": _SERVICE.sha256,
    }
    if enabled:
        values.update(
            {
                "public_onboarding_enabled": True,
                "public_onboarding_url": _PUBLIC_URL,
                "public_onboarding_hmac_secret": _PUBLIC_SECRET,
            }
        )
    if issuance_enabled is not None:
        values["public_onboarding_issuance_enabled"] = issuance_enabled
    if previous_secret is not None:
        values["public_onboarding_previous_hmac_secret"] = previous_secret
    return EOMFunnelConfig(**values)


def _app(
    crm: _CRM,
    *,
    enabled: bool = True,
    issuance_enabled: bool | None = None,
    previous_secret: str | None = None,
    sender=None,
    history=None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(funnel_mod.router)
    app.dependency_overrides[funnel_mod._crm_dependency] = lambda: crm
    app.dependency_overrides[auth_mod.get_eom_funnel_api_config] = lambda: _config(
        enabled=enabled,
        issuance_enabled=issuance_enabled,
        previous_secret=previous_secret,
    )
    app.dependency_overrides[funnel_mod._onboarding_sender_dependency] = lambda: sender
    app.dependency_overrides[funnel_mod._onboarding_email_history_dependency] = (
        lambda: history
    )
    return app


def _token(*, secret: str = _PUBLIC_SECRET) -> str:
    return format_eom_public_onboarding_token(
        token_id=uuid4(), secret=secret
    )


def _service_headers(*, actor: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_SERVICE.token}"}
    if actor:
        headers.update(
            {
                "X-EOM-Actor": "Juan Canfield",
                "X-EOM-Actor-ID": "1",
            }
        )
    return headers


def _public_token_signature_spec_oracle(token_id: UUID) -> str:
    """Compute the documented HMAC independently of the production formatter."""

    digest = hmac.new(
        _PUBLIC_SECRET.encode("utf-8"),
        f"eomob1.{token_id}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _public_token_input_shape(*, wrapper: str, bearer: str) -> object:
    """Model each direct parser input shape, including non-string containers."""

    return {
        "scalar": bearer,
        "list": [bearer],
        "tuple": (bearer,),
        "set": {bearer},
        "mapping": {"token": bearer},
        "integer": 42,
    }[wrapper]


def _public_token_grammar_spec_oracle(
    *,
    token_id: UUID,
    version: str,
    token_id_text: str,
    signature: str,
    prefix: str,
    suffix: str,
    wrapper: str,
) -> UUID | None:
    """Return the documented bearer verdict without invoking parser helpers."""

    is_canonical = (
        version == "eomob1"
        and token_id_text == str(token_id)
        and signature == _public_token_signature_spec_oracle(token_id)
        and not prefix
        and not suffix
        and wrapper == "scalar"
    )
    return token_id if is_canonical else None


def test_public_token_parser_matches_the_closed_grammar_product():
    """Generate token grammar families x framing modifiers x container wrappers.

    The expected result is a specification-derived oracle, not a value from the
    parser or formatter. It proves both the one admissible bearer and the
    fail-closed verdict for every generated alternative representation.
    """

    token_id = UUID("12345678-1234-4234-9234-123456789abc")
    canonical_signature = _public_token_signature_spec_oracle(token_id)
    version_families = ("eomob1", "eomob2", "EOMOB1")
    uuid_families = (str(token_id), str(token_id).upper(), token_id.hex)
    signature_families = (
        canonical_signature,
        canonical_signature[:-1]
        + ("A" if canonical_signature[-1] != "A" else "B"),
        canonical_signature[:-1],
    )
    framing_modifiers = (("", ""), (" ", ""), ("", ".extra"))
    container_wrappers = ("scalar", "list", "tuple", "set", "mapping", "integer")

    checked = 0
    for (
        version,
        token_id_text,
        signature,
        (prefix, suffix),
        wrapper,
    ) in itertools.product(
        version_families,
        uuid_families,
        signature_families,
        framing_modifiers,
        container_wrappers,
    ):
        bearer = f"{prefix}{version}.{token_id_text}.{signature}{suffix}"
        candidate = _public_token_input_shape(wrapper=wrapper, bearer=bearer)
        expected = _public_token_grammar_spec_oracle(
            token_id=token_id,
            version=version,
            token_id_text=token_id_text,
            signature=signature,
            prefix=prefix,
            suffix=suffix,
            wrapper=wrapper,
        )
        try:
            actual = parse_eom_public_onboarding_token(
                token=candidate, secret=_PUBLIC_SECRET
            )
        except EOMPublicOnboardingTokenError:
            actual = None

        assert actual == expected, (version, token_id_text, wrapper, prefix, suffix)
        checked += 1

    assert checked == 486


def test_public_token_parser_accepts_one_previous_verifier_with_its_key_binding():
    token_id = UUID("12345678-1234-4234-9234-123456789abc")
    old_token = format_eom_public_onboarding_token(
        token_id=token_id, secret=_PREVIOUS_PUBLIC_SECRET
    )

    authenticated = authenticate_eom_public_onboarding_token(
        token=old_token,
        secret=_PUBLIC_SECRET,
        previous_secret=_PREVIOUS_PUBLIC_SECRET,
    )

    assert authenticated.token_id == token_id
    assert authenticated.signing_key_fingerprint == eom_public_onboarding_hmac_key_fingerprint(
        secret=_PREVIOUS_PUBLIC_SECRET
    )
    assert (
        parse_eom_public_onboarding_token(
            token=old_token,
            secret=_PUBLIC_SECRET,
            previous_secret=_PREVIOUS_PUBLIC_SECRET,
        )
        == token_id
    )
    with pytest.raises(EOMPublicOnboardingTokenError):
        parse_eom_public_onboarding_token(token=old_token, secret=_PUBLIC_SECRET)


def _public_onboarding_config_spec_oracle(
    *,
    api_enabled: bool,
    enabled: bool,
    issuance_enabled: bool | None,
    url: str,
    secret: str,
    previous_secret: str,
) -> bool:
    """Encode the configuration contract independently of the model validator."""

    if any(ord(character) < 32 or ord(character) == 127 for character in url):
        return False
    base_url = url.strip()
    normalized_secret = secret.strip()
    normalized_previous_secret = previous_secret.strip()
    has_url = bool(base_url)
    has_secret = bool(normalized_secret)
    has_previous_secret = bool(normalized_previous_secret)
    if has_url != has_secret:
        return False
    if has_previous_secret and not has_secret:
        return False
    if has_previous_secret and normalized_previous_secret == normalized_secret:
        return False
    if issuance_enabled is True and not enabled:
        return False
    if enabled and (not has_url or not api_enabled):
        return False
    if not has_url:
        return True
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.netloc
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and port != 0
        and not parsed.query
        and not parsed.fragment
        and len(normalized_secret.encode("utf-8")) >= 32
        and (
            not has_previous_secret
            or len(normalized_previous_secret.encode("utf-8")) >= 32
        )
    )


def test_public_onboarding_config_matches_the_safe_url_grammar_product():
    """Generate URL families x authority/issuance states and assert the oracle.

    Config is open operator input, so every unrecognized URL/configuration
    family must settle on the safe result: model construction fails before
    minting a bearer. The test includes the blank disabled default as its own
    URL family.
    """

    scheme_families = ("https", "http", "blank")
    authority_families = (
        "effinghamofficemaids.com",
        "",
        "@",
        "user@effinghamofficemaids.com",
        "user:password@effinghamofficemaids.com",
    )
    port_families = ("", ":443", ":0", ":not-a-port")
    suffix_families = (
        "/onboarding",
        "/onboarding?preview=true",
        "/onboarding#fragment",
        "/on\nboarding",
    )
    secret_families = (_PUBLIC_SECRET, "short", "")
    previous_secret_families = (
        "",
        _PREVIOUS_PUBLIC_SECRET,
        "short",
        _PUBLIC_SECRET,
    )
    checked = 0
    for (
        scheme,
        authority,
        port,
        suffix,
        secret,
        previous_secret,
        enabled,
        issuance_enabled,
        api_enabled,
    ) in itertools.product(
        scheme_families,
        authority_families,
        port_families,
        suffix_families,
        secret_families,
        previous_secret_families,
        (False, True),
        (None, False, True),
        (False, True),
    ):
        url = "" if scheme == "blank" else f"{scheme}://{authority}{port}{suffix}"
        expected = _public_onboarding_config_spec_oracle(
            api_enabled=api_enabled,
            enabled=enabled,
            issuance_enabled=issuance_enabled,
            url=url,
            secret=secret,
            previous_secret=previous_secret,
        )
        try:
            EOMFunnelConfig(
                api_enabled=api_enabled,
                service_token_sha256=_SERVICE.sha256,
                public_onboarding_enabled=enabled,
                public_onboarding_issuance_enabled=issuance_enabled,
                public_onboarding_url=url,
                public_onboarding_hmac_secret=secret,
                public_onboarding_previous_hmac_secret=previous_secret,
            )
        except ValidationError:
            actual = False
        else:
            actual = True

        assert actual == expected, (
            scheme,
            authority,
            port,
            suffix,
            secret,
            previous_secret,
            enabled,
            issuance_enabled,
        )
        checked += 1

    assert checked == 34560


def test_public_onboarding_config_requires_a_complete_safe_pair():
    with pytest.raises(ValidationError, match="must be set together"):
        EOMFunnelConfig(
            public_onboarding_url=_PUBLIC_URL,
            public_onboarding_hmac_secret="",
        )
    with pytest.raises(ValidationError, match="HTTPS URL"):
        EOMFunnelConfig(
            public_onboarding_url="http://example.test/onboarding",
            public_onboarding_hmac_secret=_PUBLIC_SECRET,
        )
    with pytest.raises(ValidationError, match="valid HTTPS URL"):
        EOMFunnelConfig(
            public_onboarding_url="https://example.test:not-a-port/onboarding",
            public_onboarding_hmac_secret=_PUBLIC_SECRET,
        )
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        EOMFunnelConfig(
            public_onboarding_url=_PUBLIC_URL,
            public_onboarding_hmac_secret="short",
        )
    with pytest.raises(ValidationError, match="API_ENABLED=true"):
        EOMFunnelConfig(
            public_onboarding_enabled=True,
            public_onboarding_url=_PUBLIC_URL,
            public_onboarding_hmac_secret=_PUBLIC_SECRET,
        )
    with pytest.raises(ValidationError, match="issuance requires"):
        EOMFunnelConfig(
            api_enabled=True,
            service_token_sha256=_SERVICE.sha256,
            public_onboarding_enabled=False,
            public_onboarding_issuance_enabled=True,
            public_onboarding_url=_PUBLIC_URL,
            public_onboarding_hmac_secret=_PUBLIC_SECRET,
        )
    with pytest.raises(ValidationError, match="PREVIOUS_HMAC_SECRET requires"):
        EOMFunnelConfig(
            public_onboarding_previous_hmac_secret=_PREVIOUS_PUBLIC_SECRET,
        )
    with pytest.raises(ValidationError, match="must differ"):
        EOMFunnelConfig(
            public_onboarding_url=_PUBLIC_URL,
            public_onboarding_hmac_secret=_PUBLIC_SECRET,
            public_onboarding_previous_hmac_secret=_PUBLIC_SECRET,
        )
    with pytest.raises(ValidationError, match="previous HMAC secret must be at least"):
        EOMFunnelConfig(
            public_onboarding_url=_PUBLIC_URL,
            public_onboarding_hmac_secret=_PUBLIC_SECRET,
            public_onboarding_previous_hmac_secret="short",
        )


@pytest.mark.parametrize(
    "unsafe_url",
    (
        "https://@/onboarding",
        "https://:443/onboarding",
        "https://user@/onboarding",
        "https://example.test:0/onboarding",
        "https://example.test/onboarding?preview=true",
        "https://example.test/onboarding#fragment",
        "https://example.test/on\nboarding",
        "https://example.test/on\rboarding",
        "https://example.test/on\tboarding",
        "https://example.test/on\x7fboarding",
    ),
)
def test_public_onboarding_config_rejects_malformed_or_bearer_leaking_urls(unsafe_url):
    with pytest.raises(ValidationError, match="HTTPS URL|control characters"):
        EOMFunnelConfig(
            public_onboarding_url=unsafe_url,
            public_onboarding_hmac_secret=_PUBLIC_SECRET,
        )


def test_public_onboarding_invitation_preserves_the_draft_body_verbatim():
    body = "Custom draft ending\n"
    invitation = append_eom_public_onboarding_invitation(
        body=body, link="https://effinghamofficemaids.com/onboarding#token=opaque"
    )

    assert invitation.startswith(f"{body}\n\n")


@pytest.mark.asyncio
async def test_public_session_requires_service_auth_and_never_requires_an_actor():
    crm = _CRM()
    token = _token()
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        denied = await client.post(
            "/eom-funnel/public-onboarding/session", json={"token": token}
        )
        accepted = await client.post(
            "/eom-funnel/public-onboarding/session",
            headers=_service_headers(),
            json={"token": token},
        )

    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "ready"
    assert "contact_id" not in accepted.json()
    assert "handoff_id" not in accepted.json()
    assert crm.session_calls == [
        {
            "token_id": str(
                parse_eom_public_onboarding_token(token=token, secret=_PUBLIC_SECRET)
            ),
            "signing_key_fingerprint": eom_public_onboarding_hmac_key_fingerprint(
                secret=_PUBLIC_SECRET
            ),
        }
    ]


@pytest.mark.asyncio
async def test_public_session_binds_a_previous_key_to_the_private_provider_call():
    crm = _CRM()
    token = _token(secret=_PREVIOUS_PUBLIC_SECRET)
    app = _app(crm, previous_secret=_PREVIOUS_PUBLIC_SECRET)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        accepted = await client.post(
            "/eom-funnel/public-onboarding/session",
            headers=_service_headers(),
            json={"token": token},
        )

    assert accepted.status_code == 200
    assert crm.session_calls == [
        {
            "token_id": str(
                parse_eom_public_onboarding_token(
                    token=token,
                    secret=_PUBLIC_SECRET,
                    previous_secret=_PREVIOUS_PUBLIC_SECRET,
                )
            ),
            "signing_key_fingerprint": eom_public_onboarding_hmac_key_fingerprint(
                secret=_PREVIOUS_PUBLIC_SECRET
            ),
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("/eom-funnel/public-onboarding/session", {"token": "not-used"}),
        (
            "/eom-funnel/public-onboarding/finalize",
            {"token": "not-used", "tracker_customer_id": 12, "tracker_site_id": 24},
        ),
    ),
)
async def test_public_onboarding_routes_fail_closed_when_public_authority_is_disabled(
    path, payload
):
    crm = _CRM()
    app = _app(crm, enabled=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(path, headers=_service_headers(), json=payload)

    assert response.status_code == 503
    assert crm.session_calls == []
    assert crm.finalize_calls == []


@pytest.mark.asyncio
async def test_paused_public_onboarding_issuance_keeps_existing_links_redeemable():
    crm = _CRM()
    token = _token()
    app = _app(crm, issuance_enabled=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        denied = await client.post(
            "/eom-funnel/public-onboarding/session", json={"token": token}
        )
        session = await client.post(
            "/eom-funnel/public-onboarding/session",
            headers=_service_headers(),
            json={"token": token},
        )
        finalized = await client.post(
            "/eom-funnel/public-onboarding/finalize",
            headers=_service_headers(),
            json={
                "token": token,
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert denied.status_code == 401
    assert session.status_code == 200
    assert finalized.status_code == 201
    assert len(crm.session_calls) == 1
    assert len(crm.finalize_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    ("eomob1.not-a-token", 42, "x" * 512, {"not": "a bearer"}),
)
async def test_public_session_rejects_bad_bearer_before_the_crm_lookup(token):
    crm = _CRM()
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/public-onboarding/session",
            headers=_service_headers(),
            json={"token": token},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Public onboarding link is unavailable"
    assert crm.session_calls == []


@pytest.mark.asyncio
async def test_public_finalize_delegates_token_key_binding_and_tracker_ids():
    crm = _CRM()
    token = _token()
    app = _app(crm)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/eom-funnel/public-onboarding/finalize",
            headers=_service_headers(),
            json={
                "token": token,
                "tracker_customer_id": 12,
                "tracker_site_id": 24,
            },
        )

    assert response.status_code == 201
    assert response.json() == {
        "success": True,
        "status": "completed",
        "tracker_customer_id": 12,
        "tracker_site_id": 24,
        "idempotent": False,
    }
    assert crm.finalize_calls == [
        {
            "token_id": str(
                parse_eom_public_onboarding_token(token=token, secret=_PUBLIC_SECRET)
            ),
            "signing_key_fingerprint": eom_public_onboarding_hmac_key_fingerprint(
                secret=_PUBLIC_SECRET
            ),
            "tracker_customer_id": 12,
            "tracker_site_id": 24,
        }
    ]


@pytest.mark.asyncio
async def test_enabled_approval_sends_fragment_link_without_persisting_bearer():
    crm = _CRM()
    sender = _Sender()
    history = _History()
    app = _app(crm, sender=sender, history=history)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/onboarding-drafts/{crm.draft_id}/approve-send",
            headers=_service_headers(actor=True),
        )

    assert response.status_code == 201
    assert crm.claim_calls[0]["public_onboarding_base_url"] == _PUBLIC_URL
    assert "Please complete your onboarding details before your first visit." in sender.calls[0]["body"]
    assert "Complete sus datos de incorporación antes de su primera visita:" in sender.calls[0]["body"]
    assert "#token=eomob1." in sender.calls[0]["body"]
    assert "#token=" not in response.text
    assert "#token=" not in crm._draft["body"]
    assert "#token=" not in history.created[0]["body"]


@pytest.mark.asyncio
async def test_paused_public_onboarding_issuance_keeps_approval_email_untokenized():
    crm = _CRM()
    sender = _Sender()
    history = _History()
    app = _app(crm, issuance_enabled=False, sender=sender, history=history)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/eom-funnel/onboarding-drafts/{crm.draft_id}/approve-send",
            headers=_service_headers(actor=True),
        )

    assert response.status_code == 201
    assert crm.claim_calls == [
        {
            "draft_id": str(crm.draft_id),
            "actor_id": 1,
            "actor_name": "Juan Canfield",
            "public_onboarding_base_url": None,
            "public_onboarding_hmac_secret": None,
        }
    ]
    assert sender.calls[0]["body"] == crm._draft["body"]
    assert "#token=" not in sender.calls[0]["body"]
    assert "#token=" not in history.created[0]["body"]


@pytest.mark.asyncio
async def test_staff_link_revocation_requires_an_actor_and_survives_feature_disable():
    crm = _CRM()
    app = _app(crm)
    disabled_crm = _CRM()
    disabled_app = _app(disabled_crm, enabled=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        no_actor = await client.post(
            f"/eom-funnel/onboarding-drafts/{crm.draft_id}/revoke-link",
            headers=_service_headers(),
        )
        revoked = await client.post(
            f"/eom-funnel/onboarding-drafts/{crm.draft_id}/revoke-link",
            headers=_service_headers(actor=True),
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=disabled_app), base_url="http://test"
    ) as client:
        disabled = await client.post(
            f"/eom-funnel/onboarding-drafts/{disabled_crm.draft_id}/revoke-link",
            headers=_service_headers(actor=True),
        )

    assert no_actor.status_code == 422
    assert revoked.status_code == 201
    assert disabled.status_code == 201
    assert crm.revoke_calls == [str(crm.draft_id)]
    assert disabled_crm.revoke_calls == [str(disabled_crm.draft_id)]
