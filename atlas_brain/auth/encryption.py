"""Symmetric encryption for at-rest secrets (PR-D5).

Used by the BYOK key store (``services/byok_keys.py``) to encrypt
customer-supplied provider API keys before persisting to Postgres.
Built on ``cryptography.fernet`` (AES-128-CBC + HMAC-SHA256 + base64
+ timestamp) which gives us:

  - Authenticated encryption (HMAC ensures rows can't be tampered with)
  - Key rotation via ``MultiFernet`` (one or more KEKs; new writes use
    the first; reads try each in order)
  - Built-in versioning so future KEK rotations are non-destructive

Configuration: ``SaaSAuthConfig.byok_encryption_kek`` is a comma-
separated list of ``kid:base64-key`` pairs. The first entry is the
WRITE key (new rows are encrypted under it and tagged with that kid);
all entries are tried during decrypt so older rows still load. Set
ATLAS_SAAS_BYOK_ENCRYPTION_KEK to rotate.

Example: a single starting KEK looks like
  v1:dGVzdGtleXh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHg=

To rotate: prepend a new entry, leaving the old one for backward
decryption:
  v2:newkeybase64...,v1:olderkey...

Once all rows are migrated to v2 (offline job, separate from PR-D5),
drop the v1 entry.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("atlas.auth.encryption")


_DEFAULT_KEK_SENTINEL = "byok-kek-change-me"


@dataclass(frozen=True)
class _KEKEntry:
    """One configured KEK: a kid (key id) and the loaded Fernet
    instance built from its base64 key material."""

    kid: str
    fernet: Fernet


def parse_kek_string(raw: str) -> list[_KEKEntry]:
    """Parse the ``ATLAS_SAAS_BYOK_ENCRYPTION_KEK`` env value.

    Format: ``kid1:base64key1,kid2:base64key2,...`` -- whitespace
    around entries is tolerated. The first entry is the WRITE key.

    Raises ValueError on malformed input. The loader is intentionally
    strict so a typo at deploy time fails fast instead of silently
    falling back to one-of-many keys.
    """
    if not raw or raw.strip() == _DEFAULT_KEK_SENTINEL:
        raise ValueError(
            "ATLAS_SAAS_BYOK_ENCRYPTION_KEK is not configured. "
            "Set it to a comma-separated list of kid:base64-key pairs."
        )
    entries: list[_KEKEntry] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Malformed BYOK KEK entry {chunk!r}: expected 'kid:base64key'"
            )
        kid, _, key_b64 = chunk.partition(":")
        kid = kid.strip()
        key_b64 = key_b64.strip()
        if not kid or not key_b64:
            raise ValueError(f"Malformed BYOK KEK entry {chunk!r}: empty kid or key")
        # Fernet expects base64-encoded 32 bytes; validate up front.
        try:
            decoded = base64.urlsafe_b64decode(key_b64.encode("ascii"))
        except Exception as exc:
            raise ValueError(f"BYOK KEK kid={kid}: invalid base64") from exc
        if len(decoded) != 32:
            raise ValueError(
                f"BYOK KEK kid={kid}: expected 32 raw bytes (base64-encoded), got {len(decoded)}"
            )
        try:
            fernet = Fernet(key_b64.encode("ascii"))
        except Exception as exc:
            raise ValueError(f"BYOK KEK kid={kid}: invalid Fernet key") from exc
        entries.append(_KEKEntry(kid=kid, fernet=fernet))
    if not entries:
        raise ValueError("BYOK KEK list is empty after parsing")
    return entries


def _load_keks() -> list[_KEKEntry]:
    """Read the KEK list from settings each call (so test code that
    monkeypatches the env can reload without restarting the process)."""
    from .. import config as _config

    raw = getattr(_config.settings.saas_auth, "byok_encryption_kek", "") or ""
    return parse_kek_string(raw)


def _write_kek() -> _KEKEntry:
    """The KEK new rows encrypt under -- always the first entry."""
    return _load_keks()[0]


def encrypt_secret(plaintext: str) -> tuple[bytes, str]:
    """Encrypt ``plaintext`` with the active write KEK.

    Returns ``(ciphertext, kid)``. Caller persists both columns; on
    decrypt, ``kid`` selects the right Fernet key from the rotation
    list.
    """
    if not plaintext:
        raise ValueError("encrypt_secret: refusing to encrypt empty string")
    write_entry = _write_kek()
    token = write_entry.fernet.encrypt(plaintext.encode("utf-8"))
    return token, write_entry.kid


def decrypt_secret(ciphertext: bytes, kid: str) -> Optional[str]:
    """Decrypt a row's ``encrypted_key`` using the KEK identified by
    ``kid``. Returns the plaintext, or None when no configured KEK
    matches the kid (e.g., the row was written under a KEK that has
    since been rotated out; admin attention required).
    """
    keks = _load_keks()
    matching = next((k for k in keks if k.kid == kid), None)
    if matching is None:
        logger.warning(
            "decrypt_secret: no KEK matches kid=%s (rotation drift?)",
            kid,
        )
        return None
    try:
        plaintext = matching.fernet.decrypt(ciphertext).decode("utf-8")
    except InvalidToken:
        logger.exception("decrypt_secret: InvalidToken for kid=%s", kid)
        return None
    return plaintext


def generate_kek() -> str:
    """Convenience helper for ops: returns a fresh base64 KEK suitable
    for ``ATLAS_SAAS_BYOK_ENCRYPTION_KEK``. Not used at runtime --
    callable from a python -c shell when bootstrapping a deployment.
    """
    return Fernet.generate_key().decode("ascii")
