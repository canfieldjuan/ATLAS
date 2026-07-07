"""Password hashing and verification using bcrypt."""

import bcrypt

BCRYPT_MAX_PASSWORD_BYTES = 72
_BCRYPT_PASSWORD_LIMIT_MESSAGE = "Password must be 72 UTF-8 bytes or fewer"


def validate_bcrypt_password_length(plain: str) -> str:
    """Return ``plain`` when bcrypt can hash it without byte truncation."""
    if len(plain.encode("utf-8")) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(_BCRYPT_PASSWORD_LIMIT_MESSAGE)
    return plain


def _password_bytes_for_hash(plain: str) -> bytes:
    validate_bcrypt_password_length(plain)
    return plain.encode("utf-8")


def _password_bytes_for_verify(plain: str) -> bytes:
    return plain.encode("utf-8")[:BCRYPT_MAX_PASSWORD_BYTES]


def hash_password(plain: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(_password_bytes_for_hash(plain), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(_password_bytes_for_verify(plain), hashed.encode())
