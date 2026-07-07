import bcrypt
import pytest
from pydantic import ValidationError

from atlas_brain.api.auth import ChangePasswordRequest, RegisterRequest, ResetPasswordRequest
from atlas_brain.auth.passwords import BCRYPT_MAX_PASSWORD_BYTES, hash_password, verify_password


def test_hash_password_accepts_and_verifies_valid_password() -> None:
    hashed = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", hashed) is True
    assert verify_password("wrong horse battery staple", hashed) is False


def test_hash_password_rejects_multibyte_password_over_bcrypt_byte_limit() -> None:
    password = "é" * 37

    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        hash_password(password)


def test_verify_password_preserves_legacy_bcrypt_truncation_behavior() -> None:
    prefix = "a" * BCRYPT_MAX_PASSWORD_BYTES
    legacy_long_password = prefix + "ignored-by-bcrypt-4"
    legacy_hash = bcrypt.hashpw(prefix.encode("utf-8"), bcrypt.gensalt()).decode()

    assert verify_password(legacy_long_password, legacy_hash) is True


def test_register_request_rejects_multibyte_password_over_bcrypt_byte_limit() -> None:
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        RegisterRequest(
            email="owner@example.com",
            password="🔥" * 19,
            full_name="Owner",
            account_name="Example Co",
        )


def test_change_password_rejects_multibyte_new_password_over_bcrypt_byte_limit() -> None:
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        ChangePasswordRequest(
            current_password="old-password",
            new_password="🔥" * 19,
        )


def test_reset_password_rejects_multibyte_new_password_over_bcrypt_byte_limit() -> None:
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        ResetPasswordRequest(
            token="reset-token",
            new_password="🔥" * 19,
        )
