import bcrypt
import pytest
from pydantic import ValidationError

from atlas_brain.api.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
)
from atlas_brain.auth.passwords import BCRYPT_MAX_PASSWORD_BYTES, hash_password, verify_password


def test_hash_password_accepts_and_verifies_valid_password() -> None:
    hashed = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", hashed) is True
    assert verify_password("wrong horse battery staple", hashed) is False


def test_hash_password_rejects_multibyte_password_over_bcrypt_byte_limit() -> None:
    password = "é" * 37

    with pytest.raises(ValueError, match="72 UTF-8 bytes"):
        hash_password(password)


def test_verify_password_rejects_over_bcrypt_byte_limit_input() -> None:
    prefix = "a" * BCRYPT_MAX_PASSWORD_BYTES
    overlong_password = prefix + "ignored-by-bcrypt"
    password_hash = bcrypt.hashpw(prefix.encode("utf-8"), bcrypt.gensalt()).decode()

    assert verify_password(overlong_password, password_hash) is False


def test_verify_password_rejects_exact_limit_multibyte_prefix_with_suffix() -> None:
    prefix = "é" * 36
    password_hash = hash_password(prefix)

    assert len(prefix.encode("utf-8")) == BCRYPT_MAX_PASSWORD_BYTES
    assert verify_password(prefix + "anything", password_hash) is False


def test_register_request_rejects_multibyte_password_over_bcrypt_byte_limit() -> None:
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        RegisterRequest(
            email="owner@example.com",
            password="🔥" * 19,
            full_name="Owner",
            account_name="Example Co",
        )


def test_login_request_rejects_multibyte_password_over_bcrypt_byte_limit() -> None:
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        LoginRequest(
            email="owner@example.com",
            password="🔥" * 19,
        )


def test_change_password_rejects_multibyte_current_password_over_bcrypt_byte_limit() -> None:
    with pytest.raises(ValidationError, match="72 UTF-8 bytes"):
        ChangePasswordRequest(
            current_password="🔥" * 19,
            new_password="new-password",
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
