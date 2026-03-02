"""JWT token creation and decoding."""

from datetime import datetime, timedelta, timezone

import jwt

from ..config import settings

_cfg = settings.saas_auth


def create_access_token(user_id: str, account_id: str, plan: str) -> str:
    """Create a short-lived access token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "account_id": account_id,
        "plan": plan,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(hours=_cfg.jwt_expiry_hours),
    }
    return jwt.encode(payload, _cfg.jwt_secret, algorithm=_cfg.jwt_algorithm)


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=_cfg.jwt_refresh_expiry_days),
    }
    return jwt.encode(payload, _cfg.jwt_secret, algorithm=_cfg.jwt_algorithm)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises jwt.PyJWTError on failure."""
    return jwt.decode(token, _cfg.jwt_secret, algorithms=[_cfg.jwt_algorithm])
