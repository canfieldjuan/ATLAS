"""BYOK (Bring Your Own Keys) provider key resolution for LLM Gateway.

Customers configure their own provider API keys (Anthropic / OpenRouter
/ etc.) in the dashboard; the LLM Gateway router (PR-D4) calls
``lookup_provider_key`` to fetch the right key per request, then
proxies through to the provider with the customer's credentials.

PR-D4 ships a stub that supports an env-var fallback for development:

  ATLAS_BYOK_<PROVIDER>_<ACCOUNT_ID>

For example, the dev/test environment can set
``ATLAS_BYOK_ANTHROPIC_00000000_0000_0000_0000_000000000000`` to
provide a default Anthropic key for the sentinel account.

PR-D5 will replace this stub with DB-backed storage:
  - New ``byok_keys`` table (encrypted at rest, account-scoped)
  - ``/api/v1/byok-keys`` router for customer key management
  - ``lookup_provider_key`` queries that table; falls back to env
    var only when no row exists (so dev and prod paths converge).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("atlas.services.byok_keys")


SUPPORTED_PROVIDERS = ("anthropic", "openrouter", "together", "groq")


def _env_var_name(provider: str, account_id: str) -> str:
    """Compose the env-var fallback name for a (provider, account)
    pair. Hyphens in the UUID become underscores so the name is
    portable across shells.
    """
    safe_account = account_id.replace("-", "_")
    return f"ATLAS_BYOK_{provider.upper()}_{safe_account}"


def lookup_provider_key(provider: str, account_id: str) -> Optional[str]:
    """Resolve the BYOK provider key for the given account.

    Returns the raw API key string when configured, or ``None`` when
    the customer has not yet supplied a key for this provider. The
    LLM Gateway router treats ``None`` as 503 "BYOK not configured".

    PR-D4 implementation: env-var fallback only. PR-D5 layers DB
    lookup on top with the env var as the dev fallback.
    """
    if provider not in SUPPORTED_PROVIDERS:
        logger.warning("BYOK lookup: unsupported provider %r", provider)
        return None
    env_name = _env_var_name(provider, account_id)
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    return raw
