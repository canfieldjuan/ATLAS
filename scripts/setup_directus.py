#!/usr/bin/env python3
"""
Directus CRM first-run setup for Atlas.

Creates a long-lived API token in Directus and writes it to .env.local so
Atlas Brain uses DirectusCRMProvider (Directus REST) instead of the direct
asyncpg fallback.

Prerequisites:
1. Start the stack:  docker compose up -d directus postgres
2. Wait for Directus to be ready:  docker compose logs -f directus
   (look for "Server started at http://0.0.0.0:8055")
3. Run this script:  python scripts/setup_directus.py

After completion:
  - ATLAS_DIRECTUS_TOKEN is written to .env.local
  - ATLAS_DIRECTUS_ENABLED is set to true in .env.local
  - Restart brain:  docker compose restart brain

Directus Admin UI:  http://localhost:8055
  Default credentials are set via ATLAS_DIRECTUS_ADMIN_EMAIL /
  ATLAS_DIRECTUS_ADMIN_PASSWORD in docker-compose.yml.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_LOCAL = PROJECT_ROOT / ".env.local"

# Defaults matching docker-compose.yml
DEFAULT_URL = "http://localhost:8055"
DEFAULT_EMAIL = "admin@atlas.local"
DEFAULT_PASSWORD = "atlas_admin_password"


def _json_post(url: str, payload: dict, token: str | None = None) -> dict:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"******"
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc


def _json_get(url: str, token: str) -> dict:
    req = Request(url, headers={"Authorization": f"******"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def wait_for_directus(base_url: str, retries: int = 30, delay: float = 2.0) -> None:
    health_url = f"{base_url}/server/health"
    print(f"Waiting for Directus at {base_url} ", end="", flush=True)
    for i in range(retries):
        try:
            req = Request(health_url)
            with urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    print(" ready.")
                    return
        except (URLError, Exception):
            pass
        print(".", end="", flush=True)
        time.sleep(delay)
    print()
    raise RuntimeError(
        f"Directus did not become ready after {retries * delay:.0f}s.\n"
        "Make sure it is running:  docker compose up -d directus postgres"
    )


def authenticate(base_url: str, email: str, password: str) -> str:
    print(f"Authenticating as {email} ...")
    result = _json_post(
        f"{base_url}/auth/login",
        {"email": email, "password": password},
    )
    token = result.get("data", {}).get("access_token")
    if not token:
        raise RuntimeError(f"Login failed. Response: {result}")
    print("  Authentication successful.")
    return token


def create_api_token(base_url: str, access_token: str) -> str:
    """Create a named static API token via /users/me/token or /auth/static-token."""
    print("Creating Atlas API token ...")

    # Use the Directus token endpoint to create a permanent static token
    token_name = f"atlas-brain-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    # Directus ≥10: PATCH /users/me with "token" field creates a static bearer token
    body = json.dumps({"token": token_name}).encode()
    req = Request(
        f"{base_url}/users/me",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"******",
        },
        method="PATCH",
    )
    try:
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            static_token = result.get("data", {}).get("token")
            if static_token:
                print(f"  Static token created: {static_token[:8]}…")
                return static_token
    except HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        print(f"  PATCH /users/me returned {exc.code}: {raw[:200]}")

    # Fallback: return the access_token itself (still works; short-lived but useful for initial setup)
    print("  Note: could not create static token; using session token as fallback.")
    print("  Manually create a static API token in Directus Admin UI:")
    print(f"  {base_url}/admin → Settings → API Tokens → Create Token")
    return access_token


def write_env_local(token: str, directus_url: str) -> None:
    """Upsert ATLAS_DIRECTUS_TOKEN and ATLAS_DIRECTUS_ENABLED into .env.local."""
    lines: list[str] = []
    if ENV_LOCAL.exists():
        lines = ENV_LOCAL.read_text().splitlines()

    new_lines: list[str] = []
    found_token = found_enabled = False
    for line in lines:
        if line.startswith("ATLAS_DIRECTUS_TOKEN="):
            new_lines.append(f"ATLAS_DIRECTUS_TOKEN={token}")
            found_token = True
        elif line.startswith("ATLAS_DIRECTUS_ENABLED="):
            new_lines.append("ATLAS_DIRECTUS_ENABLED=true")
            found_enabled = True
        else:
            new_lines.append(line)

    if not found_token:
        new_lines.append(f"ATLAS_DIRECTUS_TOKEN={token}")
    if not found_enabled:
        new_lines.append("ATLAS_DIRECTUS_ENABLED=true")

    ENV_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    ENV_LOCAL.write_text("\n".join(new_lines) + "\n")
    print(f"\n✅  Written to {ENV_LOCAL}:")
    print(f"   ATLAS_DIRECTUS_TOKEN={token[:8]}…")
    print("   ATLAS_DIRECTUS_ENABLED=true")


def verify_connection(base_url: str, token: str) -> None:
    """Confirm the token can reach Directus."""
    print("\nVerifying token against Directus …")
    info = _json_get(f"{base_url}/users/me", token)
    email = info.get("data", {}).get("email", "unknown")
    print(f"  Verified — logged in as: {email}")


def main() -> None:
    base_url = os.environ.get("ATLAS_DIRECTUS_URL", DEFAULT_URL).rstrip("/")
    email = os.environ.get("ATLAS_DIRECTUS_ADMIN_EMAIL", DEFAULT_EMAIL)
    password = os.environ.get("ATLAS_DIRECTUS_ADMIN_PASSWORD", DEFAULT_PASSWORD)

    print("=" * 60)
    print("Atlas Directus CRM Setup")
    print("=" * 60)
    print(f"Directus URL : {base_url}")
    print(f"Admin email  : {email}")
    print()

    # 1. Wait for Directus to be healthy
    wait_for_directus(base_url)

    # 2. Log in with admin credentials
    access_token = authenticate(base_url, email, password)

    # 3. Create (or retrieve) a static API token
    api_token = create_api_token(base_url, access_token)

    # 4. Verify the token works
    verify_connection(base_url, api_token)

    # 5. Persist to .env.local
    write_env_local(api_token, base_url)

    print()
    print("Next steps:")
    print("  docker compose restart brain")
    print()
    print("Atlas Brain will now use DirectusCRMProvider for all customer lookups.")
    print(f"Directus Admin UI: {base_url}/admin")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌  Setup failed: {exc}", file=sys.stderr)
        sys.exit(1)
