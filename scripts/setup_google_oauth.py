#!/usr/bin/env python3
"""
Unified Google OAuth Setup for Atlas.

Requests Calendar + Gmail permissions in a single browser flow,
saves tokens to data/google_tokens.json. Run once -- Atlas
auto-persists any token rotations after that.

Prerequisites:
1. Go to Google Cloud Console (https://console.cloud.google.com)
2. Create a project or select existing one
3. Enable "Google Calendar API" and "Gmail API"
4. Go to "Credentials" -> "Create Credentials" -> "OAuth client ID"
5. Choose "Desktop application"
6. Copy the Client ID and Client Secret
7. Under "OAuth consent screen", set publishing status to "Production"
   (prevents 7-day token expiry for "Testing" apps)

Usage:
    python scripts/setup_google_oauth.py

Token file: data/google_tokens.json
"""

import http.server
import json
import os
import sys
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

# Calendar (full read/write) and Gmail (read + send) in one consent
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

REDIRECT_URI = "http://localhost:8085"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

# Resolve token file relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TOKEN_FILE = PROJECT_ROOT / "data" / "google_tokens.json"


def load_existing_tokens() -> dict:
    """Load existing token file if present."""
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def get_auth_url(client_id: str) -> str:
    """Generate the OAuth authorization URL."""
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code(code: str, client_id: str, client_secret: str) -> dict:
    """Exchange authorization code for tokens."""
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = Request(TOKEN_URL, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Capture OAuth callback."""

    auth_code = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            CallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:sans-serif;"
                b"text-align:center;padding:50px'>"
                b"<h1>Authorization Successful!</h1>"
                b"<p>You can close this window.</p>"
                b"</body></html>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Authorization Failed</h1></body></html>")

    def log_message(self, format, *args):
        pass


def save_token_file(client_id: str, client_secret: str, refresh_token: str) -> None:
    """Save tokens to data/google_tokens.json."""
    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "services": {
            "calendar": {"refresh_token": refresh_token},
            "gmail": {"refresh_token": refresh_token},
        },
        "scopes": SCOPES,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)
    os.chmod(TOKEN_FILE, 0o600)


def main():
    print("=" * 60)
    print("  Atlas -- Google OAuth Setup (Calendar + Gmail)")
    print("=" * 60)
    print()

    # Check for existing token file
    existing = load_existing_tokens()
    existing_client_id = existing.get("client_id", "")

    if existing_client_id:
        print(f"Existing token file found: {TOKEN_FILE}")
        print(f"Client ID: {existing_client_id[:30]}...")
        print()
        reuse = input("Reuse existing Client ID/Secret? [Y/n]: ").strip().lower()
        if reuse in ("", "y", "yes"):
            client_id = existing.get("client_id", "")
            client_secret = existing.get("client_secret", "")
            if client_id and client_secret:
                print(f"Using existing credentials.")
                print()
            else:
                print("Existing file missing credentials, please enter them.")
                existing_client_id = ""

    if not existing_client_id:
        print("Enter your Google OAuth credentials.")
        print("(From Google Cloud Console -> Credentials -> OAuth 2.0 Client)")
        print()
        client_id = input("Client ID: ").strip()
        if not client_id:
            print("Error: Client ID is required")
            sys.exit(1)
        client_secret = input("Client Secret: ").strip()
        if not client_secret:
            print("Error: Client Secret is required")
            sys.exit(1)

    print()
    print("Opening browser for Google authorization...")
    print("You will be asked to grant Calendar + Gmail access.")
    print()

    auth_url = get_auth_url(client_id)
    print("If browser does not open, visit this URL:")
    print(auth_url)
    print()

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    print("Waiting for callback on localhost:8085...")
    server = http.server.HTTPServer(("localhost", 8085), CallbackHandler)
    server.handle_request()

    if not CallbackHandler.auth_code:
        print("Error: No authorization code received")
        sys.exit(1)

    print()
    print("Exchanging code for tokens...")

    try:
        tokens = exchange_code(CallbackHandler.auth_code, client_id, client_secret)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        print("Error: No refresh token received.")
        print("This usually means the app is in 'Testing' mode.")
        print("Go to Google Cloud Console -> OAuth consent screen -> Publish.")
        print()
        print("Response:", json.dumps(tokens, indent=2))
        sys.exit(1)

    # Save to token file
    save_token_file(client_id, client_secret, refresh_token)

    print()
    print("=" * 60)
    print("  SUCCESS!")
    print("=" * 60)
    print()
    print(f"Token file saved: {TOKEN_FILE}")
    print(f"  Calendar: configured")
    print(f"  Gmail:    configured")
    print()
    print("Atlas will auto-persist any future token rotations.")
    print("You should not need to run this script again.")
    print()
    print("Optional: add these to .env as backup (not required):")
    print(f"  ATLAS_TOOLS_CALENDAR_ENABLED=true")
    print(f"  ATLAS_TOOLS_CALENDAR_CLIENT_ID={client_id}")
    print(f"  ATLAS_TOOLS_CALENDAR_CLIENT_SECRET={client_secret}")
    print(f"  ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN={refresh_token}")
    print(f"  ATLAS_TOOLS_GMAIL_ENABLED=true")
    print(f"  ATLAS_TOOLS_GMAIL_CLIENT_ID={client_id}")
    print(f"  ATLAS_TOOLS_GMAIL_CLIENT_SECRET={client_secret}")
    print(f"  ATLAS_TOOLS_GMAIL_REFRESH_TOKEN={refresh_token}")
    print()


if __name__ == "__main__":
    main()
