#!/usr/bin/env python3
"""
Google Calendar OAuth Setup Script

Run this script to get a refresh token for the Atlas calendar tool.

Prerequisites:
1. Go to Google Cloud Console (https://console.cloud.google.com)
2. Create a project or select existing one
3. Enable "Google Calendar API"
4. Go to "Credentials" -> "Create Credentials" -> "OAuth client ID"
5. Choose "Desktop application"
6. Download or copy the Client ID and Client Secret

Usage:
    python scripts/setup_google_calendar.py

Then add to your .env or environment:
    ATLAS_TOOLS_CALENDAR_ENABLED=true
    ATLAS_TOOLS_CALENDAR_CLIENT_ID=<your_client_id>
    ATLAS_TOOLS_CALENDAR_CLIENT_SECRET=<your_client_secret>
    ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN=<output_from_this_script>
"""

import http.server
import json
import urllib.parse
import webbrowser
from urllib.request import Request, urlopen


# OAuth configuration
SCOPES = ["https://www.googleapis.com/auth/calendar"]  # Full read/write access for booking
REDIRECT_URI = "http://localhost:8085"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def get_auth_url(client_id: str) -> str:
    """Generate the OAuth authorization URL."""
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",  # Force consent to get refresh token
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(code: str, client_id: str, client_secret: str) -> dict:
    """Exchange authorization code for tokens."""
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    encoded_data = urllib.parse.urlencode(data).encode("utf-8")
    req = Request(TOKEN_URL, data=encoded_data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler to capture OAuth callback."""

    auth_code = None

    def do_GET(self):
        """Handle the OAuth callback."""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            OAuthCallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""
                <html>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>Authorization Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                </body>
                </html>
            """)
        else:
            error = params.get("error", ["Unknown error"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"""
                <html>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>Authorization Failed</h1>
                    <p>Error: {error}</p>
                </body>
                </html>
            """.encode())

    def log_message(self, format, *args):
        """Suppress HTTP logging."""
        pass


def main():
    print("=" * 60)
    print("Google Calendar OAuth Setup for Atlas")
    print("=" * 60)
    print()

    # Get credentials from user
    client_id = input("Enter your Google OAuth Client ID: ").strip()
    if not client_id:
        print("Error: Client ID is required")
        return

    client_secret = input("Enter your Google OAuth Client Secret: ").strip()
    if not client_secret:
        print("Error: Client Secret is required")
        return

    print()
    print("Opening browser for Google authorization...")
    print("(If browser doesn't open, copy this URL manually)")
    print()

    auth_url = get_auth_url(client_id)
    print(auth_url)
    print()

    # Try to open browser
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    # Start local server to capture callback
    print("Waiting for authorization callback on localhost:8085...")
    server = http.server.HTTPServer(("localhost", 8085), OAuthCallbackHandler)
    server.handle_request()  # Handle single request

    if not OAuthCallbackHandler.auth_code:
        print("Error: No authorization code received")
        return

    print()
    print("Authorization code received, exchanging for tokens...")

    try:
        tokens = exchange_code_for_tokens(
            OAuthCallbackHandler.auth_code,
            client_id,
            client_secret,
        )
    except Exception as e:
        print(f"Error exchanging code: {e}")
        return

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        print("Error: No refresh token in response")
        print("Response:", json.dumps(tokens, indent=2))
        return

    print()
    print("=" * 60)
    print("SUCCESS! Add these to your environment:")
    print("=" * 60)
    print()
    print(f"ATLAS_TOOLS_CALENDAR_ENABLED=true")
    print(f"ATLAS_TOOLS_CALENDAR_CLIENT_ID={client_id}")
    print(f"ATLAS_TOOLS_CALENDAR_CLIENT_SECRET={client_secret}")
    print(f"ATLAS_TOOLS_CALENDAR_REFRESH_TOKEN={refresh_token}")
    print()
    print("Or add to your .env file in the Atlas directory.")
    print()


if __name__ == "__main__":
    main()
