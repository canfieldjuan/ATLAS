#!/usr/bin/env python3
"""One-time: mint the Reddit Listening scoped refresh token.

Runs PRAW's documented authorization-code flow with scopes exactly
``identity``/``history``/``read`` and prints the refresh token plus the
``.env`` block to add. Reuses an existing Reddit app's client id/secret:
``ATLAS_REDDIT_CLIENT_ID``/``_SECRET`` if set, else the B2B scraper's
``ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID``/``_SECRET`` (so a second app does not
have to be registered).

Prereq: the Reddit app (reddit.com/prefs/apps) must have redirect URI
``http://localhost:8080``. Usage:

    .venv/bin/python scripts/mint_reddit_listening_token.py --username YOUR_REDDIT_USERNAME

See docs/REDDIT_LISTENING_SETUP_RUNBOOK.md. The scopes are read-only; the
listening tool refuses to run with anything broader.
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys

REDIRECT_URI = "http://localhost:8080"
SCOPES = ["identity", "history", "read"]
# Precedence for the app credentials: the listening namespace first, then
# the existing B2B scraper app so a second app need not be registered.
CLIENT_ID_KEYS = ("ATLAS_REDDIT_CLIENT_ID", "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID")
CLIENT_SECRET_KEYS = ("ATLAS_REDDIT_CLIENT_SECRET", "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET")


class MintConfigError(ValueError):
    """The mint is misconfigured (missing creds/username). Fail closed."""


def load_env(path: str = ".env") -> dict[str, str]:
    """Parse a dotenv file into a dict. Missing file -> empty dict."""
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def _first_set(env: dict[str, str], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = env.get(key, "").strip()
        if value:
            return value
    return ""


def resolve_credentials(
    env: dict[str, str], *, client_id: str = "", client_secret: str = ""
) -> tuple[str, str]:
    """Resolve (client_id, client_secret): explicit args win, else the
    listening keys, else the B2B scraper keys. Fail closed if unresolved."""
    resolved_id = client_id.strip() or _first_set(env, CLIENT_ID_KEYS)
    resolved_secret = client_secret.strip() or _first_set(env, CLIENT_SECRET_KEYS)
    if not resolved_id or not resolved_secret:
        raise MintConfigError(
            "no client_id/secret found; pass --client-id/--client-secret or set "
            "ATLAS_REDDIT_CLIENT_ID/_SECRET (or the B2B scraper equivalents) in .env"
        )
    return resolved_id, resolved_secret


def parse_redirect_params(request_line_data: str) -> dict[str, str]:
    """Extract the query params from the raw HTTP request line Reddit's
    redirect delivers (``GET /?code=..&state=.. HTTP/1.1``)."""
    target = request_line_data.split(" ")[1]
    query = target.split("?", 1)[1]
    return dict(pair.split("=", 1) for pair in query.split("&"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--username",
        default=os.environ.get("ATLAS_REDDIT_USERNAME", ""),
        help="Your Reddit username (used only for the descriptive User-Agent).",
    )
    parser.add_argument("--client-id", default="", help="Override the app client id.")
    parser.add_argument("--client-secret", default="", help="Override the app client secret.")
    parser.add_argument("--env-file", default=".env", help="dotenv file to read creds from.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    env = load_env(args.env_file)
    try:
        client_id, client_secret = resolve_credentials(
            env, client_id=args.client_id, client_secret=args.client_secret
        )
    except MintConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.username:
        print(
            "error: pass --username YOUR_REDDIT_USERNAME (needed for the User-Agent)",
            file=sys.stderr,
        )
        return 2

    import praw  # lazy: only the live mint needs it, keeps the module import-light

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        user_agent=f"linux:atlas-reddit-listening:mint (by /u/{args.username})",
    )
    state = str(random.randint(0, 65000))
    url = reddit.auth.url(duration="permanent", scopes=SCOPES, state=state)
    print("\n1) Open this URL in your browser and click ALLOW:\n")
    print("   " + url + "\n")
    print("   (The consent screen must list EXACTLY three permissions: identity, history, read.)")
    print(f"\n2) Waiting for the redirect on {REDIRECT_URI} ...", flush=True)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("localhost", 8080))
    server.listen(1)
    try:
        conn = server.accept()[0]
        data = conn.recv(1024).decode("utf-8")
        conn.send(b"HTTP/1.1 200 OK\r\n\r\nToken minted; you can close this tab.")
        conn.close()
    finally:
        server.close()

    params = parse_redirect_params(data)
    if params.get("state") != state:
        print("error: state mismatch; retry", file=sys.stderr)
        return 2
    if "error" in params:
        print(
            f"error: Reddit returned {params['error']} "
            "(click Allow, and confirm the scope is exactly the three above)",
            file=sys.stderr,
        )
        return 2
    refresh = reddit.auth.authorize(params["code"])
    print("\n=== SUCCESS. Add these to .env ===\n")
    print(f"ATLAS_REDDIT_CLIENT_ID={client_id}")
    print(f"ATLAS_REDDIT_CLIENT_SECRET={client_secret}")
    print(f"ATLAS_REDDIT_REFRESH_TOKEN={refresh}")
    print(f"ATLAS_REDDIT_USERNAME={args.username}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
