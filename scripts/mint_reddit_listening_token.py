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
# Each source is a COMPLETE (client_id, client_secret) pair, tried in order:
# the listening namespace first, then the existing B2B scraper app so a
# second app need not be registered. Resolution NEVER mixes an id from one
# app with a secret from another.
CRED_SOURCES = (
    ("ATLAS_REDDIT_CLIENT_ID", "ATLAS_REDDIT_CLIENT_SECRET"),
    ("ATLAS_B2B_SCRAPE_REDDIT_CLIENT_ID", "ATLAS_B2B_SCRAPE_REDDIT_CLIENT_SECRET"),
)


class MintConfigError(ValueError):
    """The mint is misconfigured (missing creds/username). Fail closed."""


def load_env(path: str = ".env") -> dict[str, str]:
    """Parse a dotenv file into a dict. Missing or unreadable file -> {}."""
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def build_cred_source(env: dict[str, str], environ: dict[str, str]) -> dict[str, str]:
    """Merge dotenv values with exported shell ATLAS_* vars. The shell wins
    (an explicit export overrides .env), matching pydantic Settings."""
    source = dict(env)
    source.update({k: v for k, v in environ.items() if k.startswith("ATLAS_")})
    return source


def resolve_credentials(
    env: dict[str, str], *, client_id: str = "", client_secret: str = ""
) -> tuple[str, str]:
    """Resolve a COMPLETE (client_id, client_secret) pair. Explicit args win
    (both required together); otherwise the first source that has BOTH set --
    never an id from one app paired with a secret from another."""
    cid, csec = client_id.strip(), client_secret.strip()
    if cid and csec:
        return cid, csec
    if cid or csec:
        raise MintConfigError(
            "pass BOTH --client-id and --client-secret together, or neither"
        )
    for id_key, secret_key in CRED_SOURCES:
        pair_id = env.get(id_key, "").strip()
        pair_secret = env.get(secret_key, "").strip()
        if pair_id and pair_secret:
            return pair_id, pair_secret
    raise MintConfigError(
        "no complete client_id/secret pair found; set ATLAS_REDDIT_CLIENT_ID/_SECRET "
        "(or the B2B scraper equivalents) in .env or the shell, or pass "
        "--client-id/--client-secret together"
    )


def parse_redirect_params(request_line_data: str) -> dict[str, str]:
    """Extract the query params from the raw HTTP request line Reddit's
    redirect delivers (``GET /?code=..&state=.. HTTP/1.1``). Raises
    ValueError on a malformed line rather than IndexError."""
    parts = request_line_data.split(" ")
    if len(parts) < 2 or "?" not in parts[1]:
        raise ValueError("malformed redirect request line: no query params")
    query = parts[1].split("?", 1)[1]
    params: dict[str, str] = {}
    for pair in query.split("&"):
        key, _, value = pair.partition("=")
        if key:
            params[key] = value
    return params


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
    # Honor both .env and exported shell ATLAS_* vars (shell wins).
    source = build_cred_source(load_env(args.env_file), dict(os.environ))
    try:
        client_id, client_secret = resolve_credentials(
            source, client_id=args.client_id, client_secret=args.client_secret
        )
    except MintConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Reuse the production username validator (rejects trailing newlines and
    # other invalid User-Agent shapes) instead of trusting --username raw.
    from atlas_reddit.reddit_client import RedditAuthError, build_user_agent

    try:
        user_agent = build_user_agent(args.username)
    except RedditAuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    import praw  # lazy: only the live mint needs it

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        user_agent=user_agent,
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
        conn, _addr = server.accept()
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
