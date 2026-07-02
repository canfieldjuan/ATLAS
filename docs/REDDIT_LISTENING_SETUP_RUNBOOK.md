# Reddit Listening Setup Runbook (atlas_reddit, #1934 S4)

One-time operator setup for the read-only Reddit listening tool. Everything
here was doc-verified 2026-07-02 against the PRAW authentication docs, the
PRAW refresh-token tutorial, and the reddit-archive OAuth2/API wikis.

Compliance frame (product requirements, not polish):

- Scopes are exactly `identity`, `history`, `read`. The tool refuses to run
  with anything more -- including the `*` wildcard a password-grant token
  carries. Password-grant scope restriction is not documented anywhere, so
  the scoped refresh token below is the only documented way to hold exactly
  these three.
- No submit/edit/vote scopes, no write endpoints. A static test greps the
  package for Reddit write-API usage.
- Descriptive User-Agent: `linux:atlas-reddit-listening:v<version> (by
  /u/<username>)` (Reddit-required format; never a default or spoofed UA).
- OAuth clients get 60 requests/minute; PRAW respects the X-Ratelimit-*
  headers and the poller adds `ATLAS_REDDIT_PACE_SECONDS` (default 2s)
  between subreddit fetches on top.

## 1. Register a script app (once)

1. Log in to the Reddit account the tool will act as.
2. https://www.reddit.com/prefs/apps -> "create another app...".
3. Type: **script**. Name: `atlas-reddit-listening`.
4. Redirect URI: `http://localhost:8080` (needed only for the one-time
   token mint below).
5. Note the client id (under the app name) and the secret.

## 2. Mint the scoped refresh token (once)

Run this locally (requires `pip install praw`); it is PRAW's documented
authorization-code flow with our scopes preset:

```python
import random
import socket
import praw

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="http://localhost:8080",
    user_agent="linux:atlas-reddit-listening:mint (by /u/YOUR_USERNAME)",
)
state = str(random.randint(0, 65000))
print(reddit.auth.url(duration="permanent", scopes=["identity", "history", "read"], state=state))

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("localhost", 8080))
server.listen(1)
client = server.accept()[0]
data = client.recv(1024).decode("utf-8")
client.send(b"HTTP/1.1 200 OK\r\n\r\nToken minted; close this tab.")
client.close()
server.close()

params = dict(pair.split("=") for pair in data.split(" ")[1].split("?")[1].split("&"))
assert params["state"] == state, "state mismatch; retry"
print("refresh token:", reddit.auth.authorize(params["code"]))
```

Open the printed URL in a browser, approve, and copy the printed refresh
token. The consent screen must list exactly three permissions; if it lists
more, the scopes argument was altered -- abort and retry.

## 3. Configure the environment

Add to `.env` (never commit real values; these are placeholders):

```bash
ATLAS_REDDIT_CLIENT_ID=your_client_id
ATLAS_REDDIT_CLIENT_SECRET=your_client_secret
ATLAS_REDDIT_REFRESH_TOKEN=your_refresh_token
ATLAS_REDDIT_USERNAME=your_reddit_username
# Optional tuning:
# ATLAS_REDDIT_FRESHNESS_HOURS=48
# ATLAS_REDDIT_PER_SUBREDDIT_LIMIT=50
# ATLAS_REDDIT_PACE_SECONDS=2.0
# ATLAS_REDDIT_POLL_MIN_SCORE=0.5
```

## 4. Prepare the watchlist and run

```bash
mkdir -p data/atlas_reddit
cp atlas_reddit/watchlist.sample.toml data/atlas_reddit/watchlist.toml
# edit the watchlist, then:
python -m atlas_reddit poll     # one polite read-only pass
python -m atlas_reddit purge    # deletion-compliance pass FIRST...
python -m atlas_reddit digest   # ...so the digest never surfaces content
                                # deleted on Reddit since the last purge
```

Deletion compliance: run `purge` at least every 48 hours while any stored
content exists. It re-checks every stored post and reply in batched reads
(100 per request); content that is deleted/removed/missing on Reddit is
dropped locally and recorded in `purge_log` with the detection reason.

The first `poll` run proves the auth boundary end to end: if the token
carries any scope beyond identity/history/read (or the wildcard `*`), the
tool refuses to start with a `RedditAuthError` naming the excess scopes.

## 5. Verify the compliance posture (any time)

```bash
# The token's grants (requires creds configured):
python - <<'PY'
from atlas_reddit.config import RedditListeningSettings
from atlas_reddit.reddit_client import PrawListingSource
print(sorted(PrawListingSource(RedditListeningSettings()).granted_scopes()))
PY

# The static no-write probe and the full suite:
python -m pytest tests/test_atlas_reddit_poller.py -q
```

Revoking access later: https://www.reddit.com/prefs/apps -> revoke, which
also invalidates the refresh token and all related access tokens.
