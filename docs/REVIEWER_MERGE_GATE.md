# Reviewer merge gate (`claude-review`)

The builder merge condition (AGENTS.md 3c.1, point 8) requires "all
review/reconciliation gates clean." There are two review gates, and both must
be green before the active builder merges:

| Gate | Reviewer | Machine signal |
|---|---|---|
| `live-reconciliation` | Codex / bots | Existing required check; reds while unaccounted bot threads are open (`scripts/check_ai_reconciliation_live.py`). |
| `claude-review` | Claude Code (reviewer session) | Per-SHA commit status set by `scripts/set_claude_review_status.py`. |

This document defines the `claude-review` half.

## Why it exists

The Claude Code reviewer session operates as the operator's GitHub identity, so
its review is prose comments, not a distinct machine-checkable signal the
builder can gate on. `claude-review` promotes the reviewer's verdict to a
per-SHA commit status so "Claude reviewed and it is clean" becomes a real gate
next to `live-reconciliation`, satisfying the operator's two-review merge
requirement (both Codex and Claude must be clean, not either alone).

## Status semantics

The status `context` is always `claude-review`. It attaches to a specific head
SHA:

- `success` — the reviewer reviewed **this exact head SHA** and found no
  BLOCKER (an LGTM, or only non-blocking MAJOR/NIT notes per the AGENTS.md
  severity table).
- `failure` — the reviewer found a BLOCKER open at this head SHA.
- `pending` — a review of this head SHA is in progress.
- absent — never reviewed at this SHA.

A re-push produces a new head SHA with no `claude-review` status. So a required
`claude-review` gate is **fail-closed by absence**: it stays not-green until the
reviewer re-reviews the new head and re-sets the status. A re-push therefore
invalidates the prior Claude review, exactly as it invalidates prior CI.

## Setting it

The reviewer session, after completing a review of a head SHA, runs:

```
python scripts/set_claude_review_status.py --repo <owner/name> --sha <headSHA> \
    --state success|failure|pending --pr <n> [--description "..."]
```

`--dry-run` prints the `gh api` argv without calling GitHub. The tool only ever
sets the `claude-review` context; it cannot spoof another check.

## Trust boundary: it is forgeable until the reviewer has a distinct identity

`claude-review` is a plain commit status. GitHub does not permission statuses
per-context, so **any token with `status:write` on the repo can publish
`claude-review=success`**, including the builder if it runs under the same
GitHub identity as the reviewer (today both are the operator's account). So as
built this is a coordination-and-audit signal that keeps an honest builder from
merging before review, NOT a defense against a builder that forges the status.

It becomes a real, unforgeable gate only when it is posted from a reviewer
identity the builder does not have: a distinct GitHub App or bot token that
holds `status:write` while the builder's token does not. Until that exists, do
not treat a green `claude-review` as proof the reviewer actually ran.

## What is still the operator's to flip

This slice wires the **signal** only. These steps remain operator-owned and are
deliberately not done here, in order:

1. **Provision a distinct reviewer identity** (a GitHub App or bot token) that
   is the only actor with `status:write` for `claude-review`, so the builder
   cannot forge it. This is the prerequisite for the gate to mean anything once
   merge is automated.
2. **Make `claude-review` a required status check** in branch protection, so it
   actually blocks merge (until then it is advisory: visible but non-gating).
3. **Grant the active builder standing merge authorization** for the arc, and
   record the authorization source plus the scheduled-ready-only merge
   condition per AGENTS.md 3c.1 point 1. Do not do this before step 1 if the
   builder shares the reviewer's credentials.

Until these are done, nothing here changes who can merge. The reviewer publishes
a machine-readable verdict; the operator decides whether it gates and who acts
on it.
