# EOM Calendar execution receipts

Create a private directory under the operator's local state directory:

```bash
export EOM_RECEIPT_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/atlas/eom-calendar-import"
mkdir -p "$EOM_RECEIPT_DIR"
chmod 700 "$EOM_RECEIPT_DIR"
```

Supply it for the production Calendar sequence:

```bash
set -o pipefail
git show HEAD:scripts/eom_execution_receipt.py |
  python -I - --launch-reviewed scripts/import_eom_customers_live.py \
    --dry-run --receipt-dir "$EOM_RECEIPT_DIR"
git show HEAD:scripts/eom_execution_receipt.py |
  python -I - --launch-reviewed scripts/import_eom_customers_live.py \
    --receipt-dir "$EOM_RECEIPT_DIR"
git show HEAD:scripts/eom_execution_receipt.py |
  python -I - --launch-reviewed scripts/import_eom_customers_live.py \
    --dry-run --receipt-dir "$EOM_RECEIPT_DIR"
```

Live writes reject a missing receipt directory. A supplied directory must
already exist, be owned by the current user, and not be group/world writable.
Each run creates a mode-0600 `.in-progress.json` artifact before the Calendar
runtime, then exclusively publishes one `.exit-N.json` receipt.

Receipted runs pipe the receipt launcher from the reviewed `HEAD` object into
Python's isolated `-I` startup. The launcher authenticates the checkout before
loading the Calendar entrypoint from the same exact Git SHA, so the mutable
worktree entrypoint never executes first. Direct
`python -I scripts/import_eom_customers_live.py --receipt-dir ...` invocation
is rejected.

The launcher rejects tracked or untracked changes, tracked Python bytes or
executable modes that differ from `HEAD`, ignored Python import shadows,
ignored package and module-file symlinks beneath the CLI import roots, and
bytecode caches for tracked source. Run from the exact clean checkout whose
`HEAD` SHA belongs in the receipt.

Receipts contain only source bindings, UTC lifecycle timestamps, non-PII
counts, and changed contact UUIDs. They never contain credentials, tokens,
runtime URLs, or customer names, emails, phones, or addresses. Keep the
directory private and do not commit or attach its contents to a public issue.
