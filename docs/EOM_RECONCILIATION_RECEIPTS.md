# EOM Calendar execution receipts

Create a private directory under the operator's local state directory:

```bash
export EOM_RECEIPT_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/atlas/eom-calendar-import"
mkdir -p "$EOM_RECEIPT_DIR"
chmod 700 "$EOM_RECEIPT_DIR"
```

Supply it for the production Calendar sequence:

```bash
python -I scripts/import_eom_customers_live.py --dry-run --receipt-dir "$EOM_RECEIPT_DIR"
python -I scripts/import_eom_customers_live.py --receipt-dir "$EOM_RECEIPT_DIR"
python -I scripts/import_eom_customers_live.py --dry-run --receipt-dir "$EOM_RECEIPT_DIR"
```

Live writes reject a missing receipt directory. A supplied directory must
already exist, be owned by the current user, and not be group/world writable.
Each run creates a mode-0600 `.in-progress.json` artifact before the Calendar
runtime, then exclusively publishes one `.exit-N.json` receipt.

Receipted runs require Python's isolated `-I` startup so repository-controlled
`PYTHONPATH` startup hooks and user site packages cannot execute before source
trust is established. They reject tracked or untracked changes, tracked Python
bytes or executable modes that differ from `HEAD`, ignored Python import
shadows, ignored package symlinks beneath the CLI import roots, and bytecode
caches for tracked source. Run from the exact clean checkout whose `HEAD` SHA
belongs in the receipt.

Receipts contain only source bindings, UTC lifecycle timestamps, non-PII
counts, and changed contact UUIDs. They never contain credentials, tokens,
runtime URLs, or customer names, emails, phones, or addresses. Keep the
directory private and do not commit or attach its contents to a public issue.
