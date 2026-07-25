# EOM reconciliation execution receipts

Use a private directory under the operator's local state directory:

```bash
export EOM_RECEIPT_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/atlas/eom-reconciliation"
mkdir -p "$EOM_RECEIPT_DIR"
chmod 700 "$EOM_RECEIPT_DIR"
```

Supply that directory for the complete production sequence:

```bash
python scripts/import_eom_customers_live.py --dry-run --receipt-dir "$EOM_RECEIPT_DIR"
python scripts/import_eom_customers_live.py --receipt-dir "$EOM_RECEIPT_DIR"
python scripts/sync_eom_portal_customers.py --receipt-dir "$EOM_RECEIPT_DIR"
python scripts/sync_eom_portal_customers.py --apply --receipt-dir "$EOM_RECEIPT_DIR"
python scripts/sync_eom_portal_customers.py --receipt-dir "$EOM_RECEIPT_DIR"
```

Write/apply modes reject a missing receipt directory. Dry runs may omit it for
local development, but an operator run should always supply it. Each run first
creates a mode-0600 `.in-progress.json` artifact, then publishes one unique
`.exit-N.json` receipt without overwriting an existing file.

Receipted runs also reject staged or unstaged changes to tracked files. Run only
from the exact clean checkout whose `HEAD` SHA should appear in the receipt;
untracked local files do not alter the executed tracked source and are ignored.

Receipts contain only source bindings, UTC lifecycle timestamps, non-PII
counts, and changed contact UUIDs. They never contain portal credentials,
tokens, runtime URLs, or customer names, emails, phones, or addresses. Keep the
directory private and do not commit or attach its contents to a public issue.
