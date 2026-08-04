# EOM reconciliation execution receipts

The live EOM Calendar import and portal customer sync write private JSON
receipts for production write/apply runs.

Recommended local operator directory:

```bash
mkdir -p ~/.local/state/atlas/eom-receipts
chmod 700 ~/.local/state/atlas/eom-receipts
```

Write/apply runs require that directory explicitly:

```bash
python scripts/import_eom_customers_live.py \
  --receipt-dir ~/.local/state/atlas/eom-receipts

python scripts/sync_eom_portal_customers.py \
  --apply \
  --receipt-dir ~/.local/state/atlas/eom-receipts
```

Dry runs may omit `--receipt-dir`. If supplied, a dry run also writes a receipt.

Receipts are private operator evidence. They record only allowlisted non-PII
fields: schema version, receipt id, tool, mode, UTC start/end, Git SHA, script
hash, exit code, outcome counts, demotion totals where applicable, changed
contact UUIDs, and whether a mutation was interrupted with an indeterminate
result. They do not record credentials, tokens, customer names, emails, phones,
addresses, portal URLs, or receipt-directory paths.
