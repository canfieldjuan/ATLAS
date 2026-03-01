# Technical Debt Scan Automation

This repository includes an automated debt scanner:

- `scripts/generate_debt_register.py`
- `scripts/run_sprint_debt_scan.sh`

## Manual Run

```bash
./scripts/run_sprint_debt_scan.sh
```

Run with an explicit date:

```bash
./scripts/run_sprint_debt_scan.sh 2026-03-15
```

## Outputs

Each run writes:

- `docs/technical-debt/debt-register-YYYY-MM-DD.csv`
- `docs/technical-debt/debt-register-latest.csv`
- `docs/technical-debt/initial-audit-YYYY-MM-DD.md`
- `docs/technical-debt/initial-audit-latest.md`

## Sprint Automation (cron example)

Run every other Monday at 08:00:

```cron
0 8 * * 1 [ $((($(date +\%s)/86400)%14)) -eq 0 ] && cd /home/juan-canfield/Desktop/Atlas && ./scripts/run_sprint_debt_scan.sh
```

Use `crontab -e` to install the schedule.
