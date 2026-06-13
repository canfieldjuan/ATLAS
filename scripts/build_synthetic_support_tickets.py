#!/usr/bin/env python3
"""Deterministic synthetic support-ticket generator (no LLM).

Generates labeled support-ticket CSV fixtures for the deflection pipeline by
seeded template expansion over a curated intent bank. The same seed produces
byte-identical output, and a ground-truth sidecar records the expected cluster
for every ticket, so tests can assert the pipeline's clustering and report are
provably right instead of "looks plausible".

Messiness injector flags map one-to-one onto filed brittleness issues so the
generator doubles as a regression harness:

    --encoding utf-16 / utf-8-sig   encoding detection (#1455)
    --delimiter ';' or tab          csv.Sniffer determinism (#1459)
    --html-bodies                   HTML leaking into ticket text (#1463)
    --unmapped-body-column          silent row drop on unmapped column (#1457)
    --junk-rows                     banner/blank/short rows from real exports
    --no-labels                     raw untagged export (no pain_category)

Nothing here reads the clock or the network; `created_at` derives from a fixed
--base-date so output is fully reproducible.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Any

CSV_FIELDS = (
    "ticket_id",
    "subject",
    "message",
    "resolution_text",
    "pain_category",
    "created_at",
    "company_name",
)

UNMAPPED_BODY_COLUMN = "customer_msg"
DEFAULT_BASE_DATE = "2026-06-01"
DEFAULT_WINDOW_DAYS = 30
DEFAULT_TICKETS_PER_INTENT = 8
DEFAULT_START_ID = 240001

# --- slot banks (fixed tuples; order is part of determinism) ----------------

_PRODUCTS = ("Acme Portal", "Acme Mobile", "Acme API", "Acme Dashboard")
_ERROR_CODES = ("ERR-401", "ERR-1009", "TIMEOUT-7", "SYNC-22")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
_COMPANIES = (
    "Bluebird Logistics",
    "Hartman Dental",
    "Crestview Realty",
    "Marathon Supply Co",
    "Lakeside Veterinary",
    "Pinnacle Staffing",
)
_AGENT_NAMES = ("Dana", "Marcus", "Priya", "Tom")

# --- intent bank -------------------------------------------------------------
# Each intent: subjects/bodies are templates over the slot banks; resolutions
# is empty for intents that must exercise the no-resolution-evidence lane.

INTENTS: tuple[dict[str, Any], ...] = (
    {
        "key": "password_reset",
        "pain_category": "account_access",
        "question": "How do I reset my password?",
        "subjects": (
            "Can't log in to {product}",
            "Password reset link not arriving",
            "Locked out of my {product} account",
        ),
        "bodies": (
            "I tried resetting my password on {weekday} and the email never "
            "showed up. I have checked spam twice. Can someone reset it for "
            "me? I am locked out of {product}.",
            "The reset link says it expired the moment I click it. This is "
            "the third time this week. How do I get back into {product}?",
            "New employee here, never set a password. The invite from "
            "{product} went to my manager instead of me. How do I set one up?",
        ),
        "resolutions": (
            "Sent a fresh reset link from the admin console and confirmed "
            "the customer logged in. Reset emails were going to a stale "
            "address; updated their contact email.",
            "Cleared the lockout flag, customer set a new password while on "
            "the call. Advised using the Forgot Password link on {product} "
            "next time.",
        ),
    },
    {
        "key": "billing_dispute",
        "pain_category": "billing",
        "question": "Why was I charged twice this month?",
        "subjects": (
            "Double charge on my invoice",
            "Billed twice for {product} this month",
            "Unexpected second charge on card",
        ),
        "bodies": (
            "Our card was charged twice on {weekday} for the same {product} "
            "subscription. Invoice numbers are different but the amounts are "
            "identical. Please refund one of them.",
            "Accounting flagged two charges this cycle. We only have one "
            "seat plan. Can you explain why there is a second charge and "
            "reverse it?",
            "I upgraded mid-month and now I see the old plan AND the new "
            "plan billed in full. That cannot be right.",
        ),
        "resolutions": (
            "Confirmed a duplicate charge caused by a retried payment job; "
            "refunded the second invoice and added a credit memo.",
            "Explained proration: the second line was the upgrade "
            "difference, not a duplicate. Customer agreed after seeing the "
            "itemized breakdown; no refund needed.",
        ),
    },
    {
        "key": "integration_broken",
        "pain_category": "integrations",
        "question": "Why did the integration stop syncing?",
        "subjects": (
            "{product} sync stopped working",
            "Integration failing with {error_code}",
            "Webhooks from {product} stopped on {weekday}",
        ),
        "bodies": (
            "Our CRM sync from {product} has been stuck since {weekday}. "
            "The log shows {error_code} on every run. Nothing changed on "
            "our side.",
            "Webhook deliveries stopped without warning. The endpoint is up "
            "and returns 200 when we test it manually. {error_code} appears "
            "in your status page logs.",
            "After the latest release the {product} integration drops every "
            "third record. Support code {error_code}. We need this fixed "
            "before month-end close.",
        ),
        "resolutions": (),
    },
    {
        "key": "slow_dashboard",
        "pain_category": "performance",
        "question": "Why is the dashboard so slow?",
        "subjects": (
            "{product} dashboard extremely slow",
            "Reports taking minutes to load",
            "Dashboard timeouts every {weekday} morning",
        ),
        "bodies": (
            "The {product} dashboard takes four clicks and a full minute to "
            "find last month's numbers. It used to be instant. Team is "
            "complaining daily.",
            "Every {weekday} morning the main report view spins until it "
            "times out with {error_code}. Reloading sometimes helps, "
            "sometimes not.",
            "Loading the overview page with our full dataset takes over two "
            "minutes. Filtering by date makes it worse, not better.",
        ),
        "resolutions": (),
    },
    {
        "key": "refund_request",
        "pain_category": "billing",
        "question": "How do I request a refund?",
        "subjects": (
            "Requesting a refund for unused seats",
            "Refund for {product} annual plan",
            "Cancel and refund remaining balance",
        ),
        "bodies": (
            "We downsized and have six unused seats on {product}. How do we "
            "get a refund for the remainder of the term?",
            "We bought the annual plan on {weekday} and the project was "
            "cancelled two days later. Requesting a full refund per your "
            "14-day policy.",
            "Our trial converted before we meant it to. Card was charged "
            "this morning. Please reverse it and downgrade us to free.",
        ),
        "resolutions": (
            "Processed a prorated refund for the unused seats and confirmed "
            "the adjusted renewal amount in writing.",
        ),
    },
    {
        "key": "data_export",
        "pain_category": "data_management",
        "question": "How do I export all of my data?",
        "subjects": (
            "Need a full data export from {product}",
            "How to export historical records",
            "Bulk export before contract ends",
        ),
        "bodies": (
            "Our contract ends next quarter and legal needs a complete "
            "export of everything we have in {product}. CSV is fine. What "
            "is the process?",
            "The export button only gives me the current page. I need all "
            "records since 2024, ideally scheduled weekly.",
            "Compliance audit on {weekday}. We need raw exports of audit "
            "logs and user activity from {product}. The docs only cover "
            "report exports.",
        ),
        "resolutions": (
            "Walked the customer through the bulk export tool under "
            "Settings, then enabled the scheduled weekly export to their "
            "SFTP drop.",
        ),
    },
)


# --- generation ---------------------------------------------------------------

def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(_PRODUCTS),
        error_code=rng.choice(_ERROR_CODES),
        weekday=rng.choice(_WEEKDAYS),
        agent=rng.choice(_AGENT_NAMES),
    )


def generate_tickets(
    *,
    seed: int,
    tickets_per_intent: int,
    base_date: date,
    window_days: int,
    start_id: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return (rows, ground_truth). Pure: same arguments -> same output."""
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    ticket_to_intent: dict[str, str] = {}
    clusters: list[dict[str, Any]] = []
    next_id = start_id

    for intent in INTENTS:
        cluster_ids: list[str] = []
        for _ in range(tickets_per_intent):
            ticket_id = str(next_id)
            next_id += 1
            day_offset = rng.randrange(window_days)
            created = base_date - timedelta(days=day_offset)
            hour = rng.randrange(8, 18)
            minute = rng.randrange(0, 60)
            resolutions = intent["resolutions"]
            rows.append(
                {
                    "ticket_id": ticket_id,
                    "subject": _fill(rng.choice(intent["subjects"]), rng),
                    "message": _fill(rng.choice(intent["bodies"]), rng),
                    "resolution_text": (
                        _fill(rng.choice(resolutions), rng) if resolutions else ""
                    ),
                    "pain_category": intent["pain_category"],
                    "created_at": f"{created.isoformat()}T{hour:02d}:{minute:02d}:00Z",
                    "company_name": rng.choice(_COMPANIES),
                }
            )
            ticket_to_intent[ticket_id] = intent["key"]
            cluster_ids.append(ticket_id)
        clusters.append(
            {
                "intent": intent["key"],
                "pain_category": intent["pain_category"],
                "canonical_question": intent["question"],
                "has_resolution": bool(intent["resolutions"]),
                "size": len(cluster_ids),
                "ticket_ids": cluster_ids,
            }
        )

    ground_truth = {
        "seed": seed,
        "base_date": base_date.isoformat(),
        "window_days": window_days,
        "tickets_per_intent": tickets_per_intent,
        "total_tickets": len(rows),
        "ticket_to_intent": ticket_to_intent,
        "expected_clusters": clusters,
    }
    return rows, ground_truth


# --- injectors ---------------------------------------------------------------

def apply_html_bodies(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Wrap every third body in markup (#1463). Index-based, deterministic."""
    out: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        mutated = dict(row)
        if index % 3 == 0:
            mutated["message"] = (
                f"<div><p>{mutated['message']}</p><br><span>Sent from "
                f"webmail</span></div>"
            )
        out.append(mutated)
    return out


def junk_rows_for(field_count: int) -> tuple[list[str], ...]:
    """Rows real exports prepend/append: banner, blank, short (#1459/#1457)."""
    banner = ["Exported by HelpDesk Pro v3.2"] + [""] * (field_count - 1)
    blank = [""] * field_count
    short = ["980001", "FW: see below"]
    return (banner, blank, short)


def render_csv(
    rows: Sequence[Mapping[str, str]],
    *,
    delimiter: str = ",",
    unmapped_body_column: bool = False,
    no_labels: bool = False,
    junk_rows: bool = False,
) -> str:
    header = [
        UNMAPPED_BODY_COLUMN if field == "message" else field
        for field in CSV_FIELDS
    ] if unmapped_body_column else list(CSV_FIELDS)

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n")
    junk = junk_rows_for(len(header)) if junk_rows else ()
    if junk_rows:
        writer.writerow(junk[0])
    writer.writerow(header)
    for row in rows:
        values = [row[field] for field in CSV_FIELDS]
        if no_labels:
            values[CSV_FIELDS.index("pain_category")] = ""
        writer.writerow(values)
    if junk_rows:
        writer.writerow(junk[1])
        writer.writerow(junk[2])
    return buffer.getvalue()


# --- output ---------------------------------------------------------------

def write_outputs(
    rows: Sequence[Mapping[str, str]],
    ground_truth: Mapping[str, Any],
    output_dir: Path,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
    html_bodies: bool = False,
    unmapped_body_column: bool = False,
    no_labels: bool = False,
    junk_rows: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_rows = apply_html_bodies(rows) if html_bodies else list(rows)
    text = render_csv(
        final_rows,
        delimiter=delimiter,
        unmapped_body_column=unmapped_body_column,
        no_labels=no_labels,
        junk_rows=junk_rows,
    )
    csv_path = output_dir / "tickets.csv"
    csv_path.write_bytes(text.encode(encoding))
    truth_path = output_dir / "ground_truth.json"
    truth_path.write_text(
        json.dumps(ground_truth, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "csv": str(csv_path),
        "ground_truth": str(truth_path),
        "rows": len(rows),
        "encoding": encoding,
        "delimiter": delimiter,
        "injectors": {
            "html_bodies": html_bodies,
            "unmapped_body_column": unmapped_body_column,
            "no_labels": no_labels,
            "junk_rows": junk_rows,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--tickets-per-intent", type=int, default=DEFAULT_TICKETS_PER_INTENT
    )
    parser.add_argument("--base-date", default=DEFAULT_BASE_DATE)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--start-id", type=int, default=DEFAULT_START_ID)
    parser.add_argument(
        "--encoding",
        default="utf-8",
        choices=("utf-8", "utf-8-sig", "utf-16", "latin-1"),
        help="file encoding for tickets.csv (#1455 regression input)",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter, e.g. ';' or '\\t' (#1459 regression input)",
    )
    parser.add_argument("--html-bodies", action="store_true")
    parser.add_argument("--unmapped-body-column", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--junk-rows", action="store_true")
    parser.add_argument("--json", action="store_true", help="print manifest JSON")
    args = parser.parse_args(argv)
    if args.tickets_per_intent < 1:
        parser.error("--tickets-per-intent must be >= 1")
    if args.window_days < 1:
        parser.error("--window-days must be >= 1")
    delimiter = "\t" if args.delimiter == "\\t" else args.delimiter
    if len(delimiter) != 1:
        parser.error("--delimiter must be a single character")
    args.delimiter = delimiter
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        base = date.fromisoformat(args.base_date)
    except ValueError:
        print(f"invalid --base-date: {args.base_date!r}", file=sys.stderr)
        return 2
    rows, ground_truth = generate_tickets(
        seed=args.seed,
        tickets_per_intent=args.tickets_per_intent,
        base_date=base,
        window_days=args.window_days,
        start_id=args.start_id,
    )
    manifest = write_outputs(
        rows,
        ground_truth,
        args.output_dir,
        encoding=args.encoding,
        delimiter=args.delimiter,
        html_bodies=args.html_bodies,
        unmapped_body_column=args.unmapped_body_column,
        no_labels=args.no_labels,
        junk_rows=args.junk_rows,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            f"Wrote {manifest['rows']} synthetic tickets to {manifest['csv']} "
            f"(ground truth: {manifest['ground_truth']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
