#!/usr/bin/env python3
"""Alert when a write reaches EOM customer data without the canonical boundary.

Slice 0F of the canonical write boundary (website #113, under #107 / #105).

0C made cross-system customer creation durable and retryable. Nothing watched
it, so a bypass -- a writer reaching the database without going through the
domain tier, or a tracker customer minted with no Atlas contact -- stayed silent
until somebody happened to run an audit by hand. That is how the original defect
survived long enough to need a backfill.

This runs on a timer and turns each of those into an alert.

Deliberately standalone, in the same spirit as atlas-api-healthcheck.sh: it
shells out to `psql` and the Render CLI rather than importing the application,
so it keeps working when the application does not. It is the monitor; it must
not share a failure mode with the thing it monitors.

Not being able to READ a source is itself an alert. A monitor that goes quiet
when it loses its data source is worse than no monitor, because it reports clean
while seeing nothing -- the exact silent-failure class this slice exists to
close.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

EOM_TENANT = "effingham_maids"

# The event types the operator mutation boundary writes
# (atlas_brain/services/eom_crm_mutations.py::EOM_OPERATOR_CONTACT_EVENT_TYPES).
# Correlating on these specifically, not on "any lifecycle row": an EOM lead
# already carries a lead_created row from migration 351, so a bypass that adds
# operator provenance to an existing contact would be excluded by a bare
# NOT EXISTS and the audit would report zero.
OPERATOR_EVENT_TYPES = ("contact_created", "contact_updated")

# A measured breach, distinct from Python's exit 1 for an uncaught exception,
# so the unit file can accept one without masking the other.
EXIT_BREACH = 2

# Every `source` value an EOM contact writer is allowed to emit. Derived from
# the 2026-08-05 code sweep of every create path, not from whatever happens to
# be in the table -- an allowlist built from observed data would bless a bypass
# that had already run.
KNOWN_EOM_SOURCES = (
    "calendar_import",
    "email_backfill",
    "web",
    "manual",
    "manual_invoice_setup",
    "phone_call",
    "portal_sync",
    "sms",
    "booking",
)

# A reservation pending right after an Atlas failure is normal and retryable.
# One still pending an hour later means nobody is coming back for it.
STALE_RESERVATION_MINUTES = 60

DEFAULT_STATE_DIR = Path(
    os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state"))
) / "eom-write-boundary-audit"

# Deliberately NO default. On ntfy.sh the topic name IS the credential: anyone
# who knows it can read the alerts or forge them. This repository is public, so
# the topic is supplied at deploy time via EOM_AUDIT_NTFY_TOPIC and never
# committed. A blank topic is refused by validate_settings rather than silently
# publishing nowhere.
DEFAULT_NTFY_TOPIC = ""
DEFAULT_NTFY_URL = "https://ntfy.sh"
DEFAULT_REALERT_EVERY = 24  # hourly cadence -> re-alert once a day while open

TRACKER_DB_ID = "dpg-d723r3buibrs739nnpg0-a"


@dataclass(frozen=True)
class Signal:
    """One boundary invariant, and what was actually measured for it."""

    name: str
    summary: str
    threshold: int
    count: int | None = None
    error: str | None = None

    @property
    def unmeasured(self) -> bool:
        return self.count is None

    @property
    def breached(self) -> bool:
        # Unmeasured counts as breached: see the module docstring.
        if self.unmeasured:
            return True
        return self.count > self.threshold

    def describe(self) -> str:
        if self.unmeasured:
            return f"{self.name}: COULD NOT MEASURE ({self.error})"
        return f"{self.name}: {self.count} (allowed {self.threshold}) -- {self.summary}"


@dataclass
class AuditResult:
    signals: list[Signal] = field(default_factory=list)

    @property
    def breaches(self) -> list[Signal]:
        return [signal for signal in self.signals if signal.breached]

    @property
    def ok(self) -> bool:
        return not self.breaches

    def report(self) -> str:
        return "\n".join(signal.describe() for signal in self.signals)


# --- queries -----------------------------------------------------------------


def _atlas_sql() -> str:
    sources = ", ".join(f"'{value}'" for value in KNOWN_EOM_SOURCES)
    operator_events = ", ".join(f"'{value}'" for value in OPERATOR_EVENT_TYPES)
    return f"""
SELECT
  (SELECT COUNT(*) FROM contacts
     WHERE business_context_id = '{EOM_TENANT}'
       AND (source IS NULL OR source NOT IN ({sources}))),
  (SELECT COUNT(*) FROM contacts WHERE business_context_id IS NULL),
  (SELECT COUNT(*) FROM contacts c
     WHERE c.business_context_id = '{EOM_TENANT}'
       AND c.metadata ? 'eom_operator_contact_sources'
       AND NOT EXISTS (
         SELECT 1 FROM eom_lead_lifecycle_events e
          WHERE e.contact_id = c.id
            AND e.event_type IN ({operator_events})))
"""


def _tracker_sql() -> str:
    return (
        "SELECT (SELECT COUNT(*) FROM customers WHERE atlas_contact_id IS NULL), "
        "(SELECT COUNT(*) FROM eom_customer_atlas_reservations "
        f"WHERE state = 'pending' AND updated_at < NOW() - INTERVAL '{STALE_RESERVATION_MINUTES} minutes')"
    )


def _run(command: Sequence[str], timeout: int = 90) -> tuple[str | None, str | None]:
    if not command:
        raise ValueError("_run needs a command to execute")
    # Named once, without indexing: an empty command is a programmer error and
    # already raised above, so the error strings below cannot be the thing that
    # blows up while reporting why something else did.
    executable = next(iter(command))
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError as exc:
        return None, f"{executable} not found: {exc}"
    except subprocess.TimeoutExpired:
        return None, f"{executable} timed out after {timeout}s"
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else "no output"
        return None, f"{executable} exited {proc.returncode}: {detail}"
    return proc.stdout, None


def _parse_counts(raw: str, expected: int) -> tuple[list[int] | None, str | None]:
    """Read one delimited row of integers, refusing anything else.

    Strict on purpose: a partially parsed row would silently become a low count,
    which reads as healthy. Unparseable output must surface as unmeasured.
    """
    rows = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if not rows:
        return None, "query produced no output"
    if len(rows) != 1:
        # Anything beyond a single row is ambiguous: taking the first would let
        # "0|0" followed by "1|1" report clean, and a trailing notice would be
        # silently discarded. Which row is authoritative is not for a monitor to
        # decide, so it refuses instead.
        return None, f"expected exactly one output row, got {len(rows)}: {rows!r}"
    # Unpacked, not indexed: the single-row invariant is enforced by the check
    # above and restated by the binding itself.
    (row,) = rows
    parts = [part.strip() for part in row.split("|")]
    if len(parts) != expected:
        return None, f"expected {expected} counts, got {len(parts)} in {row!r}"
    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        return None, f"non-integer count in row {row!r}: {exc}"
    if any(count < 0 for count in counts):
        # COUNT(*) cannot be negative; a negative here means the output is not
        # the shape this reader was written for.
        return None, f"impossible negative count in {row!r}"
    return counts, None


def query_atlas(psql_bin: str, dsn: str) -> tuple[list[int] | None, str | None]:
    raw, error = _run(
        [psql_bin, dsn, "-A", "-t", "-F", "|", "-v", "ON_ERROR_STOP=1", "-c", _atlas_sql()]
    )
    if error:
        return None, error
    return _parse_counts(raw, 3)


def query_tracker(render_bin: str, database_id: str) -> tuple[list[int] | None, str | None]:
    raw, error = _run(
        [
            render_bin,
            "psql",
            database_id,
            "--confirm",
            "-o",
            "text",
            "-c",
            _tracker_sql(),
            "--",
            "-A",
            "-t",
            "-F",
            "|",
        ],
        timeout=120,
    )
    if error:
        return None, error
    return _parse_counts(raw, 2)


def build_signals(
    atlas: tuple[list[int] | None, str | None],
    tracker: tuple[list[int] | None, str | None],
) -> AuditResult:
    atlas_counts, atlas_error = atlas
    tracker_counts, tracker_error = tracker

    def atlas_at(index: int) -> tuple[int | None, str | None]:
        if atlas_counts is None:
            return None, f"Atlas unreadable: {atlas_error}"
        return atlas_counts[index], None

    def tracker_at(index: int) -> tuple[int | None, str | None]:
        if tracker_counts is None:
            return None, f"tracker unreadable: {tracker_error}"
        return tracker_counts[index], None

    specs = [
        ("atlas_unknown_source", "EOM contacts whose source no known writer emits", atlas_at(0)),
        ("atlas_null_tenant", "contacts with no business context", atlas_at(1)),
        (
            "atlas_operator_provenance_without_event",
            "operator-written contacts with no lifecycle event (wrote around the domain tier)",
            atlas_at(2),
        ),
        (
            "tracker_unlinked_customers",
            "tracker customers Atlas has never heard of",
            tracker_at(0),
        ),
        (
            "tracker_stale_pending_reservations",
            f"reservations pending over {STALE_RESERVATION_MINUTES}m with nobody retrying",
            tracker_at(1),
        ),
    ]
    return AuditResult(
        signals=[
            Signal(name=name, summary=summary, threshold=0, count=count, error=error)
            for name, summary, (count, error) in specs
        ]
    )


# --- alert state machine -----------------------------------------------------


def _previous_breached(previous: dict) -> set[str] | None:
    """The signal set the last run recorded, or None if it cannot be trusted.

    A state file written before this became set-aware only says "breached", not
    which signals. Returning None there makes the caller treat the current set
    as changed, which alerts rather than assuming continuity -- the safe
    direction for a monitor.
    """
    recorded = previous.get("breached_signals")
    if isinstance(recorded, list):
        return {str(name) for name in recorded}
    if previous.get("breached"):
        return None
    return set()


def decide_alert(
    previous: dict, breached: Sequence[str], realert_every: int
) -> tuple[dict, str | None]:
    """Return the next state and which alert to send, if any.

    Tracks WHICH signals are breached, not merely whether any is. Collapsing
    them to one boolean means a second incident opening while a first is still
    open produces no alert until the re-alert interval -- the monitor would sit
    on a new problem for up to a day because an unrelated one was already
    known.

    Otherwise it mirrors atlas-api-healthcheck.sh: fire on entering breach,
    re-alert every `realert_every` consecutive runs, one recovery notice.
    """
    now = {str(name) for name in breached}
    before = _previous_breached(previous)

    if not now:
        cleared = {"breached_signals": [], "consecutive": 0}
        return cleared, ("recovered" if before is None or before else None)

    if before is not None and not before:
        return {"breached_signals": sorted(now), "consecutive": 1}, "breach"
    if before is None or now != before:
        # A different set of problems than last time: a new one opened, or one
        # cleared while others remain. Either way it is news, and the reminder
        # clock restarts against the current incident.
        return {"breached_signals": sorted(now), "consecutive": 1}, "changed"

    consecutive = int(previous.get("consecutive", 0)) + 1
    state = {"breached_signals": sorted(now), "consecutive": consecutive}
    if realert_every > 0 and consecutive % realert_every == 0:
        return state, "reminder"
    return state, None


def validate_settings(realert_every: int, ntfy_topic: str) -> None:
    """Refuse a configuration that would quietly disable alerting.

    Raises rather than defaulting: a mistyped interval or a blank topic would
    otherwise leave the timer running and reporting success while no alert could
    ever reach anyone -- a monitor that looks healthy and is not.
    """
    if realert_every < 0:
        raise ValueError(f"realert-every must not be negative, got {realert_every}")
    if not ntfy_topic.strip():
        raise ValueError("ntfy topic must not be blank; nothing could be delivered")


def read_state(path: Path) -> tuple[dict, str | None]:
    """Load prior alert state, reporting rather than hiding a bad state file.

    A missing file is the normal first run. A file that exists but cannot be
    read is different: it means the alert cadence has lost its memory, so a
    recovery notice may be skipped and a breach re-alerted. That is safe, but it
    is not silent -- the caller surfaces it.
    """
    if not path.exists():
        return {}, None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, f"alert state unreadable ({exc}); treating this as a first run"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"alert state corrupt ({exc}); treating this as a first run"
    if not isinstance(value, dict):
        return {}, "alert state was not an object; treating this as a first run"
    return value, None


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def publish(ntfy_url: str, topic: str, title: str, body: str, priority: str, tags: str) -> bool:
    """Push one alert. Returns whether it was actually delivered.

    The return value is load-bearing: the caller must not record an alert as
    sent when it was not, or the re-alert interval silently swallows the next
    day of runs.
    """
    output, error = _run(
        [
            "curl", "-fsS", "-m", "10",
            "-H", f"Title: {title}",
            "-H", f"Priority: {priority}",
            "-H", f"Tags: {tags}",
            "-d", body,
            f"{ntfy_url}/{topic}",
        ],
        timeout=20,
    )
    if error:
        print(f"WARNING alert delivery failed: {error}")
        return False
    return True


def main(argv: Sequence[str] | None = None, *, notifier: Callable[..., None] = publish) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dsn", default=os.environ.get("EOM_AUDIT_ATLAS_DSN", "postgresql://atlas:atlas@localhost:5433/atlas"))
    parser.add_argument("--psql-bin", default=os.environ.get("EOM_AUDIT_PSQL_BIN", "psql"))
    parser.add_argument("--render-bin", default=os.environ.get("EOM_AUDIT_RENDER_BIN", str(Path.home() / ".local/bin/render")))
    parser.add_argument("--tracker-db-id", default=os.environ.get("EOM_AUDIT_TRACKER_DB_ID", TRACKER_DB_ID))
    parser.add_argument("--state-dir", default=os.environ.get("EOM_AUDIT_STATE_DIR", str(DEFAULT_STATE_DIR)))
    parser.add_argument("--ntfy-url", default=os.environ.get("EOM_AUDIT_NTFY_URL", DEFAULT_NTFY_URL))
    parser.add_argument("--ntfy-topic", default=os.environ.get("EOM_AUDIT_NTFY_TOPIC", DEFAULT_NTFY_TOPIC))
    parser.add_argument("--realert-every", type=int, default=int(os.environ.get("EOM_AUDIT_REALERT_EVERY", DEFAULT_REALERT_EVERY)))
    parser.add_argument("--no-alert", action="store_true", help="measure and print without notifying")
    args = parser.parse_args(argv)
    validate_settings(args.realert_every, args.ntfy_topic)

    result = build_signals(
        query_atlas(args.psql_bin, args.atlas_dsn),
        query_tracker(args.render_bin, args.tracker_db_id),
    )
    print(result.report())

    state_path = Path(args.state_dir) / "state.json"
    previous, state_warning = read_state(state_path)
    if state_warning:
        print(f"WARNING {state_warning}")
    next_state, alert = decide_alert(
        previous, [signal.name for signal in result.breaches], args.realert_every
    )

    # State advances only once the alert it represents has actually been
    # delivered. Persisting first would let a failed push record the breach as
    # notified and then suppress every run until the re-alert interval comes
    # round -- a monitor that has silently stopped alerting, which is the exact
    # failure this slice exists to make impossible.
    #
    # Leaving the old state on a failed push means the next run recomputes the
    # same transition and tries again. That is not an alert storm: nothing is
    # reaching anyone while delivery is broken.
    # A dry run observes; it must not consume the alert. Leaving `delivered`
    # true here would let `--no-alert` during a breach persist "already
    # notified" and suppress the real alert for the whole re-alert window.
    delivered = alert is None
    if alert and args.no_alert:
        print(f"WARNING --no-alert: {alert} not sent and state left unchanged")
    if alert and not args.no_alert:
        if alert == "recovered":
            delivered = notifier(
                args.ntfy_url, args.ntfy_topic,
                "EOM write boundary clean",
                "Every write-boundary signal is back to zero.",
                "default", "white_check_mark",
            )
        else:
            detail = "\n".join(signal.describe() for signal in result.breaches)
            headline = (
                "EOM write boundary: signals changed"
                if alert == "changed"
                else "EOM write boundary breached"
            )
            delivered = notifier(
                args.ntfy_url, args.ntfy_topic,
                headline,
                f"{detail}\n\n(run #{next_state['consecutive']} in breach)",
                "urgent", "rotating_light,warning",
            )

    if delivered:
        write_state(state_path, next_state)
    else:
        print("WARNING alert undelivered; state left unchanged so the next run retries")

    # A measured breach exits 2, not 1. Python already exits 1 for an uncaught
    # exception, so accepting 1 in the unit file would make a crashed monitor
    # indistinguishable from a working one that found a problem.
    return EXIT_BREACH if not result.ok else 0


if __name__ == "__main__":
    sys.exit(main())
