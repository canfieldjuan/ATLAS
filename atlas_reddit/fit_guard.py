"""Runtime claim-safety guard for fit output (v2 S3, #1931).

The enforcement half of the ruler: before any model-written fit text is
persisted or rendered, this guard scans it against the SAME catalogue the
S1 harness grades and the S2 prompt teaches. The guard adds only policy
on top of the shared scan -- which codes block -- and a parity test pins
that the policy partitions the catalogue exactly, so a rule added
upstream cannot silently ship unclassified.

Persistence policy (enforced by the S4 store): a blocked decision is
recorded flagged-with-codes but its reason/angle text is REDACTED --
audit trail without unsafe text at rest (a PII echo must not get parked
in SQLite by the very guard that caught it).

Pure and deterministic: no I/O, no network, no clock, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fit import FitDecision
from .fit_rules import ALL_RULE_CODES, scan_fit_text

# Policy: in v1 EVERY catalogue family blocks. Advisory is deliberately
# empty -- unsupported-outcome claims, pitch/reply posture, and PII echo
# all make the text unusable as advice, and a flagged-but-rendered
# "advisory" state would put the unsafe text on the digest anyway. The
# set exists (rather than being implied) so the parity test fails loudly
# when a future rule needs a real classification decision.
BLOCKING_CODES: frozenset[str] = frozenset(ALL_RULE_CODES)
ADVISORY_CODES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GuardDecision:
    """Outcome of guarding one FitDecision. ``ok`` is False when any
    blocking code fired; ``codes`` carries every code that fired (stable,
    machine-readable, privacy-safe -- never matched text)."""

    ok: bool
    codes: tuple[str, ...]


def guard_fit_decision(
    decision: FitDecision, *, pii_allowlist: frozenset[str] = frozenset()
) -> GuardDecision:
    """Scan a parsed, contract-valid FitDecision's advisory text.

    Reason and angle are scanned SEPARATELY (concatenation would blind
    start-anchored rules to the angle -- the S1 lesson). The allowlist
    parameter exists for harness parity; runtime callers pass nothing and
    get the empty default.
    """
    findings = scan_fit_text(decision.reason, pii_allowlist=pii_allowlist)
    if decision.angle is not None:
        findings += scan_fit_text(decision.angle, pii_allowlist=pii_allowlist)
    codes = tuple(sorted({finding.code for finding in findings}))
    blocked = any(code in BLOCKING_CODES for code in codes)
    return GuardDecision(ok=not blocked, codes=codes)
