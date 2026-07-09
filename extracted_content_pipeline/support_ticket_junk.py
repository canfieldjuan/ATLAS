"""Support-ticket junk/auto-reply admission gate (S6E, F2).

Design: closed structural rules, not phrase enumeration.

Machine-generated mail betrays itself by POSITION and SHAPE, not vocabulary:
auto-replies and bounces carry a generator prefix in the subject
("Automatic reply: ...", "Undeliverable: ..."), and out-of-office bodies are
first-person ASSERTIONS occupying whole lines ("I am out of the office
until Monday."). Customer text ABOUT those features is interrogative or
descriptive ("How do I set an out of office auto-reply?", "Out of office
not working") and matches neither shape, so it passes.

The gate returns a bounded reason code so diagnostics report counts, never
content.
"""

from __future__ import annotations

import re

JUNK_REASON_AUTO_REPLY = "auto_reply"
JUNK_REASON_BOUNCE = "bounce"
JUNK_REASON_NO_NEW_CONTENT = "no_new_content"

JUNK_REASONS = (
    JUNK_REASON_AUTO_REPLY,
    JUNK_REASON_BOUNCE,
    JUNK_REASON_NO_NEW_CONTENT,
)

# Subject prefixes mail software prepends to generated messages. The
# trailing colon/bracket position is the discriminator: a HUMAN subject
# about the feature ("Out of office not working") does not carry it.
_AUTO_REPLY_SUBJECT_RE = re.compile(
    r"^\s*(?:\[[^\]]{1,40}\]\s*)*"
    r"(?:subject\s*[:\uFF1A]\s*)?"
    r"(?:automatic\s+reply|auto[\s_-]*reply|autoreply|auto[\s_-]*response"
    r"|automated\s+(?:reply|response)|out\s+of\s+(?:the\s+)?office"
    r"|abwesenheitsnotiz|reponse\s+automatique)"
    r"\s*[:\uFF1A]",
    re.IGNORECASE,
)
_BOUNCE_SUBJECT_RE = re.compile(
    r"^\s*(?:\[[^\]]{1,40}\]\s*)*"
    r"(?:subject\s*[:\uFF1A]\s*)?"
    r"(?:(?:undeliverable|undelivered\s+mail|returned\s+mail"
    r"|failure\s+notice|mail\s+delivery\s+(?:failed|failure|subsystem))"
    r"\s*[:\uFF1A]"
    r"|delivery\s+status\s+notification\s*(?:\([^)]*\))?\s*$"
    r"|delivery\s+has\s+failed\s+to\s+these\s+recipients\s*"
    r"(?:or\s+groups)?\s*[:.]?\s*$)",
    re.IGNORECASE,
)

# First-person out-of-office / automated-sender assertions, matched as
# WHOLE admitted lines. Questions and feature discussions are not
# first-person assertions and never match these anchored shapes.
_AUTO_REPLY_LINE_RES = (
    re.compile(
        r"^i(?:\s+am|['\u2019]m|\s+will\s+be|['\u2019]ll\s+be)\s+"
        r"(?:currently\s+)?"
        r"(?:out\s+of\s+(?:the\s+)?office|away\s+from\s+(?:my\s+)?"
        r"(?:desk|email|the\s+office)|on\s+(?:vacation|leave|holiday|pto))"
        r"(?:\s*[.,!]|\s+(?:until|through|till|thru|from|starting"
        r"|beginning|on|this|next|for|all|today|tomorrow|and|with(?:out)?"
        r"|returning|back)\b)"
        r"[^?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i\s+(?:will\s+)?have\s+(?:limited|no)\s+access\s+to\s+"
        r"(?:my\s+)?e?mail\b[^?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^this\s+is\s+an\s+automat(?:ed|ic)\s+"
        r"(?:reply|response|message|notification)\b[^?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?do\s+not\s+reply\s+to\s+this\s+"
        r"(?:automated\s+)?(?:email|message)\b[^?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i\s+(?:will\s+)?(?:be\s+)?return(?:ing)?\s+"
        r"(?:to\s+the\s+office|from\s+(?:leave|vacation|holiday|pto))\s+"
        r"on\s+\S+[^?]*$",
        re.IGNORECASE,
    ),
)
_SUBJECT_LABEL_RE = re.compile(r"^\s*subject\s*[:\uFF1A]", re.IGNORECASE)

_BOUNCE_LINE_RES = (
    re.compile(
        r"^(?:your\s+)?message\s+(?:could\s+not|couldn't|was\s+not|wasn't)\s+"
        r"(?:be\s+)?delivered\b[^?]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^delivery\s+to\s+the\s+following\s+recipients?\s+failed\b[^?]*$",
        re.IGNORECASE,
    ),
)


def support_ticket_row_is_junk(
    subject: str,
    body_lines: str,
    *,
    had_source_text: bool = False,
) -> str | None:
    """Classify a normalized row as junk, or return None to admit it.

    ``subject`` is the admitted title text; ``body_lines`` is the
    line-preserving admitted body (``support_ticket_plain_text_lines``),
    including public comment text so comment-only rows are in scope.
    ``had_source_text`` marks rows whose raw source had text so that a body
    emptied by hygiene (an all-quote row) counts as no-new-content instead
    of being confused with a legitimately empty row.

    A row containing ANY interrogative line is a customer asking something
    -- quoted auto-reply templates inside a real question stay admitted --
    so body-shape rules apply only to question-free rows. Generator subject
    prefixes are definitive, and are also checked on the first body line
    for exports that land the email subject in the text column.
    """

    if _AUTO_REPLY_SUBJECT_RE.match(subject or ""):
        return JUNK_REASON_AUTO_REPLY
    if _BOUNCE_SUBJECT_RE.match(subject or ""):
        return JUNK_REASON_BOUNCE
    lines = [line.strip() for line in (body_lines or "").split("\n") if line.strip()]
    if lines:
        if _AUTO_REPLY_SUBJECT_RE.match(lines[0]):
            return JUNK_REASON_AUTO_REPLY
        if _BOUNCE_SUBJECT_RE.match(lines[0]):
            return JUNK_REASON_BOUNCE
        # Text exports often carry a leading header block (From:/To:/
        # Subject:); a Subject:-labeled generator prefix anywhere in it is
        # definitive. Unlabeled prefixes stay first-line-only so quoted
        # replies deeper in a body cannot junk the row.
        for line in lines[1:5]:
            if _SUBJECT_LABEL_RE.match(line) and (
                _AUTO_REPLY_SUBJECT_RE.match(line)
                or _BOUNCE_SUBJECT_RE.match(line)
            ):
                if _AUTO_REPLY_SUBJECT_RE.match(line):
                    return JUNK_REASON_AUTO_REPLY
                return JUNK_REASON_BOUNCE
    # Question VOICE is the discriminator: an UNLABELED question is the
    # customer speaking (veto -- the row is a real ticket even if it quotes
    # an OOO template or bounce text), while a "Subject:"-labeled question
    # is machine mail quoting the customer's original subject and does not
    # veto. Residual junk quoting questions is quantified by diagnostics.
    if any(
        "?" in line and not _SUBJECT_LABEL_RE.match(line) for line in lines
    ):
        return None
    for line in lines:
        for pattern in _AUTO_REPLY_LINE_RES:
            if pattern.match(line):
                return JUNK_REASON_AUTO_REPLY
        for pattern in _BOUNCE_LINE_RES:
            if pattern.match(line):
                return JUNK_REASON_BOUNCE
    if had_source_text and not lines and not (subject or "").strip():
        return JUNK_REASON_NO_NEW_CONTENT
    return None


__all__ = [
    "JUNK_REASONS",
    "JUNK_REASON_AUTO_REPLY",
    "JUNK_REASON_BOUNCE",
    "JUNK_REASON_NO_NEW_CONTENT",
    "support_ticket_row_is_junk",
]
