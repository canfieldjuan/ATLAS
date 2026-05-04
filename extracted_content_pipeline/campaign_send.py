"""Standalone send orchestration for campaign emails."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from .campaign_ports import (
    AuditSink,
    CampaignRepository,
    CampaignSender,
    Clock,
    SendRequest,
)
from .campaign_suppression import CampaignSuppressionService, domain_from_email, normalize_email


@dataclass(frozen=True)
class CampaignSendConfig:
    """Runtime send config owned by the campaign product."""

    default_from_email: str = ""
    default_reply_to: str | None = None
    unsubscribe_base_url: str = ""
    unsubscribe_token_secret: str = ""
    company_address: str = ""
    limit: int = 20


@dataclass(frozen=True)
class CampaignSendSummary:
    sent: int = 0
    failed: int = 0
    suppressed: int = 0
    skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sent": self.sent,
            "failed": self.failed,
            "suppressed": self.suppressed,
            "skipped": self.skipped,
        }


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def build_unsubscribe_token(recipient_email: str, token_secret: str) -> str:
    cleaned_email = normalize_email(recipient_email)
    cleaned_secret = str(token_secret or "").strip()
    if not cleaned_email or not cleaned_secret:
        return ""
    return hmac.new(
        cleaned_secret.encode("utf-8"),
        cleaned_email.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_unsubscribe_token(
    recipient_email: str,
    token: str,
    token_secret: str,
) -> bool:
    expected = build_unsubscribe_token(recipient_email, token_secret)
    candidate = str(token or "").strip()
    return bool(expected and candidate and hmac.compare_digest(expected, candidate))


def build_unsubscribe_url(
    base_url: str,
    recipient_email: str,
    *,
    token_secret: str = "",
) -> str:
    sep = "&" if "?" in base_url else "?"
    url = f"{base_url}{sep}email={quote(recipient_email, safe='')}"
    token = build_unsubscribe_token(recipient_email, token_secret)
    if token:
        url = f"{url}&token={quote(token, safe='')}"
    return url


def unsubscribe_headers(
    base_url: str,
    recipient_email: str,
    *,
    token_secret: str = "",
) -> dict[str, str]:
    if not base_url:
        return {}
    unsubscribe_url = build_unsubscribe_url(
        base_url,
        recipient_email,
        token_secret=token_secret,
    )
    return {
        "List-Unsubscribe": f"<{unsubscribe_url}>",
        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
    }


def wrap_with_footer(
    body: str,
    *,
    recipient_email: str,
    config: CampaignSendConfig,
) -> str:
    if not config.unsubscribe_base_url and not config.company_address:
        return body

    parts: list[str] = []
    if config.company_address:
        parts.append(config.company_address)
    if config.unsubscribe_base_url:
        unsubscribe_url = build_unsubscribe_url(
            config.unsubscribe_base_url,
            recipient_email,
            token_secret=config.unsubscribe_token_secret,
        )
        parts.append(f'<a href="{unsubscribe_url}" style="color:#999;">Unsubscribe</a>')

    footer = (
        '<p style="font-size:11px;color:#999;margin-top:24px;'
        'border-top:1px solid #eee;padding-top:8px;">'
        + "<br>".join(parts)
        + "</p>"
    )
    return body + footer


def _value(row: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _merge_headers(
    row_headers: Any,
    *,
    unsubscribe_base_url: str,
    recipient_email: str,
    unsubscribe_token_secret: str,
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if isinstance(row_headers, Mapping):
        headers.update({str(k): str(v) for k, v in row_headers.items()})
    headers.update(
        unsubscribe_headers(
            unsubscribe_base_url,
            recipient_email,
            token_secret=unsubscribe_token_secret,
        )
    )
    return headers


def _tags_for(row: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    tags: list[dict[str, str]] = []
    raw_tags = row.get("tags")
    if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes)):
        for item in raw_tags:
            if isinstance(item, Mapping):
                name = str(item.get("name") or "").strip()
                value = str(item.get("value") or "").strip()
                if name and value:
                    tags.append({"name": name, "value": value})

    campaign_id = str(_value(row, "id", "campaign_id", default="") or "").strip()
    company_name = str(row.get("company_name") or "").strip()
    step_number = str(row.get("step_number") or "").strip()
    if campaign_id and not any(item["name"] == "campaign" for item in tags):
        tags.append({"name": "campaign", "value": campaign_id})
    if company_name and not any(item["name"] == "company" for item in tags):
        tags.append({"name": "company", "value": company_name})
    if step_number and not any(item["name"] == "step" for item in tags):
        tags.append({"name": "step", "value": step_number})
    return tuple(tags)


class CampaignSendService:
    """Orchestrate queued campaign sends through injected product ports."""

    def __init__(
        self,
        *,
        campaigns: CampaignRepository,
        suppression: CampaignSuppressionService,
        sender: CampaignSender,
        audit: AuditSink,
        clock: Clock | None = None,
        config: CampaignSendConfig | None = None,
    ) -> None:
        self._campaigns = campaigns
        self._suppression = suppression
        self._sender = sender
        self._audit = audit
        self._clock = clock or SystemClock()
        self._config = config or CampaignSendConfig()

    async def send_due(self, *, limit: int | None = None) -> CampaignSendSummary:
        now = self._clock.now()
        rows = await self._campaigns.list_due_sends(
            limit=int(limit or self._config.limit),
            now=now,
        )

        sent = failed = suppressed = skipped = 0
        for row in rows:
            campaign_id = str(_value(row, "id", "campaign_id", default="") or "").strip()
            sequence_id = _value(row, "sequence_id")
            recipient_email = normalize_email(_value(row, "recipient_email", "to_email"))
            from_email = str(row.get("from_email") or self._config.default_from_email or "").strip()

            if not campaign_id:
                skipped += 1
                continue
            if not recipient_email:
                await self._campaigns.mark_send_failed(
                    campaign_id=campaign_id,
                    error="recipient_email_missing",
                    metadata={"reason": "recipient_email_missing"},
                )
                await self._audit.record(
                    "send_skipped",
                    campaign_id=campaign_id,
                    sequence_id=str(sequence_id) if sequence_id else None,
                    metadata={"reason": "recipient_email_missing"},
                )
                skipped += 1
                continue
            if not from_email:
                await self._campaigns.mark_send_failed(
                    campaign_id=campaign_id,
                    error="from_email_missing",
                    metadata={"reason": "from_email_missing"},
                )
                await self._audit.record(
                    "send_skipped",
                    campaign_id=campaign_id,
                    sequence_id=str(sequence_id) if sequence_id else None,
                    metadata={"reason": "from_email_missing"},
                )
                skipped += 1
                continue

            domain = domain_from_email(recipient_email)
            if await self._suppression.is_suppressed(
                email=recipient_email,
                domain=domain,
            ):
                await self._campaigns.mark_cancelled(
                    campaign_id=campaign_id,
                    reason="suppressed",
                    metadata={"recipient_email": recipient_email, "domain": domain},
                )
                await self._audit.record(
                    "suppressed",
                    campaign_id=campaign_id,
                    sequence_id=str(sequence_id) if sequence_id else None,
                    metadata={"recipient_email": recipient_email, "domain": domain},
                )
                suppressed += 1
                continue

            body = str(_value(row, "html_body", "body", default="") or "")
            request = SendRequest(
                campaign_id=campaign_id,
                to_email=recipient_email,
                from_email=from_email,
                reply_to=str(row.get("reply_to") or self._config.default_reply_to or "") or None,
                subject=str(row.get("subject") or ""),
                html_body=wrap_with_footer(
                    body,
                    recipient_email=recipient_email,
                    config=self._config,
                ),
                text_body=row.get("text_body"),
                headers=_merge_headers(
                    row.get("headers"),
                    unsubscribe_base_url=self._config.unsubscribe_base_url,
                    recipient_email=recipient_email,
                    unsubscribe_token_secret=self._config.unsubscribe_token_secret,
                ),
                tags=_tags_for(row),
                metadata=dict(row.get("metadata") or {}) if isinstance(row.get("metadata"), dict) else {},
            )

            try:
                result = await self._sender.send(request)
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                await self._campaigns.mark_send_failed(
                    campaign_id=campaign_id,
                    error=error,
                    metadata={"recipient_email": recipient_email},
                )
                await self._audit.record(
                    "send_failed",
                    campaign_id=campaign_id,
                    sequence_id=str(sequence_id) if sequence_id else None,
                    metadata={"error": error, "recipient_email": recipient_email},
                )
                failed += 1
                continue

            await self._campaigns.mark_sent(
                campaign_id=campaign_id,
                result=result,
                sent_at=now,
            )
            await self._audit.record(
                "sent",
                campaign_id=campaign_id,
                sequence_id=str(sequence_id) if sequence_id else None,
                metadata={
                    "provider": result.provider,
                    "message_id": result.message_id,
                    "recipient_email": recipient_email,
                },
            )
            sent += 1

        return CampaignSendSummary(
            sent=sent,
            failed=failed,
            suppressed=suppressed,
            skipped=skipped,
        )
