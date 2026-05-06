"""Read-only readiness checks for standalone campaign product installs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
import os
from typing import Literal


CheckStatus = Literal["ok", "warning", "error"]
ProfileName = Literal[
    "offline",
    "generation",
    "send",
    "sequence",
    "webhooks",
    "analytics",
    "export",
    "all",
]
SenderName = Literal["none", "resend", "ses"]

DATABASE_URL_ENV = ("EXTRACTED_DATABASE_URL", "DATABASE_URL")
RESEND_API_KEY_ENV = (
    "EXTRACTED_RESEND_API_KEY",
    "EXTRACTED_CAMPAIGN_RESEND_API_KEY",
    "EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY",
)
FROM_EMAIL_ENV = (
    "EXTRACTED_CAMPAIGN_FROM_EMAIL",
    "EXTRACTED_RESEND_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_RESEND_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_SEQ_RESEND_FROM_EMAIL",
    "EXTRACTED_SES_FROM_EMAIL",
)
SES_FROM_EMAIL_ENV = FROM_EMAIL_ENV
SEQUENCE_FROM_EMAIL_ENV = (
    "EXTRACTED_CAMPAIGN_SEQUENCE_FROM_EMAIL",
    "EXTRACTED_CAMPAIGN_FROM_EMAIL",
)
RESEND_WEBHOOK_SECRET_ENV = (
    "EXTRACTED_RESEND_WEBHOOK_SECRET",
    "EXTRACTED_CAMPAIGN_RESEND_WEBHOOK_SECRET",
)
CAMPAIGN_LLM_ENV = (
    "EXTRACTED_CAMPAIGN_LLM_WORKLOAD",
    "EXTRACTED_CAMPAIGN_LLM_PREFER_CLOUD",
    "EXTRACTED_CAMPAIGN_LLM_TRY_OPENROUTER",
    "EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA",
    "EXTRACTED_CAMPAIGN_LLM_OPENROUTER_MODEL",
)

_ALL_PROFILES: tuple[ProfileName, ...] = (
    "generation",
    "send",
    "sequence",
    "webhooks",
    "analytics",
    "export",
)


@dataclass(frozen=True)
class InstallCheck:
    """One readiness finding."""

    name: str
    status: CheckStatus
    message: str
    env_names: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.env_names:
            data["env_names"] = list(self.env_names)
        return data


@dataclass(frozen=True)
class InstallCheckReport:
    """Readiness summary for a selected host install profile."""

    profiles: tuple[str, ...]
    sender: str
    checks: tuple[InstallCheck, ...]

    @property
    def passed(self) -> bool:
        return not any(check.status == "error" for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        counts = {
            "ok": sum(1 for check in self.checks if check.status == "ok"),
            "warning": sum(1 for check in self.checks if check.status == "warning"),
            "error": sum(1 for check in self.checks if check.status == "error"),
        }
        return {
            "passed": self.passed,
            "profiles": list(self.profiles),
            "sender": self.sender,
            "counts": counts,
            "checks": [check.as_dict() for check in self.checks],
        }


def check_campaign_install(
    *,
    environ: Mapping[str, str] | None = None,
    profiles: Sequence[str] = ("offline",),
    sender: str = "none",
    llm: str = "pipeline",
    require_webhook_secret: bool = True,
) -> InstallCheckReport:
    """Inspect host env/import readiness without opening network or DB handles."""

    env = os.environ if environ is None else environ
    normalized_profiles = _normalize_profiles(profiles)
    normalized_sender = _normalize_sender(sender)
    checks: list[InstallCheck] = [
        _module_check("product_package", "extracted_content_pipeline", required=True),
    ]
    if _requires_database(normalized_profiles):
        checks.extend(_database_checks(env))
    if "generation" in normalized_profiles:
        checks.extend(_generation_checks(env, llm=llm))
    if "send" in normalized_profiles:
        checks.extend(_send_checks(env, sender=normalized_sender))
    if "sequence" in normalized_profiles:
        checks.extend(_sequence_checks(env))
        if "generation" not in normalized_profiles:
            checks.extend(_generation_checks(env, llm=llm))
    if "webhooks" in normalized_profiles:
        checks.extend(_webhook_checks(env, require_secret=require_webhook_secret))
    return InstallCheckReport(
        profiles=tuple(normalized_profiles),
        sender=normalized_sender,
        checks=tuple(checks),
    )


def _normalize_profiles(profiles: Sequence[str]) -> tuple[ProfileName, ...]:
    requested = tuple(str(profile or "").strip().lower() for profile in profiles)
    if not requested:
        return ("offline",)
    if "all" in requested:
        return _ALL_PROFILES
    allowed = {"offline", *_ALL_PROFILES}
    invalid = [profile for profile in requested if profile not in allowed]
    if invalid:
        raise ValueError(f"Unsupported install profile: {invalid[0]!r}")
    return tuple(dict.fromkeys(requested))  # preserves caller order


def _normalize_sender(sender: str) -> SenderName:
    normalized = str(sender or "").strip().lower()
    if normalized in {"", "none"}:
        return "none"
    if normalized in {"resend", "ses"}:
        return normalized  # type: ignore[return-value]
    raise ValueError(f"Unsupported sender: {sender!r}")


def _requires_database(profiles: Sequence[str]) -> bool:
    database_profiles = {
        "generation",
        "send",
        "sequence",
        "webhooks",
        "analytics",
        "export",
    }
    return any(profile in database_profiles for profile in profiles)


def _database_checks(env: Mapping[str, str]) -> list[InstallCheck]:
    return [
        _env_check(
            "database_url",
            env,
            DATABASE_URL_ENV,
            required=True,
            present_message="Postgres DSN is configured.",
            missing_message=(
                "Set EXTRACTED_DATABASE_URL or DATABASE_URL for DB-backed commands."
            ),
        ),
        _module_check("asyncpg", "asyncpg", required=True),
    ]


def _generation_checks(env: Mapping[str, str], *, llm: str) -> list[InstallCheck]:
    if str(llm or "").strip().lower() == "offline":
        return []
    return [
        _env_check(
            "campaign_llm_route",
            env,
            CAMPAIGN_LLM_ENV,
            required=False,
            present_message="Campaign LLM route overrides are configured.",
            missing_message=(
                "No EXTRACTED_CAMPAIGN_LLM_* overrides found; the pipeline "
                "LLM adapter will use its default host route."
            ),
        )
    ]


def _send_checks(env: Mapping[str, str], *, sender: SenderName) -> list[InstallCheck]:
    if sender == "none":
        return [
            InstallCheck(
                "sender_provider",
                "warning",
                "Send profile selected without --sender; provider credentials were not checked.",
            )
        ]
    checks = [_module_check("httpx", "httpx", required=True)]
    if sender == "ses":
        checks.extend([
            _module_check("boto3", "boto3", required=True),
            _env_check(
                "ses_from_email",
                env,
                SES_FROM_EMAIL_ENV,
                required=True,
                present_message="SES sender email is configured.",
                missing_message=(
                    "Set EXTRACTED_SES_FROM_EMAIL or EXTRACTED_CAMPAIGN_FROM_EMAIL."
                ),
            ),
        ])
        return checks
    checks.extend([
        _env_check(
            "resend_api_key",
            env,
            RESEND_API_KEY_ENV,
            required=True,
            present_message="Resend API key is configured.",
            missing_message=(
                "Set EXTRACTED_RESEND_API_KEY, EXTRACTED_CAMPAIGN_RESEND_API_KEY, "
                "or EXTRACTED_CAMPAIGN_SEQ_RESEND_API_KEY."
            ),
        ),
        _env_check(
            "from_email",
            env,
            FROM_EMAIL_ENV,
            required=True,
            present_message="Campaign From email is configured.",
            missing_message=(
                "Set EXTRACTED_CAMPAIGN_FROM_EMAIL or a provider-specific From email."
            ),
        ),
    ])
    return checks


def _sequence_checks(env: Mapping[str, str]) -> list[InstallCheck]:
    return [
        _env_check(
            "sequence_from_email",
            env,
            SEQUENCE_FROM_EMAIL_ENV,
            required=True,
            present_message="Campaign sequence From email is configured.",
            missing_message=(
                "Set EXTRACTED_CAMPAIGN_SEQUENCE_FROM_EMAIL or "
                "EXTRACTED_CAMPAIGN_FROM_EMAIL."
            ),
        )
    ]


def _webhook_checks(
    env: Mapping[str, str],
    *,
    require_secret: bool,
) -> list[InstallCheck]:
    return [
        _env_check(
            "resend_webhook_secret",
            env,
            RESEND_WEBHOOK_SECRET_ENV,
            required=require_secret,
            present_message="Resend webhook signing secret is configured.",
            missing_message=(
                "Set EXTRACTED_RESEND_WEBHOOK_SECRET or "
                "EXTRACTED_CAMPAIGN_RESEND_WEBHOOK_SECRET."
            ),
        )
    ]


def _env_check(
    name: str,
    env: Mapping[str, str],
    env_names: Sequence[str],
    *,
    required: bool,
    present_message: str,
    missing_message: str,
) -> InstallCheck:
    if any(str(env.get(env_name) or "").strip() for env_name in env_names):
        return InstallCheck(name, "ok", present_message, tuple(env_names))
    return InstallCheck(
        name,
        "error" if required else "warning",
        missing_message,
        tuple(env_names),
    )


def _module_check(name: str, module_name: str, *, required: bool) -> InstallCheck:
    if find_spec(module_name) is not None:
        return InstallCheck(name, "ok", f"Python module {module_name!r} is importable.")
    return InstallCheck(
        name,
        "error" if required else "warning",
        f"Python module {module_name!r} is not importable.",
    )


__all__ = [
    "CAMPAIGN_LLM_ENV",
    "DATABASE_URL_ENV",
    "FROM_EMAIL_ENV",
    "InstallCheck",
    "InstallCheckReport",
    "RESEND_API_KEY_ENV",
    "RESEND_WEBHOOK_SECRET_ENV",
    "SEQUENCE_FROM_EMAIL_ENV",
    "SES_FROM_EMAIL_ENV",
    "check_campaign_install",
]
