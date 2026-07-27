#!/usr/bin/env python3
"""Preview or apply one exact Atlas service -> EOM portal Site rate update.

Identity is operator-selected and verified, never inferred. Preview is the
default. Applying requires the SHA-256 hash printed by a fresh preview.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import re
import sys
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from sync_eom_portal_customers import (
    DEFAULT_BASE_URL,
    EOM_CONTEXT_ID,
    fetch_portal_customers,
    portal_login,
    settings_default,
)

MAX_PORTAL_RATE = Decimal("999999.99")
RATE_TYPES = {
    "Per Visit": "per_visit",
    "Per Hour": "hourly",
    "Per Month": "monthly",
}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _portal_origin(value: str) -> str:
    try:
        parsed = urlsplit(str(value).strip())
        port = parsed.port
    except ValueError:
        raise argparse.ArgumentTypeError("portal origin is invalid") from None
    host = parsed.hostname
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "portal origin must be an HTTP(S) origin without credentials or a path"
        )
    if parsed.scheme == "http" and host not in {"localhost", "127.0.0.1", "::1"}:
        raise argparse.ArgumentTypeError("non-loopback portal origins must use HTTPS")
    host = f"[{host}]" if ":" in host else host.lower()
    default_port = 443 if parsed.scheme == "https" else 80
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{parsed.scheme}://{host}{port_suffix}"


async def load_service(
    pool: Any, service_id: uuid.UUID, *, lock: bool = False
) -> dict[str, Any]:
    query = """
        SELECT s.id AS service_id, s.rate, s.rate_label,
               c.id AS contact_id, c.metadata AS contact_metadata
          FROM customer_services s
          JOIN contacts c ON c.id = s.contact_id
         WHERE s.id = $1
           AND s.business_context_id = $2
           AND c.business_context_id = $2
           AND s.status = 'active'
           AND c.status = 'active'
           AND c.contact_type = 'customer'
        """
    if lock:
        query += " FOR SHARE OF s, c"
    row = await pool.fetchrow(query, service_id, EOM_CONTEXT_ID)
    if row is None:
        raise SystemExit("Active EOM service and customer Contact not found")
    return dict(row)


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if not isinstance(value, dict):
        raise SystemExit("Atlas Contact metadata is not a JSON object")
    return value


def _money(value: Any, *, source: str) -> float:
    try:
        amount = Decimal(str(value))
        normalized = amount.quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError):
        raise SystemExit(f"{source} rate is not a valid amount") from None
    if not amount.is_finite() or amount != normalized:
        raise SystemExit(f"{source} rate must have at most two decimal places")
    if amount < 0 or amount > MAX_PORTAL_RATE:
        raise SystemExit(f"{source} rate is outside the portal's allowed range")
    return float(normalized)


def _selected_site(
    customers: Sequence[dict[str, Any]], site_id: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for customer in customers:
        if not isinstance(customer, dict):
            continue
        for site in customer.get("sites") or []:
            if (
                customer.get("active") is True
                and isinstance(site, dict)
                and type(site.get("id")) is int
                and site["id"] == site_id
                and site.get("active") is True
            ):
                matches.append((customer, site))
    if len(matches) != 1:
        raise SystemExit("Exact active portal Site was not found uniquely")
    return matches[0]


def build_plan(
    service: Mapping[str, Any],
    customers: Sequence[dict[str, Any]],
    site_id: int,
    base_url: str,
) -> dict[str, Any]:
    metadata = _metadata(service.get("contact_metadata"))
    portal_customer_id = metadata.get("portal_customer_id")
    if type(portal_customer_id) is not int or portal_customer_id <= 0:
        raise SystemExit("Atlas Contact has no valid stamped portal Customer ID")

    customer, site = _selected_site(customers, site_id)
    customer_id = customer.get("id")
    if type(customer_id) is not int or customer_id != portal_customer_id:
        raise SystemExit("Selected portal Site is owned by another Customer")
    site_customer_id = site.get("customerId")
    if type(site_customer_id) is not int or site_customer_id != portal_customer_id:
        raise SystemExit("Selected portal Site has inconsistent Customer ownership")

    try:
        service_id = str(uuid.UUID(str(service.get("service_id"))))
    except ValueError:
        raise SystemExit("Atlas service ID is not a UUID") from None
    contact_id = str(service.get("contact_id"))
    try:
        contact_id = str(uuid.UUID(contact_id))
    except ValueError:
        raise SystemExit("Atlas Contact ID is not a UUID") from None
    portal_contact_id = customer.get("atlasContactId")
    if portal_contact_id not in (None, ""):
        try:
            portal_contact_id = str(uuid.UUID(str(portal_contact_id)))
        except ValueError:
            raise SystemExit("Portal Customer atlasContactId is not a UUID") from None
        if portal_contact_id != contact_id:
            raise SystemExit("Portal Customer is linked to another Atlas Contact")

    rate_label = service.get("rate_label")
    if rate_label not in RATE_TYPES:
        raise SystemExit(f"Unsupported Atlas rate label: {rate_label!r}")
    desired = {
        "rate": _money(service.get("rate"), source="Atlas"),
        "rateType": RATE_TYPES[rate_label],
    }
    current_rate = site.get("rate")
    current = {
        "rate": (
            None
            if current_rate is None
            else _money(current_rate, source="Portal")
        ),
        "rateType": site.get("rateType"),
    }
    token = site.get("updateToken")
    if not isinstance(token, str) or HASH_PATTERN.fullmatch(token) is None:
        raise SystemExit("Portal Site has no valid update token")
    customer_token = customer.get("updateToken")
    if (
        not isinstance(customer_token, str)
        or HASH_PATTERN.fullmatch(customer_token) is None
    ):
        raise SystemExit("Portal Customer has no valid update token")

    return {
        "action": "noop" if current == desired else "update",
        "atlas": {
            "contactId": contact_id,
            "serviceId": service_id,
        },
        "current": current,
        "desired": desired,
        "portal": {
            "baseUrl": base_url,
            "customerId": portal_customer_id,
            "expectedCustomerUpdateToken": customer_token,
            "expectedUpdateToken": token,
            "siteId": site_id,
        },
    }


def plan_hash(plan: Mapping[str, Any]) -> str:
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_plan(client: Any, token: str, plan: Mapping[str, Any]) -> None:
    if plan["action"] == "noop":
        print("No portal update required.")
        return
    portal = plan["portal"]
    desired = plan["desired"]
    response = client.patch(
        f"{portal['baseUrl']}/api/admin/locations/{portal['siteId']}",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "expectedCustomerUpdateToken": portal["expectedCustomerUpdateToken"],
            "expectedUpdateToken": portal["expectedUpdateToken"],
            "rate": desired["rate"],
            "rateType": desired["rateType"],
        },
    )
    if response.status_code != 200:
        suffix = "; rerun preview" if response.status_code == 409 else ""
        raise SystemExit(f"Portal Site update failed (HTTP {response.status_code}){suffix}")
    try:
        body = response.json() or {}
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        raise SystemExit("Portal Site update returned an invalid confirmation")
    location = body.get("location")
    if (
        body.get("success") is not True
        or not isinstance(location, dict)
        or type(location.get("id")) is not int
        or location.get("id") != portal["siteId"]
        or location.get("rateType") != desired["rateType"]
        or _money(location.get("rate"), source="Portal confirmation")
        != desired["rate"]
    ):
        raise SystemExit("Portal Site update returned an invalid confirmation")
    print(f"Updated portal Site {portal['siteId']}.")


def _finish(
    args: argparse.Namespace,
    client: Any,
    token: str,
    service: Mapping[str, Any],
    customers: Sequence[dict[str, Any]],
) -> int:
    plan = build_plan(service, customers, args.site_id, args.base_url)
    digest = plan_hash(plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    print(f"Plan hash: {digest}")
    if not args.apply:
        print("Preview only; no portal data changed.")
        return 0
    if not hmac.compare_digest(args.plan_hash, digest):
        raise SystemExit("Plan hash mismatch; rerun preview before applying")
    apply_plan(client, token, plan)
    return 0


async def run(
    args: argparse.Namespace,
    *,
    pool: Any = None,
    client_factory: Callable[[], Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    if pool is None:
        from atlas_brain.storage.database import get_db_pool

        pool = get_db_pool()
    await pool.initialize()
    if client_factory is None:
        import httpx

        def client_factory() -> Any:
            return httpx.Client(timeout=30.0)

    with client_factory() as client:
        runtime_env = os.environ if environ is None else environ
        token = portal_login(client, args.base_url, dict(runtime_env))
        customers = fetch_portal_customers(client, args.base_url, token)
        if args.apply:
            async with pool.transaction() as connection:
                service = await load_service(
                    connection, args.service_id, lock=True
                )
                return _finish(args, client, token, service, customers)
        service = await load_service(pool, args.service_id)
        return _finish(args, client, token, service, customers)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile one exact Atlas service to one EOM portal Site"
    )
    parser.add_argument("--service-id", required=True, type=uuid.UUID)
    parser.add_argument("--site-id", required=True, type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--plan-hash")
    parser.add_argument(
        "--base-url",
        type=_portal_origin,
        help="Portal origin; must match the configured credential origin",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    credential_origin: str | None = None,
    **run_kwargs: Any,
) -> int:
    parser = _parser()
    configured = (
        credential_origin
        or settings_default("eom_portal_base_url")
        or DEFAULT_BASE_URL
    )
    try:
        configured = _portal_origin(configured)
    except argparse.ArgumentTypeError as exc:
        parser.error(f"configured portal credential origin: {exc}")
    parser.set_defaults(base_url=configured)
    args = parser.parse_args(argv)
    if args.site_id <= 0:
        parser.error("--site-id must be greater than zero")
    if args.apply and (
        not isinstance(args.plan_hash, str)
        or HASH_PATTERN.fullmatch(args.plan_hash) is None
    ):
        parser.error("--apply requires the exact lowercase --plan-hash from preview")
    if not args.apply and args.plan_hash is not None:
        parser.error("--plan-hash is only valid with --apply")
    if args.base_url != configured:
        parser.error("--base-url must match the configured portal credential origin")
    return asyncio.run(run(args, **run_kwargs))


if __name__ == "__main__":
    sys.exit(main())
