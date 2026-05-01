"""B2B Churn MCP -- vendor registry tools."""
import json
from typing import Optional

from ._shared import logger
from .server import mcp


def _clean_optional_text(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _clean_required_text(value: Optional[str]) -> str | None:
    return _clean_optional_text(value)


def _clean_alias_csv(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    cleaned = []
    for raw_alias in value.split(","):
        alias = _clean_optional_text(raw_alias)
        if alias:
            cleaned.append(alias)
    return cleaned


@mcp.tool()
async def list_vendors_registry(limit: int = 100) -> str:
    """
    List all canonical vendors in the vendor registry with their aliases.

    limit: Maximum results (default 100, cap 500)
    """
    limit = max(1, min(limit, 500))
    try:
        from atlas_brain.services.vendor_registry import list_vendors

        vendors = await list_vendors()
        result = [
            {
                "id": str(v["id"]),
                "canonical_name": v["canonical_name"],
                "aliases": list(v["aliases"]) if isinstance(v["aliases"], list) else [],
                "created_at": v["created_at"],
                "updated_at": v["updated_at"],
            }
            for v in vendors[:limit]
        ]
        return json.dumps({"vendors": result, "count": len(result)}, default=str)
    except Exception:
        logger.exception("list_vendors_registry error")
        return json.dumps({"error": "Internal error", "vendors": [], "count": 0})


@mcp.tool()
async def fuzzy_vendor_search(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> str:
    """
    Search vendors by name using fuzzy matching (trigram similarity).

    Finds vendors even with typos or partial names (e.g. "Salesfroce" -> "Salesforce").

    query: Vendor name to search for
    limit: Max results (default 10, max 100)
    min_similarity: Minimum similarity threshold 0.0-1.0 (default 0.3)
    """
    clean_query = _clean_required_text(query)
    if clean_query is None:
        return json.dumps({"error": "query is required"})
    try:
        from atlas_brain.services.vendor_registry import fuzzy_search_vendors

        results = await fuzzy_search_vendors(
            clean_query, limit=limit, min_similarity=min_similarity,
        )
        return json.dumps({"query": clean_query, "results": results, "count": len(results)}, default=str)
    except Exception:
        logger.exception("fuzzy_vendor_search error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def fuzzy_company_search(
    query: str,
    vendor_name: Optional[str] = None,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> str:
    """
    Search company names using fuzzy matching (trigram similarity).

    Finds companies even with typos or partial names. Optionally scoped to a vendor.

    query: Company name to search for
    vendor_name: Optional vendor name to scope the search
    limit: Max results (default 10, max 100)
    min_similarity: Minimum similarity threshold 0.0-1.0 (default 0.3)
    """
    clean_query = _clean_required_text(query)
    if clean_query is None:
        return json.dumps({"error": "query is required"})
    clean_vendor_name = _clean_optional_text(vendor_name)
    try:
        from atlas_brain.services.vendor_registry import fuzzy_search_companies

        results = await fuzzy_search_companies(
            clean_query, vendor_name=clean_vendor_name, limit=limit, min_similarity=min_similarity,
        )
        return json.dumps({
            "query": clean_query,
            "vendor_filter": clean_vendor_name,
            "results": results,
            "count": len(results),
        }, default=str)
    except Exception:
        logger.exception("fuzzy_company_search error")
        return json.dumps({"error": "Internal error"})


@mcp.tool()
async def add_vendor_to_registry(
    canonical_name: str,
    aliases: Optional[str] = None,
) -> str:
    """
    Add or update a vendor in the canonical vendor registry.

    canonical_name: The official vendor name (e.g. "Salesforce")
    aliases: Comma-separated lowercase aliases (e.g. "sf,sfdc,salesforce.com")
    """
    clean_canonical_name = _clean_required_text(canonical_name)
    if clean_canonical_name is None:
        return json.dumps({"success": False, "error": "canonical_name is required"})
    try:
        from atlas_brain.services.vendor_registry import add_vendor

        alias_list = _clean_alias_csv(aliases)

        row = await add_vendor(clean_canonical_name, alias_list)
        return json.dumps({
            "success": True,
            "vendor": {
                "id": str(row["id"]),
                "canonical_name": row["canonical_name"],
                "aliases": list(row["aliases"]) if isinstance(row["aliases"], list) else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_vendor_to_registry error")
        return json.dumps({"success": False, "error": "Internal error"})


@mcp.tool()
async def add_vendor_alias(
    canonical_name: str,
    alias: str,
) -> str:
    """
    Add an alias to an existing vendor in the registry.

    canonical_name: Existing canonical vendor name (e.g. "Salesforce")
    alias: New alias to add (e.g. "salesforce.com")
    """
    clean_canonical_name = _clean_required_text(canonical_name)
    if clean_canonical_name is None:
        return json.dumps({"success": False, "error": "canonical_name is required"})
    clean_alias = _clean_required_text(alias)
    if clean_alias is None:
        return json.dumps({"success": False, "error": "alias is required"})
    try:
        from atlas_brain.services.vendor_registry import add_alias

        row = await add_alias(clean_canonical_name, clean_alias)
        if row is None:
            return json.dumps({
                "success": False,
                "error": f"Vendor '{clean_canonical_name}' not found in registry",
            })
        return json.dumps({
            "success": True,
            "vendor": {
                "id": str(row["id"]),
                "canonical_name": row["canonical_name"],
                "aliases": list(row["aliases"]) if isinstance(row["aliases"], list) else [],
                "updated_at": row["updated_at"],
            },
        }, default=str)
    except Exception:
        logger.exception("add_vendor_alias error")
        return json.dumps({"success": False, "error": "Internal error"})
