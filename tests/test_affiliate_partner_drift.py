"""Unit tests for the affiliate-partner drift audit's pure parser + reconciler.

No database required -- these cover the SQL-parsing and reconciliation logic
directly, which is the part that runs everywhere (CI included). The live-DB
path (`_run`) is exercised by the operator against :5433/atlas.
"""

import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "audit_affiliate_partner_drift",
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_affiliate_partner_drift.py",
)
drift = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drift)


# --- 326-style: 9 columns, one value per line, array + NULL literals ---------

SQL_326_STYLE = """
INSERT INTO affiliate_partners (
    name, product_name, product_aliases, category, affiliate_url,
    commission_type, commission_value, notes, enabled
) VALUES (
    'Monday.com',
    'Monday.com',
    ARRAY['monday', 'monday CRM', 'monday work OS']::text[],
    'Project Management',
    'https://try.monday.com/1p7bntdd5bui',
    'rev_share',
    '$100/signup',
    NULL,
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;

INSERT INTO affiliate_partners (
    name, product_name, product_aliases, category, affiliate_url,
    commission_type, commission_value, notes, enabled
) VALUES (
    'Shopify Affiliates',
    'Shopify',
    '{"shopify plus","shopify basic","shopify advanced"}'::text[],
    'E-commerce',
    'https://shopify.pxf.io/c/7062841/1424184/13624',
    'cpa',
    'up to $150/merchant',
    'Impact Radius program. Publisher ID: 7062841.',
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;
"""

# --- 088-style: 7 columns (no aliases/notes), packed values ------------------

SQL_088_STYLE = """
INSERT INTO affiliate_partners (
    name, product_name, category, affiliate_url,
    commission_type, commission_value, enabled
) VALUES (
    'Amazon Associates', 'Amazon', 'consumer',
    'https://www.amazon.com?tag=atlas0e9b-20',
    'percentage', '1-10%', true
) ON CONFLICT ((lower(product_name))) DO NOTHING;
"""


def test_parses_326_style_with_arrays_and_null():
    rows, errors = drift.parse_seeded_partners(SQL_326_STYLE)
    assert errors == []
    assert len(rows) == 2
    monday = next(r for r in rows if r["product_name"] == "Monday.com")
    assert monday["product_aliases"] == ["monday", "monday CRM", "monday work OS"]
    assert monday["category"] == "Project Management"
    assert monday["commission_value"] == "$100/signup"
    assert monday["notes"] is None
    assert monday["enabled"] is True

    shopify = next(r for r in rows if r["product_name"] == "Shopify")
    # Curly-brace array literal form parses identically to ARRAY[...].
    assert shopify["product_aliases"] == [
        "shopify plus",
        "shopify basic",
        "shopify advanced",
    ]
    assert shopify["notes"] == "Impact Radius program. Publisher ID: 7062841."


def test_parses_088_style_packed_values_fewer_columns():
    rows, errors = drift.parse_seeded_partners(SQL_088_STYLE)
    assert errors == []
    assert len(rows) == 1
    amazon = rows[0]
    assert amazon["product_name"] == "Amazon"
    assert amazon["affiliate_url"] == "https://www.amazon.com?tag=atlas0e9b-20"
    # Columns not present in this INSERT simply aren't keys.
    assert "product_aliases" not in amazon
    assert amazon["enabled"] is True


def test_empty_array_and_quote_escape():
    assert drift._parse_value("'{}'::text[]") == []
    # Embedded apostrophe via '' escape.
    assert drift._parse_value("'it''s fine'") == "it's fine"
    assert drift._parse_value("NULL") is None
    assert drift._parse_value("true") is True


def _seed_map(sql: str) -> dict:
    rows, errors = drift.parse_seeded_partners(sql)
    assert errors == []
    return {
        r["product_name"].lower(): {**r, "__migration__": "326_seed.sql"} for r in rows
    }


def test_reconcile_clean_when_all_match():
    # Build seeded map directly from parsed rows (avoid touching the filesystem).
    parsed = _seed_map(SQL_326_STYLE)
    db = [
        {
            "name": "Monday.com",
            "product_name": "Monday.com",
            "product_aliases": ["monday work OS", "monday", "monday CRM"],  # reordered
            "category": "Project Management",
            "affiliate_url": "https://try.monday.com/1p7bntdd5bui",
            "commission_type": "rev_share",
            "commission_value": "$100/signup",
            "notes": None,
            "enabled": True,
        },
        {
            "name": "Shopify Affiliates",
            "product_name": "Shopify",
            "product_aliases": ["shopify plus", "shopify basic", "shopify advanced"],
            "category": "E-commerce",
            "affiliate_url": "https://shopify.pxf.io/c/7062841/1424184/13624",
            "commission_type": "cpa",
            "commission_value": "up to $150/merchant",
            "notes": "Impact Radius program. Publisher ID: 7062841.",
            "enabled": True,
        },
    ]
    checks = {c["name"]: c for c in drift.reconcile(db, parsed)}
    assert checks["all_live_partners_versioned"]["status"] == "pass"
    # Alias reorder must NOT trigger divergence (compared as sets).
    assert checks["no_seed_value_divergence"]["status"] == "pass"
    assert checks["no_orphan_seeds"]["status"] == "pass"
    assert drift._exit_code(list(checks.values())) == 0


def test_reconcile_flags_unversioned_partner_as_fail():
    parsed = _seed_map(SQL_088_STYLE)
    db = [
        {"product_name": "Amazon", "product_aliases": [], "name": "Amazon Associates",
         "category": "consumer", "affiliate_url": "https://www.amazon.com?tag=atlas0e9b-20",
         "commission_type": "percentage", "commission_value": "1-10%", "notes": None, "enabled": True},
        # Created via API, never migrated:
        {"product_name": "Notion", "product_aliases": [], "name": "Notion Partner",
         "category": "Project Management", "affiliate_url": "https://notion.so/?ref=x",
         "commission_type": "cpa", "commission_value": "$50", "notes": None, "enabled": True},
    ]
    checks = {c["name"]: c for c in drift.reconcile(db, parsed)}
    assert checks["all_live_partners_versioned"]["status"] == "fail"
    unversioned = checks["all_live_partners_versioned"]["detail"]["unversioned_partners"]
    assert [p["product_name"] for p in unversioned] == ["Notion"]
    assert drift._exit_code(list(checks.values())) == 1


def test_reconcile_warns_on_value_divergence():
    parsed = _seed_map(SQL_326_STYLE)
    db = [
        {
            "name": "Monday.com",
            "product_name": "Monday.com",
            "product_aliases": ["monday", "monday CRM", "monday work OS"],
            "category": "Project Management",
            # URL edited in the DB after seeding:
            "affiliate_url": "https://try.monday.com/EDITED-IN-DB",
            "commission_type": "rev_share",
            "commission_value": "$100/signup",
            "notes": None,
            "enabled": False,  # toggled off -- must NOT count as divergence
        },
    ]
    checks = {c["name"]: c for c in drift.reconcile(db, parsed)}
    # Unversioned check still passes (Monday IS seeded).
    assert checks["all_live_partners_versioned"]["status"] == "pass"
    # Divergence is a warning, not a failure.
    div = checks["no_seed_value_divergence"]
    assert div["status"] == "warn"
    fields = div["detail"]["divergent_partners"][0]["fields"]
    assert "affiliate_url" in fields
    assert "enabled" not in fields  # excluded by design
    # Monday seeded but Shopify (also in SQL_326) has no live row -> orphan warn.
    assert checks["no_orphan_seeds"]["status"] == "warn"
    # Warnings do not fail the audit.
    assert drift._exit_code(list(checks.values())) == 0


# --- multi-row VALUES: every tuple must be read, not just the first ----------

SQL_MULTI_ROW = """
INSERT INTO affiliate_partners (
    name, product_name, category, affiliate_url, commission_type,
    commission_value, enabled
) VALUES
    ('A Co', 'Avendor', 'crm', 'https://a.example/?ref=x', 'cpa', '$10', true),
    ('B Co', 'Bvendor', 'crm', 'https://b.example/?ref=x', 'cpa', '$20', false)
ON CONFLICT ((lower(product_name))) DO NOTHING;
"""


def test_parses_multi_row_values():
    rows, errors = drift.parse_seeded_partners(SQL_MULTI_ROW)
    assert errors == []
    assert [r["product_name"] for r in rows] == ["Avendor", "Bvendor"]
    assert rows[0]["enabled"] is True
    assert rows[1]["enabled"] is False


def test_column_value_mismatch_is_surfaced_not_silently_dropped():
    bad = (
        "INSERT INTO affiliate_partners (name, product_name, category) "
        "VALUES ('X', 'Xvendor', 'crm', 'EXTRA') "
        "ON CONFLICT ((lower(product_name))) DO NOTHING;"
    )
    rows, errors = drift.parse_seeded_partners(bad)
    assert rows == []  # the mismatched tuple is not mis-mapped
    assert len(errors) == 1
    assert "mismatch" in errors[0]


def test_reconcile_fails_when_seeds_unparseable():
    # A parse error must fail the audit: reconciliation below would be against
    # an incomplete seed set, so a "pass" would be misleading.
    checks = {
        c["name"]: c
        for c in drift.reconcile(
            [], {}, parse_errors=["326_seed.sql: column/value count mismatch (9 vs 8)"]
        )
    }
    assert checks["migration_seeds_parseable"]["status"] == "fail"
    assert drift._exit_code(list(checks.values())) == 1


def test_finds_partner_mutations():
    assert drift.find_partner_mutations(SQL_326_STYLE) == []  # INSERT-only
    assert drift.find_partner_mutations(
        "UPDATE affiliate_partners SET affiliate_url = 'x' WHERE product_name = 'HubSpot';"
    ) == ["UPDATE"]
    both = drift.find_partner_mutations(
        "DELETE FROM affiliate_partners WHERE enabled = false;\n"
        "UPDATE affiliate_partners SET notes = NULL;"
    )
    assert set(both) == {"UPDATE", "DELETE"}


def test_reconcile_warns_on_unmodeled_mutation_without_failing():
    # A mutation migration is not an error, but the audit can't model its
    # effect -- surface it as a warning, not a failure.
    checks = {
        c["name"]: c
        for c in drift.reconcile(
            [], {}, mutations=["340_fix_hubspot_url.sql: UPDATE affiliate_partners"]
        )
    }
    assert checks["partner_mutations_modeled"]["status"] == "warn"
    assert checks["migration_seeds_parseable"]["status"] == "pass"
    assert drift._exit_code(list(checks.values())) == 0
