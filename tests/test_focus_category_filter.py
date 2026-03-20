"""Unit tests for filter_vendors_by_focus_categories."""

import sys
import os

# Allow importing from the project root without full install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from atlas_brain.autonomous.tasks._b2b_shared import filter_vendors_by_focus_categories


SAMPLE_VENDORS = [
    {"vendor_name": "Salesforce", "product_category": "CRM"},
    {"vendor_name": "HubSpot", "product_category": "CRM"},
    {"vendor_name": "Zendesk", "product_category": "Help Desk"},
    {"vendor_name": "Freshdesk", "product_category": "Help Desk"},
    {"vendor_name": "Asana", "product_category": "Project Management"},
    {"vendor_name": "Monday.com", "product_category": "Project Management"},
    {"vendor_name": "Notion", "product_category": "Knowledge Management"},
]


def test_all_returns_full_list():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "all")
    assert result is SAMPLE_VENDORS  # same object, no copy


def test_empty_string_returns_full_list():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "")
    assert result is SAMPLE_VENDORS


def test_none_returns_full_list():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, None)
    assert result is SAMPLE_VENDORS


def test_single_category_filters():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "CRM")
    assert len(result) == 2
    assert all(v["product_category"] == "CRM" for v in result)


def test_multiple_categories():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "CRM,Help Desk")
    assert len(result) == 4
    names = {v["vendor_name"] for v in result}
    assert names == {"Salesforce", "HubSpot", "Zendesk", "Freshdesk"}


def test_case_insensitive():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "crm,help desk")
    assert len(result) == 4


def test_extra_whitespace():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "  CRM , Help Desk  ")
    assert len(result) == 4


def test_empty_input_list():
    result = filter_vendors_by_focus_categories([], "CRM")
    assert result == []


def test_no_matching_category():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "ERP")
    assert result == []


def test_vendor_missing_category_excluded():
    vendors = SAMPLE_VENDORS + [{"vendor_name": "NoCat"}]
    result = filter_vendors_by_focus_categories(vendors, "CRM")
    assert len(result) == 2


def test_all_with_whitespace():
    result = filter_vendors_by_focus_categories(SAMPLE_VENDORS, "  ALL  ")
    assert result is SAMPLE_VENDORS
