from extracted_content_pipeline.landing_page_input_contract import (
    LANDING_PAGE_CONTEXT_INPUT_KEYS,
    LANDING_PAGE_EXISTING_CONTEXT_KEYS,
    LANDING_PAGE_INPUT_ASSET,
    LANDING_PAGE_SEO_GEO_AEO_INPUT_KEYS,
    landing_page_seo_geo_aeo_input_contracts,
)


def test_landing_page_context_inputs_keep_existing_keys_and_add_seo_geo_aeo_keys() -> None:
    for key in LANDING_PAGE_EXISTING_CONTEXT_KEYS:
        assert key in LANDING_PAGE_CONTEXT_INPUT_KEYS

    for key in LANDING_PAGE_SEO_GEO_AEO_INPUT_KEYS:
        assert key in LANDING_PAGE_CONTEXT_INPUT_KEYS


def test_landing_page_seo_geo_aeo_contracts_are_catalog_ready() -> None:
    contracts = landing_page_seo_geo_aeo_input_contracts()

    assert set(contracts) == set(LANDING_PAGE_SEO_GEO_AEO_INPUT_KEYS)
    assert contracts["target_keyword"] == {
        "key": "target_keyword",
        "label": "Target keyword",
        "type": "string",
        "placeholder": "customer support FAQ",
        "asset": LANDING_PAGE_INPUT_ASSET,
        "group": "seo_geo_aeo",
    }
    assert contracts["secondary_keywords"]["type"] == "string_list"
    assert contracts["faq_questions"]["group"] == "seo_geo_aeo"
    assert all(item["asset"] == LANDING_PAGE_INPUT_ASSET for item in contracts.values())
