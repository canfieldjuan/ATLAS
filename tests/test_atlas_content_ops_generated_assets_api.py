"""Atlas host mount tests for generated Content Ops asset routes.

The extracted router owns behavior-level tests. These tests only pin
the Atlas API aggregator wiring so operators can reach review/export
routes through the hosted API surface.
"""

from __future__ import annotations

import importlib
import sys


def _fresh_api_package():
    original = sys.modules.pop("atlas_brain.api", None)
    try:
        return importlib.import_module("atlas_brain.api")
    finally:
        if original is not None:
            sys.modules["atlas_brain.api"] = original


def _route(api_pkg, path: str):
    route = next((route for route in api_pkg.router.routes if route.path == path), None)
    assert route is not None, f"Route {path!r} not mounted"
    return route


def test_api_aggregator_mounts_generated_asset_routes() -> None:
    api_pkg = _fresh_api_package()

    paths = {getattr(route, "path", "") for route in api_pkg.router.routes}

    assert "/content-assets/{asset}/drafts" in paths
    assert "/content-assets/{asset}/drafts/export" in paths
    assert "/content-assets/{asset}/drafts/review" in paths
    assert "/content-assets/landing_page/public/sitemap.xml" in paths
    assert "/content-assets/landing_page/public/{landing_page_id}" in paths


def test_generated_asset_routes_use_shared_content_ops_auth_scope_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-assets/{asset}/drafts")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert closure["scope_provider"].__name__ == "build_content_ops_scope"
    assert "_capture_content_ops_auth_user" in dependency_names


def test_faq_deflection_search_route_uses_shared_content_ops_auth_scope_and_pool() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-ops/faq-deflection-search")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert closure["scope_provider"].__name__ == "build_content_ops_scope"
    assert "_capture_content_ops_auth_user" in dependency_names


def test_public_landing_page_route_uses_pool_without_content_ops_auth_dependency() -> None:
    api_pkg = _fresh_api_package()
    route = _route(api_pkg, "/content-assets/landing_page/public/{landing_page_id}")
    closure = dict(
        zip(
            route.endpoint.__code__.co_freevars,
            (cell.cell_contents for cell in route.endpoint.__closure__ or ()),
        )
    )
    dependency_names = [
        getattr(dependency.call, "__name__", "")
        for dependency in route.dependant.dependencies
    ]

    assert closure["pool_provider"].__name__ == "get_db_pool"
    assert "_capture_content_ops_auth_user" not in dependency_names
