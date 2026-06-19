"""Session-safe ``mcp`` stubbing for MCP-server unit tests.

Several MCP-server unit tests import a server module that constructs a
``mcp.server.fastmcp.FastMCP`` at import time. Historically each test planted a
fake ``mcp`` package into ``sys.modules`` at module top level with
``setdefault`` and never removed it. Under whole-suite collection that fake
survived for the rest of the session, with two failure modes:

* It poisoned sibling tests that need the real ``mcp`` package -- notably the
  invoicing OAuth tests, which import ``mcp.server.auth.provider`` and then
  saw ``'mcp.server' is not a package``.
* Because ``setdefault`` plants only when the key is absent, whichever test
  was collected first won, and later tests inherited the *wrong* fake -- e.g.
  a content-ops server built against a b2b passthrough ``FastMCP`` whose
  ``tool()`` rejected ``structured_output`` and lacked ``custom_route``.

``stub_mcp`` plants the fake only for the duration of the caller's server
import, then restores ``sys.modules``. The imported server module keeps its
reference to the fake ``FastMCP`` instance, so the tests behave exactly as
before, while collection of every other module sees the real ``mcp`` (CI) or
a clean absence (local sandbox).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, Mapping
from unittest.mock import MagicMock


@contextmanager
def stub_mcp(
    fastmcp_cls: type,
    *,
    extra_modules: Mapping[str, object] | None = None,
) -> Iterator[None]:
    """Temporarily install a fake ``mcp`` package, restoring afterward.

    ``fastmcp_cls`` is installed as ``mcp.server.fastmcp.FastMCP``. Any
    ``extra_modules`` (e.g. ``{"mcp.server.auth.provider": <stub>}``) are
    installed and restored alongside the defaults. Import the server module(s)
    under test inside the ``with`` block; their bound reference to the fake
    survives after ``sys.modules`` is restored.
    """
    fastmcp_mod = MagicMock(name="mcp.server.fastmcp")
    fastmcp_mod.FastMCP = fastmcp_cls
    planted: dict[str, object] = {
        "mcp": MagicMock(name="mcp"),
        "mcp.server": MagicMock(name="mcp.server"),
        "mcp.server.fastmcp": fastmcp_mod,
    }
    planted.update(extra_modules or {})
    saved = {name: sys.modules.get(name) for name in planted}
    try:
        sys.modules.update(planted)
        yield
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
