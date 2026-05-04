"""PDF renderer bridge for extracted competitive intelligence.

Default mode re-exports atlas_brain.services.b2b.pdf_renderer. Standalone
mode exposes a configurable PDF renderer port and fails closed until
a host adapter is registered.
"""
from __future__ import annotations

import importlib as _importlib
import os as _os

if _os.environ.get("EXTRACTED_COMP_INTEL_STANDALONE") == "1":
    from ..._standalone.pdf_renderer import (  # noqa: F401
        PDFRenderer,
        PDFRendererNotConfigured,
        configure_pdf_renderer,
        get_pdf_renderer,
        render_report_pdf,
        render_vendor_full_report_pdf,
    )
else:
    def _bridge() -> None:
        src = _importlib.import_module("atlas_brain.services.b2b.pdf_renderer")
        g = globals()
        for name in dir(src):
            if not name.startswith("__"):
                g[name] = getattr(src, name)


    _bridge()
    del _bridge

del _importlib, _os
