"""Slim EOM-facing API package.

This package intentionally lives outside ``atlas_brain.api``. Importing any
``atlas_brain.api.*`` submodule executes that package's all-router
``__init__`` first, which pulls in the full Atlas/B2B/voice surface. The EOM
Render profile needs a narrow import graph so it can boot without the local
model and B2B dependency stack.
"""
