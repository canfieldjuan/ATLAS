"""Read-only Reddit listening tool for the Resolution Audit (issues #1931 / #1934).

This package is deliberately standalone: it does not import atlas_brain,
performs no Reddit writes anywhere in any slice, and keeps all runtime
data under the gitignored data/ directory. See
plans/PR-Reddit-Listening-Config-Scoring.md and the #1934 arc for scope.
"""

__version__ = "0.1.0"
