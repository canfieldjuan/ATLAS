"""Parity tests for extracted_competitive_intelligence.storage.models.

The extracted package owns surgical copies of two dataclasses
(``ScheduledTask`` and ``CompetitiveSet``) rather than mirroring all
22 dataclasses from ``atlas_brain.storage.models``. That gives the
extracted product a clean narrow surface but introduces drift risk:
atlas could add a field on either dataclass and the extracted copy
would silently stay stale.

These tests catch that drift at CI time. If they fail, update the
extracted copy in ``extracted_competitive_intelligence/storage/models.py``
to match the atlas peer (and bump the relevant migration if a new
column was added).
"""

from __future__ import annotations

from dataclasses import fields

from atlas_brain.storage import models as atlas_models
from extracted_competitive_intelligence.storage import models as ext_models


def _field_names(dataclass_cls: type) -> set[str]:
    return {f.name for f in fields(dataclass_cls)}


def _field_defaults(dataclass_cls: type) -> dict[str, object]:
    return {
        f.name: (f.default, f.default_factory)
        for f in fields(dataclass_cls)
    }


def test_scheduled_task_field_parity() -> None:
    """``ScheduledTask`` columns must match atlas peer exactly."""
    assert _field_names(ext_models.ScheduledTask) == _field_names(
        atlas_models.ScheduledTask
    )


def test_competitive_set_field_parity() -> None:
    """``CompetitiveSet`` columns must match atlas peer exactly."""
    assert _field_names(ext_models.CompetitiveSet) == _field_names(
        atlas_models.CompetitiveSet
    )


def test_scheduled_task_to_dict_keys_parity() -> None:
    """``ScheduledTask.to_dict()`` keys must match atlas peer.

    Catches drift in the serialization contract -- e.g. if atlas
    starts emitting a new key, downstream consumers reading
    ``ext_models.ScheduledTask().to_dict()`` would silently miss it.
    """
    from uuid import uuid4

    ext = ext_models.ScheduledTask(
        id=uuid4(), name="t", task_type="builtin", schedule_type="cron"
    )
    atlas = atlas_models.ScheduledTask(
        id=uuid4(), name="t", task_type="builtin", schedule_type="cron"
    )
    assert set(ext.to_dict()) == set(atlas.to_dict())


def test_competitive_set_to_dict_keys_parity() -> None:
    """``CompetitiveSet.to_dict()`` keys must match atlas peer."""
    from uuid import uuid4

    ext = ext_models.CompetitiveSet(
        id=uuid4(), account_id=uuid4(), name="s", focal_vendor_name="v"
    )
    atlas = atlas_models.CompetitiveSet(
        id=uuid4(), account_id=uuid4(), name="s", focal_vendor_name="v"
    )
    assert set(ext.to_dict()) == set(atlas.to_dict())


def test_extracted_models_exposes_only_used_classes() -> None:
    """Pins the narrow surface: only ScheduledTask + CompetitiveSet.

    If a future caller in extracted_competitive_intelligence starts
    importing a third dataclass from ``models``, this test breaks and
    the maintainer must consciously decide whether to add the new
    dataclass (with its own parity test) or refactor the caller.
    """
    public = {name for name in dir(ext_models) if not name.startswith("_")}
    # Filter out re-exported stdlib helpers and module-level constants
    # so the assertion focuses on dataclass-shaped exports.
    public -= {"annotations", "dataclass", "field", "datetime", "timezone",
               "Any", "Optional", "UUID"}
    assert public == {"CompetitiveSet", "ScheduledTask"}
