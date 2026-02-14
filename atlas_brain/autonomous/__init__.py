"""
Autonomous task scheduler for Atlas Brain.

Provides scheduled and alert-driven headless agent execution.

Usage:
    from atlas_brain.autonomous import init_autonomous, shutdown_autonomous

    scheduler = await init_autonomous()
    # ...
    await shutdown_autonomous()
"""

import logging

from .scheduler import TaskScheduler, get_task_scheduler
from .hooks import HookManager, get_hook_manager
from .runner import HeadlessRunner, get_headless_runner
from .config import autonomous_config

logger = logging.getLogger("atlas.autonomous")

_hook_callback_ref = None
_event_queue = None
_presence_callback_ref = None
_broadcast_callback_ref = None


async def init_autonomous() -> TaskScheduler:
    """
    Initialize the autonomous scheduler and hook manager.

    Returns the TaskScheduler instance.
    """
    global _hook_callback_ref, _event_queue, _presence_callback_ref, _broadcast_callback_ref

    # Start the task scheduler
    scheduler = get_task_scheduler()
    await scheduler.start()

    # Load hook mappings
    hook_manager = get_hook_manager()
    await hook_manager.load_hooks_from_db()

    # Register hook callback with AlertManager if hooks are enabled
    if autonomous_config.hooks_enabled:
        try:
            from ..alerts import get_alert_manager
            from ..config import settings

            if settings.alerts.enabled:
                alert_manager = get_alert_manager()

                # Initialize event queue for debounced hook dispatch
                if autonomous_config.event_queue_enabled:
                    from .event_queue import EventQueue, EventQueueConfig

                    eq_config = EventQueueConfig(
                        debounce_seconds=autonomous_config.event_queue_debounce_seconds,
                        max_batch_size=autonomous_config.event_queue_max_batch_size,
                        max_age_seconds=autonomous_config.event_queue_max_age_seconds,
                    )
                    eq = EventQueue(eq_config)
                    eq.register_callback(hook_manager.on_alert_batch)
                    _event_queue = eq

                    # Register queue-routing callback with AlertManager.
                    # Capture `eq` by value to avoid relying on the global.
                    async def _route_to_queue(message, rule, event, _eq=eq):
                        await _eq.enqueue(event, rule, message)

                    _hook_callback_ref = _route_to_queue
                    alert_manager.register_callback(_hook_callback_ref)
                    logger.info(
                        "Registered event queue with AlertManager "
                        "(debounce=%.1fs, %d hooks)",
                        autonomous_config.event_queue_debounce_seconds,
                        hook_manager.hook_count,
                    )
                else:
                    # Direct hook dispatch (no queue)
                    _hook_callback_ref = hook_manager.on_alert
                    alert_manager.register_callback(_hook_callback_ref)
                    logger.info(
                        "Registered hook callback with AlertManager (%d hooks)",
                        hook_manager.hook_count,
                    )
        except Exception as e:
            logger.warning("Could not register hook callback: %s", e)

    # Initialize presence tracker if enabled
    if autonomous_config.presence_enabled:
        try:
            from .presence import get_presence_tracker, PresenceConfig, OccupancyState
            from ..alerts import get_alert_manager, PresenceAlertEvent
            from ..config import settings

            tracker = get_presence_tracker()
            tracker._config = PresenceConfig(
                empty_delay_seconds=autonomous_config.presence_empty_delay_seconds,
                arrival_cooldown_seconds=autonomous_config.presence_arrival_cooldown_seconds,
            )

            alert_mgr = get_alert_manager()

            async def _presence_to_alert(transition, state, person):
                """Fire a presence alert event so hook tasks can trigger."""
                evt = PresenceAlertEvent.from_presence_state(
                    transition=transition,
                    state_value=state.state.value,
                    occupants=list(state.occupants.keys()),
                    person=person,
                )
                await alert_mgr.process_event(evt)

            tracker.register_callback(_presence_to_alert)
            _presence_callback_ref = _presence_to_alert

            # Broadcast occupancy state changes to edge nodes
            if settings.escalation.broadcast_occupancy:
                async def _broadcast_occupancy(transition, state, person):
                    """Broadcast occupancy state to all connected edge nodes."""
                    try:
                        from ..api.edge.websocket import get_all_connections
                    except Exception:
                        return
                    broadcast = {
                        "type": "occupancy_update",
                        "state": state.state.value,
                        "occupants": list(state.occupants.keys()),
                        "changed_at": state.changed_at.isoformat() if state.changed_at else None,
                        "transition": transition,
                        "person": person,
                    }
                    for conn in get_all_connections().values():
                        try:
                            await conn.send(broadcast)
                        except Exception:
                            pass

                tracker.register_callback(_broadcast_occupancy)
                _broadcast_callback_ref = _broadcast_occupancy
                logger.info("Occupancy broadcast callback registered")

            # Recover state from last DB row on startup
            try:
                from ..storage.database import get_db_pool
                from datetime import datetime as _dt
                import json as _json

                _pool = get_db_pool()
                if _pool.is_initialized:
                    _row = await _pool.fetchrow(
                        """
                        SELECT transition, occupancy_state, occupants,
                               arrival_times, unknown_count, created_at
                        FROM presence_events
                        ORDER BY created_at DESC LIMIT 1
                        """
                    )
                    if _row and _row["transition"] == "arrival":
                        # Restore occupants with arrival timestamps
                        _at = _row["arrival_times"]  # JSONB -> dict or None
                        if _at and isinstance(_at, str):
                            _at = _json.loads(_at)  # handle double-encoded rows
                        if _at and isinstance(_at, dict):
                            tracker._state.occupants = {
                                n: _dt.fromisoformat(t) for n, t in _at.items()
                            }
                        elif _row["occupants"]:
                            # Pre-migration rows: use created_at as fallback
                            _ts = _row["created_at"]
                            tracker._state.occupants = {
                                n: _ts for n in _row["occupants"]
                            }
                        # Restore occupancy state and unknown count
                        tracker._state.state = OccupancyState(_row["occupancy_state"])
                        tracker._state.changed_at = _row["created_at"]
                        tracker._state.last_activity = _row["created_at"]
                        tracker._unknown_count = _row["unknown_count"] or 0
                        logger.info(
                            "Presence state recovered: %s, occupants=%s",
                            tracker._state.state.value,
                            list(tracker._state.occupants.keys()),
                        )
                    else:
                        logger.info("Presence state: starting EMPTY (last=%s)",
                                    _row["transition"] if _row else "none")
            except Exception as _e:
                logger.warning("Could not recover presence state: %s", _e)

            logger.info("Presence tracker initialized (empty_delay=%ds)",
                        autonomous_config.presence_empty_delay_seconds)
        except Exception as e:
            logger.warning("Could not initialize presence tracker: %s", e)

    return scheduler


async def shutdown_autonomous() -> None:
    """Shutdown the autonomous scheduler and unregister hooks."""
    global _hook_callback_ref, _event_queue, _presence_callback_ref, _broadcast_callback_ref

    # 1. Unregister hook callback FIRST -- stop new events from entering queue
    if _hook_callback_ref is not None:
        try:
            from ..alerts import get_alert_manager
            from ..config import settings

            if settings.alerts.enabled:
                alert_manager = get_alert_manager()
                alert_manager.unregister_callback(_hook_callback_ref)
                logger.info("Unregistered hook callback from AlertManager")
        except Exception as e:
            logger.warning("Could not unregister hook callback: %s", e)
        _hook_callback_ref = None

    # 2. Shutdown event queue -- flush any remaining batched events
    if _event_queue is not None:
        try:
            await _event_queue.shutdown()
            logger.info("Event queue shut down")
        except Exception as e:
            logger.warning("Event queue shutdown error: %s", e)
        _event_queue = None

    # 3. Shutdown presence tracker
    if _presence_callback_ref is not None:
        try:
            from .presence import get_presence_tracker
            tracker = get_presence_tracker()
            await tracker.shutdown()
            logger.info("Presence tracker shut down")
        except Exception as e:
            logger.warning("Presence tracker shutdown error: %s", e)
        _presence_callback_ref = None
        _broadcast_callback_ref = None

    # 4. Stop the scheduler
    scheduler = get_task_scheduler()
    await scheduler.stop()


__all__ = [
    "init_autonomous",
    "shutdown_autonomous",
    "TaskScheduler",
    "get_task_scheduler",
    "HookManager",
    "get_hook_manager",
    "HeadlessRunner",
    "get_headless_runner",
]
