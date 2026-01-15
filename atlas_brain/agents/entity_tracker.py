"""
Entity tracker for pronoun resolution in conversations.

Tracks recently mentioned devices/entities to resolve pronouns like
"them", "it", "that" in follow-up commands.

Example:
    User: "Turn on the living room lights"
    Agent: tracks entity(type="light", name="living room")
    User: "Dim them to 50%"
    Agent: resolves "them" -> "living room lights"
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("atlas.agents.entity_tracker")

# Pronouns that refer to singular entities
SINGULAR_PRONOUNS = {"it", "that", "this"}

# Pronouns that refer to plural or collective entities
PLURAL_PRONOUNS = {"them", "those", "these"}

# All pronouns we handle
ALL_PRONOUNS = SINGULAR_PRONOUNS | PLURAL_PRONOUNS


@dataclass
class TrackedEntity:
    """A tracked entity from a previous command."""

    entity_type: str  # light, switch, tv, thermostat
    entity_name: str  # living room, kitchen, bedroom
    entity_id: Optional[str] = None  # Capability ID if known
    timestamp: float = field(default_factory=time.time)

    def matches_type(self, entity_type: str) -> bool:
        """Check if this entity matches a type (case-insensitive)."""
        return self.entity_type.lower() == entity_type.lower()

    def age_seconds(self) -> float:
        """Get age of this entity in seconds."""
        return time.time() - self.timestamp


class EntityTracker:
    """
    Tracks recently mentioned entities for pronoun resolution.

    Maintains a list of entities mentioned in recent commands,
    allowing resolution of pronouns like "them" and "it".
    """

    def __init__(
        self,
        max_entities: int = 10,
        ttl_seconds: float = 300.0,
    ):
        """
        Initialize entity tracker.

        Args:
            max_entities: Maximum entities to track
            ttl_seconds: Time-to-live for entities (default 5 minutes)
        """
        self._entities: list[TrackedEntity] = []
        self._max_entities = max_entities
        self._ttl_seconds = ttl_seconds

    def track(
        self,
        entity_type: str,
        entity_name: str,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Track a new entity.

        Args:
            entity_type: Type of entity (light, switch, etc.)
            entity_name: Name of entity (living room, kitchen, etc.)
            entity_id: Optional capability ID
        """
        if not entity_type or not entity_name:
            return

        # Remove expired entities first
        self._prune_expired()

        # Check for duplicate (same type and name)
        for i, existing in enumerate(self._entities):
            if (existing.entity_type.lower() == entity_type.lower() and
                existing.entity_name.lower() == entity_name.lower()):
                # Update timestamp and move to front
                self._entities.pop(i)
                break

        # Add new entity at front (most recent)
        entity = TrackedEntity(
            entity_type=entity_type,
            entity_name=entity_name,
            entity_id=entity_id,
        )
        self._entities.insert(0, entity)

        # Trim to max size
        if len(self._entities) > self._max_entities:
            self._entities = self._entities[:self._max_entities]

        logger.debug(
            "Tracked entity: %s/%s (total: %d)",
            entity_type,
            entity_name,
            len(self._entities),
        )

    def resolve_pronoun(
        self,
        pronoun: str,
        entity_type: Optional[str] = None,
    ) -> Optional[TrackedEntity]:
        """
        Resolve a pronoun to the most recent matching entity.

        Args:
            pronoun: The pronoun to resolve (it, them, that, etc.)
            entity_type: Optional type filter

        Returns:
            TrackedEntity if resolved, None otherwise
        """
        pronoun_lower = pronoun.lower()
        if pronoun_lower not in ALL_PRONOUNS:
            return None

        self._prune_expired()

        if not self._entities:
            logger.debug("No entities tracked for pronoun resolution")
            return None

        # Filter by type if specified
        candidates = self._entities
        if entity_type:
            candidates = [e for e in candidates if e.matches_type(entity_type)]
            if not candidates:
                logger.debug("No entities match type: %s", entity_type)
                return None

        # Return most recent
        resolved = candidates[0]
        logger.info(
            "Resolved pronoun '%s' -> %s/%s",
            pronoun,
            resolved.entity_type,
            resolved.entity_name,
        )
        return resolved

    def get_recent(
        self,
        entity_type: Optional[str] = None,
        limit: int = 1,
    ) -> list[TrackedEntity]:
        """
        Get recent entities, optionally filtered by type.

        Args:
            entity_type: Optional type filter
            limit: Maximum entities to return

        Returns:
            List of tracked entities
        """
        self._prune_expired()

        candidates = self._entities
        if entity_type:
            candidates = [e for e in candidates if e.matches_type(entity_type)]

        return candidates[:limit]

    def clear(self) -> None:
        """Clear all tracked entities."""
        self._entities.clear()
        logger.debug("Entity tracker cleared")

    def _prune_expired(self) -> None:
        """Remove entities older than TTL."""
        now = time.time()
        before = len(self._entities)
        self._entities = [
            e for e in self._entities
            if (now - e.timestamp) < self._ttl_seconds
        ]
        pruned = before - len(self._entities)
        if pruned > 0:
            logger.debug("Pruned %d expired entities", pruned)

    @property
    def count(self) -> int:
        """Get number of tracked entities."""
        return len(self._entities)

    def __len__(self) -> int:
        """Get number of tracked entities."""
        return len(self._entities)


def has_pronoun(text: str) -> bool:
    """
    Check if text contains a pronoun we can resolve.

    Args:
        text: Input text to check

    Returns:
        True if text contains a resolvable pronoun
    """
    text_lower = text.lower()
    words = set(text_lower.split())
    return bool(words & ALL_PRONOUNS)


def extract_pronoun(text: str) -> Optional[str]:
    """
    Extract the first pronoun from text.

    Args:
        text: Input text

    Returns:
        First pronoun found, or None
    """
    text_lower = text.lower()
    words = text_lower.split()
    for word in words:
        # Strip punctuation
        clean = word.strip(".,!?")
        if clean in ALL_PRONOUNS:
            return clean
    return None
