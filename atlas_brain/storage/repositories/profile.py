"""
User profile repository for personalization settings.

Manages user preferences and profile data.
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from ..database import get_db_pool
from ..models import UserProfile

logger = logging.getLogger("atlas.storage.profile")


class ProfileRepository:
    """
    Repository for user profile management.

    Handles personalization settings like response style,
    expertise level, and RAG preferences.
    """

    async def get_profile(self, user_id: UUID) -> Optional[UserProfile]:
        """
        Get a user's profile.

        Args:
            user_id: The user ID

        Returns:
            UserProfile if found, None otherwise
        """
        pool = get_db_pool()
        row = await pool.fetchrow(
            """
            SELECT id, user_id, display_name, timezone, locale,
                   response_style, expertise_level, enable_rag,
                   enable_context_injection, created_at, updated_at
            FROM user_profiles
            WHERE user_id = $1
            """,
            user_id,
        )

        if not row:
            return None

        return UserProfile(
            id=row["id"],
            user_id=row["user_id"],
            display_name=row["display_name"],
            timezone=row["timezone"],
            locale=row["locale"],
            response_style=row["response_style"],
            expertise_level=row["expertise_level"],
            enable_rag=row["enable_rag"],
            enable_context_injection=row["enable_context_injection"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def get_or_create_profile(self, user_id: UUID) -> UserProfile:
        """
        Get a user's profile or create default if not exists.

        Args:
            user_id: The user ID

        Returns:
            UserProfile (existing or newly created)
        """
        existing = await self.get_profile(user_id)
        if existing:
            return existing

        return await self.create_profile(user_id)

    async def create_profile(
        self,
        user_id: UUID,
        display_name: Optional[str] = None,
        timezone: str = "UTC",
        locale: str = "en-US",
        response_style: str = "balanced",
        expertise_level: str = "intermediate",
    ) -> UserProfile:
        """
        Create a new user profile.

        Args:
            user_id: The user ID
            display_name: Optional display name
            timezone: User timezone
            locale: User locale
            response_style: brief, balanced, or detailed
            expertise_level: beginner, intermediate, or expert

        Returns:
            The created UserProfile
        """
        pool = get_db_pool()
        profile_id = uuid4()
        now = datetime.utcnow()

        await pool.execute(
            """
            INSERT INTO user_profiles
                (id, user_id, display_name, timezone, locale,
                 response_style, expertise_level, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (user_id) DO NOTHING
            """,
            profile_id,
            user_id,
            display_name,
            timezone,
            locale,
            response_style,
            expertise_level,
            now,
            now,
        )

        logger.info("Created profile for user %s", user_id)

        return UserProfile(
            id=profile_id,
            user_id=user_id,
            display_name=display_name,
            timezone=timezone,
            locale=locale,
            response_style=response_style,
            expertise_level=expertise_level,
            created_at=now,
            updated_at=now,
        )

    async def update_profile(
        self,
        user_id: UUID,
        display_name: Optional[str] = None,
        timezone: Optional[str] = None,
        locale: Optional[str] = None,
        response_style: Optional[str] = None,
        expertise_level: Optional[str] = None,
        enable_rag: Optional[bool] = None,
        enable_context_injection: Optional[bool] = None,
    ) -> Optional[UserProfile]:
        """
        Update a user's profile.

        Only updates fields that are provided (not None).

        Args:
            user_id: The user ID
            **kwargs: Fields to update

        Returns:
            Updated UserProfile if successful, None if not found
        """
        pool = get_db_pool()

        updates = []
        params = []
        param_idx = 1

        if display_name is not None:
            updates.append(f"display_name = ${param_idx}")
            params.append(display_name)
            param_idx += 1

        if timezone is not None:
            updates.append(f"timezone = ${param_idx}")
            params.append(timezone)
            param_idx += 1

        if locale is not None:
            updates.append(f"locale = ${param_idx}")
            params.append(locale)
            param_idx += 1

        if response_style is not None:
            updates.append(f"response_style = ${param_idx}")
            params.append(response_style)
            param_idx += 1

        if expertise_level is not None:
            updates.append(f"expertise_level = ${param_idx}")
            params.append(expertise_level)
            param_idx += 1

        if enable_rag is not None:
            updates.append(f"enable_rag = ${param_idx}")
            params.append(enable_rag)
            param_idx += 1

        if enable_context_injection is not None:
            updates.append(f"enable_context_injection = ${param_idx}")
            params.append(enable_context_injection)
            param_idx += 1

        if not updates:
            return await self.get_profile(user_id)

        updates.append(f"updated_at = ${param_idx}")
        params.append(datetime.utcnow())
        param_idx += 1

        params.append(user_id)

        query = f"""
            UPDATE user_profiles
            SET {', '.join(updates)}
            WHERE user_id = ${param_idx}
        """

        await pool.execute(query, *params)
        logger.debug("Updated profile for user %s", user_id)

        return await self.get_profile(user_id)

    async def get_preference(
        self,
        user_id: UUID,
        key: str,
    ) -> Optional[str]:
        """Get a specific user preference."""
        pool = get_db_pool()
        return await pool.fetchval(
            """
            SELECT preference_value
            FROM user_preferences
            WHERE user_id = $1 AND preference_key = $2
            """,
            user_id,
            key,
        )

    async def set_preference(
        self,
        user_id: UUID,
        key: str,
        value: str,
    ) -> None:
        """Set a user preference."""
        pool = get_db_pool()
        await pool.execute(
            """
            INSERT INTO user_preferences (id, user_id, preference_key, preference_value)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, preference_key)
            DO UPDATE SET preference_value = $4
            """,
            uuid4(),
            user_id,
            key,
            value,
        )

    async def get_all_preferences(self, user_id: UUID) -> dict[str, str]:
        """Get all preferences for a user."""
        pool = get_db_pool()
        rows = await pool.fetch(
            """
            SELECT preference_key, preference_value
            FROM user_preferences
            WHERE user_id = $1
            """,
            user_id,
        )

        return {row["preference_key"]: row["preference_value"] for row in rows}

    async def delete_preference(self, user_id: UUID, key: str) -> bool:
        """Delete a user preference."""
        pool = get_db_pool()
        result = await pool.execute(
            """
            DELETE FROM user_preferences
            WHERE user_id = $1 AND preference_key = $2
            """,
            user_id,
            key,
        )
        return "DELETE 1" in result


# Global repository instance
_profile_repo: Optional[ProfileRepository] = None


def get_profile_repo() -> ProfileRepository:
    """Get the global profile repository."""
    global _profile_repo
    if _profile_repo is None:
        _profile_repo = ProfileRepository()
    return _profile_repo
