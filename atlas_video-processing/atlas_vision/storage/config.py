"""
Database configuration for atlas_vision.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_VISION_DB_")

    enabled: bool = Field(default=True, description="Enable database")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="atlas", description="Database name")
    user: str = Field(default="atlas", description="Database user")
    password: Optional[str] = Field(default=None, description="Database password")

    min_pool_size: int = Field(default=2, description="Min pool connections")
    max_pool_size: int = Field(default=10, description="Max pool connections")
    connect_timeout: float = Field(default=10.0, description="Connection timeout")
    command_timeout: float = Field(default=30.0, description="Command timeout")


db_settings = DatabaseSettings()
