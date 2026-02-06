"""
Database configuration for Atlas Brain.

Configuration is loaded from environment variables with sensible defaults.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_DB_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable database persistence"
    )
    host: str = Field(
        default="localhost",
        description="PostgreSQL host"
    )
    port: int = Field(
        default=5433,
        description="PostgreSQL port"
    )
    database: str = Field(
        default="atlas",
        description="Database name"
    )
    user: str = Field(
        default="atlas",
        description="Database user"
    )
    password: str = Field(
        default="",
        description="Database password"
    )
    min_pool_size: int = Field(
        default=2,
        description="Minimum connections in pool"
    )
    max_pool_size: int = Field(
        default=10,
        description="Maximum connections in pool"
    )
    # Unix socket for lowest latency (optional)
    socket_path: Optional[str] = Field(
        default=None,
        description="Unix socket path (overrides host/port if set)"
    )
    # Connection timeout
    connect_timeout: float = Field(
        default=10.0,
        description="Connection timeout in seconds"
    )
    # Command timeout
    command_timeout: float = Field(
        default=30.0,
        description="Command timeout in seconds"
    )

    @property
    def dsn(self) -> str:
        """Build PostgreSQL connection string."""
        if self.socket_path:
            # Unix socket connection (lowest latency)
            return f"postgresql://{self.user}:{self.password}@/{self.database}?host={self.socket_path}"
        else:
            # TCP connection
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


# Singleton settings instance
db_settings = DatabaseConfig()
