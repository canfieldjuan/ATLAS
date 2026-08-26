"""
Database configuration for Atlas Brain.

Configuration is loaded from environment variables with sensible defaults.
"""

from typing import Optional
from urllib.parse import quote, urlsplit

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILES = (".env", ".env.local")


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_DB_",
        env_file=ENV_FILES,
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable database persistence"
    )
    host: str = Field(
        default="localhost",
        description="PostgreSQL host (override: ATLAS_DB_HOST)",
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
    connection_string: str = Field(
        default="",
        description="Full PostgreSQL connection string (override: ATLAS_DB_CONNECTION_STRING)",
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
        if self.connection_string.strip():
            return self.connection_string.strip()
        if self.socket_path:
            # Unix socket connection (lowest latency)
            return (
                f"postgresql://{self.user}:{self.password}@/{self.database}"
                f"?host={quote(self.socket_path, safe='/')}&port={self.port}"
            )
        else:
            # TCP connection
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def target_label(self) -> str:
        """Return a log-safe database target label without credentials."""
        if self.connection_string.strip():
            try:
                parsed = urlsplit(self.connection_string.strip())
                database = parsed.path.lstrip("/") or "<database>"
                host = parsed.hostname or "connection-string"
                port = f":{parsed.port}" if parsed.port else ""
                return f"dsn={host}{port}/{database}"
            except ValueError:
                return "dsn=<connection-string>"
        if self.socket_path:
            return (
                f"socket={self.socket_path}, port={self.port}, db={self.database}"
            )
        return f"host={self.host}, port={self.port}, db={self.database}"

    def connection_kwargs(
        self,
        *,
        command_timeout: float | None = None,
    ) -> dict[str, object]:
        """Return asyncpg connection kwargs from the configured DB target."""
        timeout_kwargs: dict[str, object] = {
            "timeout": self.connect_timeout,
            "command_timeout": (
                self.command_timeout
                if command_timeout is None
                else command_timeout
            ),
        }
        if self.connection_string.strip():
            return {
                "dsn": self.connection_string.strip(),
                **timeout_kwargs,
            }
        return {
            "host": self.socket_path or self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            **timeout_kwargs,
        }


# Singleton settings instance
db_settings = DatabaseConfig()
