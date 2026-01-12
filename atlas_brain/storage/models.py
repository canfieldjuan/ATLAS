"""
Data models for Atlas Brain storage.

These are plain dataclasses, not ORM models.
We use raw SQL with asyncpg for maximum performance.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID


@dataclass
class User:
    """A registered user/speaker."""

    id: UUID
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    speaker_embedding: Optional[bytes] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "has_speaker_embedding": self.speaker_embedding is not None,
        }


@dataclass
class Session:
    """An active conversation session (one per user per day)."""

    id: UUID
    user_id: Optional[UUID] = None
    terminal_id: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    session_date: date = field(default_factory=date.today)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "terminal_id": self.terminal_id,
            "started_at": self.started_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "is_active": self.is_active,
            "session_date": self.session_date.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    id: UUID
    session_id: UUID
    role: str  # "user" or "assistant"
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    speaker_id: Optional[str] = None
    intent: Optional[str] = None
    turn_type: str = "conversation"  # "conversation" or "command"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "speaker_id": self.speaker_id,
            "intent": self.intent,
            "turn_type": self.turn_type,
            "metadata": self.metadata,
        }


@dataclass
class Terminal:
    """A registered Atlas terminal (device/location)."""

    id: str  # User-defined ID like "office", "car", "home"
    name: str
    location: Optional[str] = None
    capabilities: list[str] = field(default_factory=list)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "capabilities": self.capabilities,
            "registered_at": self.registered_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DiscoveredDevice:
    """A device discovered on the network."""

    id: UUID
    device_id: str  # Unique identifier like "roku.192_168_1_2"
    name: str  # Human-readable name
    device_type: str  # "roku", "chromecast", "smart_tv", etc.
    protocol: str  # Discovery protocol: "ssdp", "mdns", "manual"
    host: str  # IP address or hostname
    port: Optional[int] = None  # Port if applicable
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True  # Currently reachable
    auto_registered: bool = False  # Auto-added to capability registry
    metadata: dict[str, Any] = field(default_factory=dict)  # Protocol-specific data

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type,
            "protocol": self.protocol,
            "host": self.host,
            "port": self.port,
            "discovered_at": self.discovered_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "is_active": self.is_active,
            "auto_registered": self.auto_registered,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeDocument:
    """A document in the knowledge base."""

    id: UUID
    filename: str
    file_type: str
    content: str
    content_hash: str
    user_id: Optional[UUID] = None
    processed: bool = False
    chunk_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "filename": self.filename,
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "processed": self.processed,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DocumentChunk:
    """A chunk of a document with embedding."""

    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    embedding: Optional[bytes] = None
    token_count: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_index": self.chunk_index,
            "content": self.content,
            "has_embedding": self.embedding is not None,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Memory:
    """A long-term memory entry."""

    id: UUID
    memory_type: str
    content: str
    user_id: Optional[UUID] = None
    embedding: Optional[bytes] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "memory_type": self.memory_type,
            "content": self.content,
            "has_embedding": self.embedding is not None,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed_at": (
                self.last_accessed_at.isoformat() if self.last_accessed_at else None
            ),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


@dataclass
class Entity:
    """A knowledge graph entity."""

    id: UUID
    name: str
    entity_type: str
    description: Optional[str] = None
    embedding: Optional[bytes] = None
    source_chunk_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "has_embedding": self.embedding is not None,
            "source_chunk_id": (
                str(self.source_chunk_id) if self.source_chunk_id else None
            ),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class UserProfile:
    """User profile for personalization settings."""

    id: UUID
    user_id: Optional[UUID] = None
    display_name: Optional[str] = None
    timezone: str = "UTC"
    locale: str = "en-US"
    response_style: str = "balanced"
    expertise_level: str = "intermediate"
    enable_rag: bool = True
    enable_context_injection: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "display_name": self.display_name,
            "timezone": self.timezone,
            "locale": self.locale,
            "response_style": self.response_style,
            "expertise_level": self.expertise_level,
            "enable_rag": self.enable_rag,
            "enable_context_injection": self.enable_context_injection,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class RAGSourceUsage:
    """Tracks individual RAG source usage for feedback."""

    id: UUID
    session_id: Optional[UUID] = None
    query: str = ""
    source_id: Optional[str] = None
    source_fact: str = ""
    confidence: float = 0.0
    was_helpful: Optional[bool] = None
    feedback_type: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "session_id": str(self.session_id) if self.session_id else None,
            "query": self.query,
            "source_id": self.source_id,
            "source_fact": self.source_fact,
            "confidence": self.confidence,
            "was_helpful": self.was_helpful,
            "feedback_type": self.feedback_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class RAGSourceStats:
    """Aggregate statistics for RAG source effectiveness."""

    id: UUID
    source_id: str
    times_retrieved: int = 0
    times_helpful: int = 0
    times_not_helpful: int = 0
    avg_confidence: float = 0.0
    last_retrieved_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def helpfulness_rate(self) -> float:
        """Calculate the rate of helpful vs total feedback."""
        total_feedback = self.times_helpful + self.times_not_helpful
        if total_feedback == 0:
            return 0.0
        return self.times_helpful / total_feedback

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "source_id": self.source_id,
            "times_retrieved": self.times_retrieved,
            "times_helpful": self.times_helpful,
            "times_not_helpful": self.times_not_helpful,
            "avg_confidence": self.avg_confidence,
            "helpfulness_rate": self.helpfulness_rate,
            "last_retrieved_at": (
                self.last_retrieved_at.isoformat() if self.last_retrieved_at else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
