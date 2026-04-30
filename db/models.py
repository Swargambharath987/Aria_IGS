import uuid
import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, Enum as SAEnum,
    ForeignKey, Index, Integer, SmallInteger, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ── Enums ──────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    user  = "user"
    admin = "admin"


class MessageRole(str, enum.Enum):
    user      = "user"
    assistant = "assistant"
    tool      = "tool"


# ── Tables ─────────────────────────────────────────────────────────────────

class User(Base):
    """
    One row per IGS user.
    ldap_uid matches their IGS/Active Directory username.
    Until LDAP auth is wired (Phase 3), the caller supplies user_id in ChatRequest.
    """
    __tablename__ = "users"

    id           : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ldap_uid     : Mapped[str]              = mapped_column(String(64),  nullable=False, unique=True)
    email        : Mapped[Optional[str]]    = mapped_column(String(255), nullable=True,  unique=True)
    display_name : Mapped[str]              = mapped_column(String(255), nullable=False)
    lab          : Mapped[Optional[str]]    = mapped_column(String(255), nullable=True)
    role         : Mapped[UserRole]         = mapped_column(SAEnum(UserRole, name="user_role"), nullable=False, default=UserRole.user)
    is_active    : Mapped[bool]             = mapped_column(Boolean,     nullable=False, default=True)
    preferences  : Mapped[Optional[dict]]   = mapped_column(JSONB,       nullable=True,  default=dict)
    created_at   : Mapped[datetime]         = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions    : Mapped[list["Session"]]      = relationship(back_populates="user", cascade="all, delete-orphan")
    api_tokens  : Mapped[list["ApiToken"]]     = relationship(back_populates="user", cascade="all, delete-orphan")
    feedback    : Mapped[list["Feedback"]]     = relationship(back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_users_ldap_uid", "ldap_uid"),
    )


class Session(Base):
    """
    A conversation thread. One user has many sessions.
    title is auto-set from the first message of the session.
    """
    __tablename__ = "sessions"

    id            : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id       : Mapped[uuid.UUID]        = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title         : Mapped[Optional[str]]    = mapped_column(String(255), nullable=True)
    is_archived   : Mapped[bool]             = mapped_column(Boolean, nullable=False, default=False)
    created_at    : Mapped[datetime]         = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_active_at: Mapped[datetime]         = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    user     : Mapped["User"]          = relationship(back_populates="sessions")
    messages : Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")

    __table_args__ = (
        # Fast listing of a user's recent sessions
        Index("ix_sessions_user_recent", "user_id", "last_active_at"),
    )


class Message(Base):
    """
    A single turn in a conversation (user or assistant).
    sources_used: list of RAG chunks retrieved for this response.
    tool_calls: summary of tools the agent invoked (detailed in tool_executions).
    """
    __tablename__ = "messages"

    id           : Mapped[uuid.UUID]       = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id   : Mapped[uuid.UUID]       = mapped_column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role         : Mapped[MessageRole]     = mapped_column(SAEnum(MessageRole, name="message_role"), nullable=False)
    content      : Mapped[str]             = mapped_column(Text,    nullable=False)
    tool_calls   : Mapped[Optional[list]]  = mapped_column(JSONB,   nullable=True)   # [{tool, input, success}]
    sources_used : Mapped[Optional[list]]  = mapped_column(JSONB,   nullable=True)   # [{chunk_text, score, doc}]
    model_used   : Mapped[Optional[str]]   = mapped_column(String(128), nullable=True)
    latency_ms   : Mapped[Optional[int]]   = mapped_column(Integer, nullable=True)
    input_tokens : Mapped[Optional[int]]   = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]]   = mapped_column(Integer, nullable=True)
    created_at   : Mapped[datetime]        = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    session         : Mapped["Session"]             = relationship(back_populates="messages")
    feedback        : Mapped[Optional["Feedback"]]  = relationship(back_populates="message", uselist=False)
    tool_executions : Mapped[list["ToolExecution"]] = relationship(back_populates="message", cascade="all, delete-orphan")

    __table_args__ = (
        # Fast fetch of all messages in a session, ordered chronologically
        Index("ix_messages_session_time", "session_id", "created_at"),
    )


class Feedback(Base):
    """
    User rating on a single assistant message. One rating per message.
    rating: +1 (helpful) or -1 (not helpful).
    This is the primary training signal for future fine-tuning.
    """
    __tablename__ = "feedback"

    id         : Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id : Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    user_id    : Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), ForeignKey("users.id",    ondelete="CASCADE"), nullable=False)
    rating     : Mapped[int]            = mapped_column(SmallInteger, nullable=False)
    comment    : Mapped[Optional[str]]  = mapped_column(Text, nullable=True)
    created_at : Mapped[datetime]       = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    message : Mapped["Message"] = relationship(back_populates="feedback")
    user    : Mapped["User"]    = relationship(back_populates="feedback")

    __table_args__ = (
        UniqueConstraint("message_id", name="uq_feedback_message"),       # one rating per message
        CheckConstraint("rating IN (1, -1)", name="ck_feedback_rating"),  # only +1 or -1
        Index("ix_feedback_user", "user_id"),
    )


class KnowledgeDoc(Base):
    """
    Metadata for every document ingested into ChromaDB.
    ChromaDB holds the vectors; this table holds who added it, when, and how many chunks.
    is_active=False lets admins soft-deactivate a doc without losing the audit trail.
    """
    __tablename__ = "knowledge_docs"

    id            : Mapped[uuid.UUID]       = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename      : Mapped[str]             = mapped_column(String(512), nullable=False)
    original_name : Mapped[str]             = mapped_column(String(512), nullable=False)
    collection    : Mapped[str]             = mapped_column(String(128), nullable=False)
    source_url    : Mapped[Optional[str]]   = mapped_column(String(1024), nullable=True)
    ingested_by   : Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    ingested_at   : Mapped[datetime]        = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    chunk_count   : Mapped[int]             = mapped_column(Integer,    nullable=False, default=0)
    file_size_bytes: Mapped[Optional[int]]  = mapped_column(BigInteger, nullable=True)
    is_active     : Mapped[bool]            = mapped_column(Boolean,    nullable=False, default=True)
    doc_metadata  : Mapped[Optional[dict]]  = mapped_column(JSONB,      nullable=True)

    __table_args__ = (
        Index("ix_knowledge_docs_collection", "collection"),
        Index("ix_knowledge_docs_active",     "is_active"),
    )


class ToolExecution(Base):
    """
    Every MCP tool call logged here for observability and debugging.
    Answers: how often does slurm-mcp fail? what's the p95 latency of file-mcp?
    Links back to the message that triggered it.
    """
    __tablename__ = "tool_executions"

    id            : Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id    : Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    tool_name     : Mapped[str]            = mapped_column(String(128), nullable=False)
    input_params  : Mapped[Optional[dict]] = mapped_column(JSONB,       nullable=True)
    output        : Mapped[Optional[dict]] = mapped_column(JSONB,       nullable=True)
    success       : Mapped[bool]           = mapped_column(Boolean,     nullable=False)
    latency_ms    : Mapped[Optional[int]]  = mapped_column(Integer,     nullable=True)
    error_message : Mapped[Optional[str]]  = mapped_column(Text,        nullable=True)
    created_at    : Mapped[datetime]       = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    message : Mapped["Message"] = relationship(back_populates="tool_executions")

    __table_args__ = (
        Index("ix_tool_exec_message",   "message_id"),
        Index("ix_tool_exec_analytics", "tool_name", "success"),  # tool failure rate queries
        Index("ix_tool_exec_time",      "created_at"),             # time-range queries
    )


class ApiToken(Base):
    """
    Bearer tokens for CLI and non-browser access.
    token_hash is a bcrypt hash — the raw token is shown only once at creation and never stored.
    Browser users authenticate via LDAP/SSO (Phase 3) instead of API tokens.
    """
    __tablename__ = "api_tokens"

    id          : Mapped[uuid.UUID]         = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     : Mapped[uuid.UUID]         = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash  : Mapped[str]               = mapped_column(String(255), nullable=False, unique=True)
    name        : Mapped[str]               = mapped_column(String(128), nullable=False)
    last_used_at: Mapped[Optional[datetime]]= mapped_column(DateTime(timezone=True), nullable=True)
    expires_at  : Mapped[Optional[datetime]]= mapped_column(DateTime(timezone=True), nullable=True)
    is_active   : Mapped[bool]              = mapped_column(Boolean, nullable=False, default=True)
    created_at  : Mapped[datetime]          = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user : Mapped["User"] = relationship(back_populates="api_tokens")

    __table_args__ = (
        Index("ix_api_tokens_user",  "user_id"),
        Index("ix_api_tokens_hash",  "token_hash"),  # fast lookup on every request
    )
