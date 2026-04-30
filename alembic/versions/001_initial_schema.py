"""Initial schema — all Aria tables

Revision ID: 001
Revises:
Create Date: 2026-04-29
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── PostgreSQL enums ────────────────────────────────────────────────────
    op.execute("CREATE TYPE user_role    AS ENUM ('user', 'admin')")
    op.execute("CREATE TYPE message_role AS ENUM ('user', 'assistant', 'tool')")

    # ── users ───────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id",             UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("ldap_uid",       sa.String(64),  nullable=False),
        sa.Column("email",          sa.String(255), nullable=True),
        sa.Column("display_name",   sa.String(255), nullable=False),
        sa.Column("lab",            sa.String(255), nullable=True),
        sa.Column("role",           sa.Enum("user", "admin", name="user_role", create_type=False), nullable=False, server_default="user"),
        sa.Column("is_active",      sa.Boolean(),   nullable=False, server_default="true"),
        sa.Column("preferences",    JSONB,          nullable=True,  server_default="{}"),
        sa.Column("created_at",     sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("ldap_uid", name="uq_users_ldap_uid"),
        sa.UniqueConstraint("email",    name="uq_users_email"),
    )
    op.create_index("ix_users_ldap_uid", "users", ["ldap_uid"])

    # ── sessions ────────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id",             UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",        UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title",          sa.String(255), nullable=True),
        sa.Column("is_archived",    sa.Boolean(),   nullable=False, server_default="false"),
        sa.Column("created_at",     sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_sessions_user_recent", "sessions", ["user_id", "last_active_at"])

    # ── messages ────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id",            UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id",    UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role",          sa.Enum("user", "assistant", "tool", name="message_role", create_type=False), nullable=False),
        sa.Column("content",       sa.Text(),  nullable=False),
        sa.Column("tool_calls",    JSONB,      nullable=True),
        sa.Column("sources_used",  JSONB,      nullable=True),
        sa.Column("model_used",    sa.String(128), nullable=True),
        sa.Column("latency_ms",    sa.Integer(), nullable=True),
        sa.Column("input_tokens",  sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_messages_session_time", "messages", ["session_id", "created_at"])

    # ── feedback ────────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id",         UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id",    UUID(as_uuid=True), sa.ForeignKey("users.id",    ondelete="CASCADE"), nullable=False),
        sa.Column("rating",     sa.SmallInteger(), nullable=False),
        sa.Column("comment",    sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("message_id", name="uq_feedback_message"),
        sa.CheckConstraint("rating IN (1, -1)", name="ck_feedback_rating"),
    )
    op.create_index("ix_feedback_user",    "feedback", ["user_id"])
    op.create_index("ix_feedback_message", "feedback", ["message_id"])

    # ── knowledge_docs ──────────────────────────────────────────────────────
    op.create_table(
        "knowledge_docs",
        sa.Column("id",              UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename",        sa.String(512), nullable=False),
        sa.Column("original_name",   sa.String(512), nullable=False),
        sa.Column("collection",      sa.String(128), nullable=False),
        sa.Column("source_url",      sa.String(1024), nullable=True),
        sa.Column("ingested_by",     UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("ingested_at",     sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("chunk_count",     sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("is_active",       sa.Boolean(),   nullable=False, server_default="true"),
        sa.Column("doc_metadata",    JSONB,          nullable=True),
    )
    op.create_index("ix_knowledge_docs_collection", "knowledge_docs", ["collection"])
    op.create_index("ix_knowledge_docs_active",     "knowledge_docs", ["is_active"])

    # ── tool_executions ─────────────────────────────────────────────────────
    op.create_table(
        "tool_executions",
        sa.Column("id",            UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("message_id",    UUID(as_uuid=True), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tool_name",     sa.String(128), nullable=False),
        sa.Column("input_params",  JSONB, nullable=True),
        sa.Column("output",        JSONB, nullable=True),
        sa.Column("success",       sa.Boolean(), nullable=False),
        sa.Column("latency_ms",    sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_tool_exec_message",   "tool_executions", ["message_id"])
    op.create_index("ix_tool_exec_analytics", "tool_executions", ["tool_name", "success"])
    op.create_index("ix_tool_exec_time",      "tool_executions", ["created_at"])

    # ── api_tokens ──────────────────────────────────────────────────────────
    op.create_table(
        "api_tokens",
        sa.Column("id",           UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id",      UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash",   sa.String(255), nullable=False),
        sa.Column("name",         sa.String(128), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active",    sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at",   sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("token_hash", name="uq_api_tokens_hash"),
    )
    op.create_index("ix_api_tokens_user", "api_tokens", ["user_id"])
    op.create_index("ix_api_tokens_hash", "api_tokens", ["token_hash"])


def downgrade() -> None:
    op.drop_table("api_tokens")
    op.drop_table("tool_executions")
    op.drop_table("knowledge_docs")
    op.drop_table("feedback")
    op.drop_table("messages")
    op.drop_table("sessions")
    op.drop_table("users")
    op.execute("DROP TYPE message_role")
    op.execute("DROP TYPE user_role")
