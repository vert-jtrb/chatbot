"""Initial schema.

Revision ID: 0001
Revises:
Create Date: 2026-11-03
"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.types import CHAR, TypeDecorator

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


class GUID(TypeDecorator):
    """Portable UUID column matching :class:`app.database.base.GUID`."""

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=False))
        return dialect.type_descriptor(CHAR(36))


def _vector_type() -> Any:
    """Use a native pgvector column on PostgreSQL, JSON elsewhere."""

    if op.get_bind().dialect.name == "postgresql":
        from pgvector.sqlalchemy import Vector

        return Vector(1024)
    return sa.JSON()


def upgrade() -> None:
    is_postgres = op.get_bind().dialect.name == "postgresql"
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "ai_provider_configurations",
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("provider_type", sa.String(length=50), nullable=False),
        sa.Column("api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("base_url", sa.String(length=1024), nullable=True),
        sa.Column("default_model", sa.String(length=255), nullable=True),
        sa.Column("default_inference_provider", sa.String(length=100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("timeout", sa.Float(), nullable=True),
        sa.Column("max_retries", sa.Integer(), nullable=True),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("health_status", sa.String(length=20), nullable=False),
        sa.Column("last_validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_ai_provider_configurations_name",
        "ai_provider_configurations",
        ["name"],
        unique=True,
    )
    op.create_index(
        "ix_ai_provider_configurations_deleted_at",
        "ai_provider_configurations",
        ["deleted_at"],
    )
    op.create_index(
        "ix_ai_provider_configurations_is_deleted",
        "ai_provider_configurations",
        ["is_deleted"],
    )

    op.create_table(
        "permissions",
        sa.Column("resource", sa.String(length=100), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("resource", "action", name="uq_permission"),
    )

    op.create_table(
        "roles",
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_system", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "system_settings",
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("value_json", sa.JSON(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("is_encrypted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_system_settings_key", "system_settings", ["key"], unique=True)

    op.create_table(
        "ai_models",
        sa.Column("provider_id", GUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=100), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("task", sa.String(length=100), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("context_length", sa.Integer(), nullable=True),
        sa.Column("supports_streaming", sa.Boolean(), nullable=False),
        sa.Column("supports_tools", sa.Boolean(), nullable=False),
        sa.Column("supports_vision", sa.Boolean(), nullable=False),
        sa.Column("supports_embeddings", sa.Boolean(), nullable=False),
        sa.Column("supports_audio", sa.Boolean(), nullable=False),
        sa.Column("supports_json_mode", sa.Boolean(), nullable=False),
        sa.Column("input_price_per_million", sa.Float(), nullable=True),
        sa.Column("output_price_per_million", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["provider_id"], ["ai_provider_configurations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider_id", "model_id", name="uq_model_per_provider"),
    )
    op.create_index("ix_ai_models_provider_id", "ai_models", ["provider_id"])
    op.create_index("ix_ai_models_model_id", "ai_models", ["model_id"])

    op.create_table(
        "role_permissions",
        sa.Column("role_id", GUID(), nullable=False),
        sa.Column("permission_id", GUID(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["permission_id"], ["permissions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_role_permissions_role_id", "role_permissions", ["role_id"])
    op.create_index(
        "ix_role_permissions_permission_id", "role_permissions", ["permission_id"]
    )

    op.create_table(
        "users",
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=True),
        sa.Column("display_name", sa.String(length=255), nullable=False),
        sa.Column("avatar_url", sa.String(length=512), nullable=True),
        sa.Column("auth_provider", sa.String(length=32), nullable=False),
        sa.Column("auth_provider_id", sa.String(length=255), nullable=True),
        sa.Column("role_id", GUID(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_superuser", sa.Boolean(), nullable=False),
        sa.Column("two_factor_enabled", sa.Boolean(), nullable=False),
        sa.Column("two_factor_secret", sa.String(length=255), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("preferences", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["role_id"], ["roles.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_auth_provider_id", "users", ["auth_provider_id"])
    op.create_index("ix_users_deleted_at", "users", ["deleted_at"])
    op.create_index("ix_users_is_deleted", "users", ["is_deleted"])

    op.create_table(
        "audit_logs",
        sa.Column("user_id", GUID(), nullable=True),
        sa.Column("action", sa.String(length=255), nullable=False),
        sa.Column("resource", sa.String(length=255), nullable=False),
        sa.Column("resource_id", sa.String(length=64), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])
    op.create_index("ix_audit_logs_resource", "audit_logs", ["resource"])
    op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"])
    op.create_index("ix_audit_logs_severity", "audit_logs", ["severity"])

    op.create_table(
        "chats",
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("provider_name", sa.String(length=100), nullable=True),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("model_id", GUID(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("top_p", sa.Float(), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["model_id"], ["ai_models.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chats_user_id", "chats", ["user_id"])

    op.create_table(
        "knowledge_bases",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("embedding_provider", sa.String(length=100), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("retrieval_config", sa.JSON(), nullable=True),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_bases_user_id", "knowledge_bases", ["user_id"])
    op.create_index("ix_knowledge_bases_deleted_at", "knowledge_bases", ["deleted_at"])
    op.create_index("ix_knowledge_bases_is_deleted", "knowledge_bases", ["is_deleted"])

    op.create_table(
        "notifications",
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("is_read", sa.Boolean(), nullable=False),
        sa.Column("is_dismissed", sa.Boolean(), nullable=False),
        sa.Column("action_url", sa.String(length=512), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])

    op.create_table(
        "organizations",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("avatar_url", sa.String(length=512), nullable=True),
        sa.Column("owner_id", GUID(), nullable=False),
        sa.Column("is_personal", sa.Boolean(), nullable=False),
        sa.Column("max_members", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_organizations_slug", "organizations", ["slug"], unique=True)
    op.create_index("ix_organizations_owner_id", "organizations", ["owner_id"])
    op.create_index("ix_organizations_deleted_at", "organizations", ["deleted_at"])
    op.create_index("ix_organizations_is_deleted", "organizations", ["is_deleted"])

    op.create_table(
        "refresh_tokens",
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("family_id", GUID(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_revoked", sa.Boolean(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_hash", sa.String(length=64), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"], unique=True
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_family_id", "refresh_tokens", ["family_id"])
    op.create_index("ix_refresh_tokens_deleted_at", "refresh_tokens", ["deleted_at"])
    op.create_index("ix_refresh_tokens_is_deleted", "refresh_tokens", ["is_deleted"])

    op.create_table(
        "user_api_keys",
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("key_hash", sa.String(length=255), nullable=False),
        sa.Column("key_prefix", sa.String(length=16), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key_hash"),
    )
    op.create_index("ix_user_api_keys_user_id", "user_api_keys", ["user_id"])
    op.create_index("ix_user_api_keys_deleted_at", "user_api_keys", ["deleted_at"])
    op.create_index("ix_user_api_keys_is_deleted", "user_api_keys", ["is_deleted"])

    op.create_table(
        "user_settings",
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("value_json", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "key", name="uq_user_setting"),
    )
    op.create_index("ix_user_settings_user_id", "user_settings", ["user_id"])

    op.create_table(
        "conversations",
        sa.Column("chat_id", GUID(), nullable=True),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("provider_name", sa.String(length=100), nullable=True),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("model_id", GUID(), nullable=True),
        sa.Column("knowledge_base_id", GUID(), nullable=True),
        sa.Column("rag_enabled", sa.Boolean(), nullable=False),
        sa.Column("retrieval_config", sa.JSON(), nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("top_p", sa.Float(), nullable=True),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("is_pinned", sa.Boolean(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["model_id"], ["ai_models.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_conversations_chat_id", "conversations", ["chat_id"])
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index("ix_conversations_status", "conversations", ["status"])
    op.create_index(
        "ix_conversations_knowledge_base_id", "conversations", ["knowledge_base_id"]
    )
    op.create_index(
        "ix_conversations_last_message_at", "conversations", ["last_message_at"]
    )
    op.create_index("ix_conversations_deleted_at", "conversations", ["deleted_at"])
    op.create_index("ix_conversations_is_deleted", "conversations", ["is_deleted"])

    op.create_table(
        "documents",
        sa.Column("knowledge_base_id", GUID(), nullable=False),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(length=127), nullable=False),
        sa.Column("document_type", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_knowledge_base_id", "documents", ["knowledge_base_id"])
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_deleted_at", "documents", ["deleted_at"])
    op.create_index("ix_documents_is_deleted", "documents", ["is_deleted"])
    op.create_index(
        "ix_documents_kb_status", "documents", ["knowledge_base_id", "status"]
    )

    op.create_table(
        "organization_members",
        sa.Column("organization_id", GUID(), nullable=False),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_organization_members_organization_id",
        "organization_members",
        ["organization_id"],
    )
    op.create_index("ix_organization_members_user_id", "organization_members", ["user_id"])

    op.create_table(
        "prompt_templates",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=100), nullable=True),
        sa.Column("user_id", GUID(), nullable=True),
        sa.Column("organization_id", GUID(), nullable=True),
        sa.Column("is_global", sa.Boolean(), nullable=False),
        sa.Column("is_system", sa.Boolean(), nullable=False),
        sa.Column("variables", sa.JSON(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["organization_id"], ["organizations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prompt_templates_user_id", "prompt_templates", ["user_id"])
    op.create_index(
        "ix_prompt_templates_organization_id", "prompt_templates", ["organization_id"]
    )
    op.create_index("ix_prompt_templates_category", "prompt_templates", ["category"])
    op.create_index("ix_prompt_templates_deleted_at", "prompt_templates", ["deleted_at"])
    op.create_index(
        "ix_prompt_templates_is_deleted", "prompt_templates", ["is_deleted"]
    )

    op.create_table(
        "document_chunks",
        sa.Column("document_id", GUID(), nullable=False),
        sa.Column("knowledge_base_id", GUID(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunk_index"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index(
        "ix_document_chunks_knowledge_base_id", "document_chunks", ["knowledge_base_id"]
    )

    op.create_table(
        "messages",
        sa.Column("conversation_id", GUID(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("provider_name", sa.String(length=100), nullable=True),
        sa.Column("model_name", sa.String(length=255), nullable=True),
        sa.Column("model_id", GUID(), nullable=True),
        sa.Column("provider_id", GUID(), nullable=True),
        sa.Column("tool_calls", sa.JSON(), nullable=True),
        sa.Column("tool_call_id", sa.String(length=255), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("finish_reason", sa.String(length=50), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("is_edited", sa.Boolean(), nullable=False),
        sa.Column("parent_id", GUID(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["model_id"], ["ai_models.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["provider_id"], ["ai_provider_configurations.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["parent_id"], ["messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    op.create_table(
        "usage_records",
        sa.Column("user_id", GUID(), nullable=True),
        sa.Column("conversation_id", GUID(), nullable=True),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("operation", sa.String(length=50), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_records_user_id", "usage_records", ["user_id"])
    op.create_index("ix_usage_records_conversation_id", "usage_records", ["conversation_id"])
    op.create_index("ix_usage_records_provider", "usage_records", ["provider"])
    op.create_index("ix_usage_records_model", "usage_records", ["model"])
    op.create_index("ix_usage_records_operation", "usage_records", ["operation"])
    op.create_index("ix_usage_records_status", "usage_records", ["status"])
    op.create_index("ix_usage_records_request_id", "usage_records", ["request_id"])

    op.create_table(
        "embeddings",
        sa.Column("chunk_id", GUID(), nullable=False),
        sa.Column("knowledge_base_id", GUID(), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("vector_store", sa.String(length=50), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("vector", _vector_type(), nullable=True),
        sa.Column("norm", sa.Float(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["chunk_id"], ["document_chunks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"], ["knowledge_bases.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_embeddings_chunk_id", "embeddings", ["chunk_id"], unique=True)
    op.create_index(
        "ix_embeddings_knowledge_base_id", "embeddings", ["knowledge_base_id"]
    )
    if is_postgres:
        op.create_index(
            "ix_embeddings_vector_hnsw",
            "embeddings",
            ["vector"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"vector": "vector_cosine_ops"},
        )

    op.create_table(
        "message_attachments",
        sa.Column("message_id", GUID(), nullable=False),
        sa.Column("user_id", GUID(), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("mime_type", sa.String(length=127), nullable=False),
        sa.Column("storage_type", sa.String(length=50), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_message_attachments_message_id", "message_attachments", ["message_id"]
    )
    op.create_index("ix_message_attachments_user_id", "message_attachments", ["user_id"])

    op.create_table(
        "vector_indices",
        sa.Column("document_chunk_id", GUID(), nullable=True),
        sa.Column("vector_store_type", sa.String(length=50), nullable=False),
        sa.Column("vector_id", sa.String(length=255), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("id", GUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_chunk_id"], ["document_chunks.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_vector_indices_document_chunk_id", "vector_indices", ["document_chunk_id"]
    )
    op.create_index("ix_vector_indices_vector_id", "vector_indices", ["vector_id"])


def downgrade() -> None:
    op.drop_table("vector_indices")
    op.drop_table("message_attachments")
    op.drop_table("embeddings")
    op.drop_table("usage_records")
    op.drop_table("messages")
    op.drop_table("document_chunks")
    op.drop_table("prompt_templates")
    op.drop_table("organization_members")
    op.drop_table("documents")
    op.drop_table("conversations")
    op.drop_table("user_settings")
    op.drop_table("user_api_keys")
    op.drop_table("refresh_tokens")
    op.drop_table("organizations")
    op.drop_table("notifications")
    op.drop_table("knowledge_bases")
    op.drop_table("chats")
    op.drop_table("audit_logs")
    op.drop_table("users")
    op.drop_table("role_permissions")
    op.drop_table("ai_models")
    op.drop_table("system_settings")
    op.drop_table("roles")
    op.drop_table("permissions")
    op.drop_table("ai_provider_configurations")
