"""Application settings.

All credentials are read from the server environment (or a local ``.env``
file).  They are never serialised to API responses or exposed to the browser.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import AliasChoices, Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class StorageType(StrEnum):
    LOCAL = "local"
    S3 = "s3"


class VectorStoreType(StrEnum):
    PGVECTOR = "pgvector"
    CHROMA = "chroma"
    QDRANT = "qdrant"
    MILVUS = "milvus"


class LogFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name: str = "Chatbot Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "development-secret-key-change-in-production"
    environment: Environment = Environment.DEVELOPMENT
    api_v1_prefix: str = "/api/v1"
    auto_create_tables: bool = True
    bootstrap_demo_data: bool = False
    admin_email: str | None = None
    admin_password: str | None = None

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot"
    database_pool_size: int = 20
    database_max_overflow: int = 40
    database_pool_recycle: int = 1800
    database_echo: bool = False

    # ------------------------------------------------------------------
    # Redis / Celery
    # ------------------------------------------------------------------
    redis_url: RedisDsn = "redis://localhost:6379/0"
    redis_max_connections: int = 50
    celery_broker_url: RedisDsn = "redis://localhost:6379/1"
    celery_result_backend: RedisDsn = "redis://localhost:6379/2"
    celery_task_always_eager: bool = False

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    storage_type: StorageType = StorageType.LOCAL
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket: str = "chatbot-uploads"
    s3_region: str = "us-east-1"
    local_storage_path: Path = Path("uploads")
    max_upload_size_mb: int = 25
    allowed_upload_extensions: list[str] = [
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".html",
        ".htm",
        ".json",
    ]
    allowed_image_extensions: list[str] = [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
    ]
    allowed_audio_extensions: list[str] = [
        ".mp3",
        ".wav",
        ".m4a",
        ".ogg",
        ".webm",
        ".flac",
    ]

    # ------------------------------------------------------------------
    # JWT / sessions
    # ------------------------------------------------------------------
    jwt_secret_key: str = "development-jwt-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "chatbot-platform"
    jwt_audience: str = "chatbot-platform-api"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    encryption_key: str | None = None

    # ------------------------------------------------------------------
    # OAuth (authorization-code exchange is intentionally delegated to a
    # configured identity provider; the architecture is OAuth-ready)
    # ------------------------------------------------------------------
    oauth_google_client_id: str | None = None
    oauth_google_client_secret: str | None = None
    oauth_github_client_id: str | None = None
    oauth_github_client_secret: str | None = None

    # ------------------------------------------------------------------
    # AI providers
    # ------------------------------------------------------------------
    openai_api_key: str | None = None
    openai_api_base: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_default_model: str = "claude-3-5-sonnet-latest"
    google_api_key: str | None = None
    google_default_model: str = "gemini-2.0-flash"
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_default_deployment: str | None = None
    groq_api_key: str | None = None
    groq_default_model: str = "llama-3.3-70b-versatile"
    deepseek_api_key: str | None = None
    deepseek_default_model: str = "deepseek-chat"
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2"
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str | None = None
    lm_studio_base_url: str = "http://localhost:1234"
    lm_studio_api_key: str | None = None

    # Hugging Face first-class configuration
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_API_KEY"),
    )
    hf_default_model: str = "openai/gpt-oss-120b"
    hf_default_provider: str = "auto"
    hf_endpoint_url: str | None = None
    hf_timeout: float = 120.0
    hf_max_retries: int = 3
    hf_router_url: str = "https://router.huggingface.co/v1"
    hf_hub_url: str = "https://huggingface.co"
    hf_model_cache_ttl: int = 300
    hf_embedding_model: str = "BAAI/bge-m3"
    hf_embedding_provider: str | None = None
    hf_asr_model: str = "openai/whisper-large-v3"
    hf_tts_model: str = "hexgrad/Kokoro-82M"

    provider_timeout: float = 120.0
    provider_max_retries: int = 3
    provider_model_cache_ttl: int = 300

    # ------------------------------------------------------------------
    # Embeddings / vector store
    # ------------------------------------------------------------------
    embedding_provider: str = "huggingface"
    embedding_model: str | None = None
    embedding_dimensions: int = 1024
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text"
    vector_store_type: VectorStoreType = VectorStoreType.PGVECTOR
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    rag_top_k: int = 6
    rag_similarity_threshold: float = 0.25
    rag_hybrid_alpha: float = 0.65
    rag_max_context_chars: int = 12000
    chat_context_messages: int = 30

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 120
    rate_limit_period: int = 60
    auth_rate_limit_requests: int = 10
    auth_rate_limit_period: int = 60
    stream_rate_limit_requests: int = 30
    stream_rate_limit_period: int = 60

    # ------------------------------------------------------------------
    # CORS / proxy
    # ------------------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    trusted_hosts: list[str] = ["*"]

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("allowed_upload_extensions", mode="before")
    @classmethod
    def parse_upload_extensions(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip().lower() for item in value.split(",") if item.strip()]
        return [str(item).lower() for item in value]

    # ------------------------------------------------------------------
    # Logging / observability
    # ------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: LogFormat = LogFormat.JSON
    sentry_dsn: str | None = None
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None
    metrics_enabled: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    @property
    def sync_database_url(self) -> str:
        url = str(self.database_url)
        if url.startswith("postgresql+asyncpg"):
            return url.replace("postgresql+asyncpg", "postgresql+psycopg")
        if url.startswith("sqlite+aiosqlite"):
            return url.replace("sqlite+aiosqlite", "sqlite")
        return url

    @property
    def is_postgres(self) -> bool:
        return str(self.database_url).startswith("postgresql")

    @property
    def is_sqlite(self) -> bool:
        return str(self.database_url).startswith("sqlite")

    def resolved_embedding_model(self, provider: str | None = None) -> str:
        provider = provider or self.embedding_provider
        if self.embedding_model:
            return self.embedding_model
        if provider == "huggingface":
            return self.hf_embedding_model
        if provider == "ollama":
            return self.ollama_embedding_model
        if provider == "openai":
            return self.openai_embedding_model
        return self.hf_embedding_model

    def validate_production_secrets(self) -> None:
        """Fail fast when production starts with development credentials."""
        if not self.is_production:
            return
        weak = {
            "secret_key": "development-secret-key-change-in-production",
            "jwt_secret_key": "development-jwt-secret-change-in-production",
        }
        for name, default in weak.items():
            if getattr(self, name) == default or len(getattr(self, name)) < 32:
                raise RuntimeError(
                    f"{name.upper()} must be a strong, unique value in production"
                )
        if "*" in self.cors_origins:
            raise RuntimeError("Wildcard CORS origins are not allowed in production")


settings = Settings()
