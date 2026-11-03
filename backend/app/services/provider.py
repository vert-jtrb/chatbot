from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import asdict
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security.encryption import decrypt_secret, encrypt_secret, mask_secret
from app.models.provider import AIModel, AIProvider, ProviderType
from app.providers.ai.base import AIProvider as RuntimeProvider
from app.providers.ai.base import ProviderConfig
from app.providers.ai.exceptions import ProviderError
from app.providers.ai.huggingface.provider import HuggingFaceProvider
from app.providers.ai.registry import (
    PROVIDER_CLASSES,
    create_provider,
    default_config,
    has_credentials,
)
from app.repositories.provider import AIModelRepository, AIProviderRepository
from app.schemas.provider import (
    AIProviderCreate,
    AIProviderUpdate,
)
from app.services.base import ServiceResult


class ProviderService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.providers = AIProviderRepository(session)
        self.models = AIModelRepository(session)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    async def list_providers(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        providers = (
            await self.providers.get_active()
            if active_only
            else (await self.providers.get_many(limit=200))[0]
        )
        return [await self.serialize(provider) for provider in providers]

    async def get_provider(self, provider_id: str) -> ServiceResult:
        provider = await self.providers.get(provider_id)
        if provider is None or provider.is_deleted:
            return ServiceResult.error_response("Provider not found", 404)
        return ServiceResult.ok(await self.serialize(provider))

    async def create_provider(self, data: AIProviderCreate) -> ServiceResult:
        provider_type = data.provider_type.lower()
        if provider_type not in {item.value for item in ProviderType}:
            return ServiceResult.error_response(
                f"Unsupported provider type: {data.provider_type}", 400
            )
        if await self.providers.get_by_name(data.name):
            return ServiceResult.error_response("Provider name already exists", 409)
        if data.is_default:
            await self.providers.clear_default()
        provider = await self.providers.create(
            name=data.name,
            display_name=data.display_name or provider_type.replace("_", " ").title(),
            provider_type=provider_type,
            api_key_encrypted=encrypt_secret(data.api_key) if data.api_key else None,
            base_url=data.base_url,
            default_model=data.default_model,
            default_inference_provider=data.default_inference_provider,
            timeout=data.timeout or settings.provider_timeout,
            max_retries=(
                data.max_retries
                if data.max_retries is not None
                else settings.provider_max_retries
            ),
            config=data.config,
            is_active=data.is_active,
            is_default=data.is_default,
        )
        return ServiceResult.ok(await self.serialize(provider), status_code=201)

    async def update_provider(
        self, provider_id: str, data: AIProviderUpdate
    ) -> ServiceResult:
        provider = await self.providers.get(provider_id)
        if provider is None or provider.is_deleted:
            return ServiceResult.error_response("Provider not found", 404)
        if data.is_default:
            await self.providers.clear_default(except_id=provider.id)
        payload = data.model_dump(exclude_unset=True)
        payload.pop("clear_api_key", None)
        if data.clear_api_key:
            provider.api_key_encrypted = None
        if "api_key" in payload:
            api_key = payload.pop("api_key")
            if api_key:
                provider.api_key_encrypted = encrypt_secret(api_key)
        for key, value in payload.items():
            setattr(provider, key, value)
        await self.session.flush()
        return ServiceResult.ok(await self.serialize(provider))

    async def delete_provider(self, provider_id: str) -> ServiceResult:
        provider = await self.providers.get(provider_id)
        if provider is None or provider.is_deleted:
            return ServiceResult.error_response("Provider not found", 404)
        await self.providers.delete(provider_id, soft=True)
        return ServiceResult.ok({"message": "Provider deleted", "id": provider_id})

    async def serialize(self, provider: AIProvider) -> dict[str, Any]:
        count = await self.session.scalar(
            select(func.count())
            .select_from(AIModel)
            .where(AIModel.provider_id == provider.id, AIModel.is_active.is_(True))
        )
        key = None
        if provider.api_key_encrypted:
            try:
                key = mask_secret(decrypt_secret(provider.api_key_encrypted))
            except Exception:
                key = "********"
        return {
            "id": provider.id,
            "name": provider.name,
            "display_name": provider.display_name,
            "provider_type": provider.provider_type,
            "base_url": provider.base_url,
            "default_model": provider.default_model,
            "default_inference_provider": provider.default_inference_provider,
            "is_active": provider.is_active,
            "is_default": provider.is_default,
            "timeout": provider.timeout,
            "max_retries": provider.max_retries,
            "config": provider.config,
            "api_key_hint": key,
            "health_status": provider.health_status,
            "last_validated_at": provider.last_validated_at,
            "last_error": provider.last_error,
            "model_count": int(count or 0),
            "created_at": provider.created_at,
            "updated_at": provider.updated_at,
        }

    # ------------------------------------------------------------------
    # Runtime resolution
    # ------------------------------------------------------------------
    async def resolve(
        self,
        *,
        provider_name: str | None = None,
        model_id: str | None = None,
        model_name: str | None = None,
    ) -> tuple[RuntimeProvider, AIProvider | None, AIModel | None, str]:
        """Resolve a runnable provider and model identifier.

        Priority: explicit ``model_id`` -> provider name -> configured default
        -> first active provider with credentials.
        """

        db_provider: AIProvider | None = None
        db_model: AIModel | None = None
        if model_id:
            db_model = await self.models.get_with_provider(model_id)
            if db_model is None or not db_model.is_active:
                raise ProviderError("The selected model is not available")
            db_provider = db_model.provider
        if db_provider is None and provider_name:
            db_provider = await self.providers.get_by_name(provider_name)
            if db_provider is None and provider_name in {
                item.value for item in ProviderType
            }:
                db_provider = await self._ensure_env_provider(provider_name)
        if db_provider is None and not provider_name and not model_id:
            db_provider = await self.providers.get_default()
            if db_provider is None:
                db_provider = await self._first_available()
        if db_provider is None:
            raise ProviderError(
                "No AI provider is configured. Add a provider in the admin panel "
                "or set the corresponding API key in the environment."
            )
        if not db_provider.is_active:
            raise ProviderError(f"Provider '{db_provider.name}' is disabled")
        runtime = self.runtime_for(db_provider)
        resolved_model = (
            db_model.model_id
            if db_model is not None
            else model_name or db_provider.default_model or ""
        )
        return runtime, db_provider, db_model, resolved_model

    def runtime_for(self, db_provider: AIProvider) -> RuntimeProvider:
        api_key = None
        if db_provider.api_key_encrypted:
            api_key = decrypt_secret(db_provider.api_key_encrypted)
        env_default = default_config(db_provider.provider_type)
        config = ProviderConfig(
            name=db_provider.name,
            provider_type=db_provider.provider_type,
            api_key=api_key or env_default.api_key,
            base_url=db_provider.base_url or env_default.base_url,
            default_model=db_provider.default_model or env_default.default_model,
            default_inference_provider=(
                db_provider.default_inference_provider
                or env_default.default_inference_provider
            ),
            timeout=db_provider.timeout or settings.provider_timeout,
            max_retries=(
                db_provider.max_retries
                if db_provider.max_retries is not None
                else settings.provider_max_retries
            ),
            extra={**env_default.extra, **(db_provider.config or {})},
        )
        return create_provider(config)

    async def _first_available(self) -> AIProvider | None:
        for provider in await self.providers.get_active():
            if provider.api_key_encrypted or has_credentials(provider.provider_type):
                return provider
        return None

    async def _ensure_env_provider(self, provider_type: str) -> AIProvider | None:
        if not has_credentials(provider_type):
            return None
        existing = await self.providers.get_by_name(provider_type)
        if existing:
            return existing
        defaults = default_config(provider_type)
        return await self.providers.create(
            name=provider_type,
            display_name=defaults.name.replace("_", " ").title(),
            provider_type=provider_type,
            base_url=defaults.base_url,
            default_model=defaults.default_model,
            default_inference_provider=defaults.default_inference_provider,
            timeout=defaults.timeout,
            max_retries=defaults.max_retries,
            config={},
            is_active=True,
        )

    async def bootstrap_from_environment(self) -> dict[str, int]:
        """Materialise environment-configured providers as database rows.

        This keeps ``HF_TOKEN``/``OPENAI_API_KEY`` usable without a manual admin
        step while still allowing the database to override any value later.
        Soft-deleted rows are revived rather than recreated so unique names are
        never violated.
        """

        created = 0
        revived = 0
        models = 0
        for provider_type in PROVIDER_CLASSES:
            if not has_credentials(provider_type):
                continue
            defaults = default_config(provider_type)
            provider = await self.providers.get_by_name(
                provider_type, include_deleted=True
            )
            if provider is None:
                provider = await self.providers.create(
                    name=provider_type,
                    display_name=defaults.name.replace("_", " ").title(),
                    provider_type=provider_type,
                    base_url=defaults.base_url,
                    default_model=defaults.default_model,
                    default_inference_provider=defaults.default_inference_provider,
                    timeout=defaults.timeout,
                    max_retries=defaults.max_retries,
                    config={},
                    is_active=True,
                )
                created += 1
            elif provider.is_deleted:
                provider.is_deleted = False
                provider.deleted_at = None
                provider.is_active = True
                revived += 1
            if provider.default_model:
                existing_model = await self.models.get_by_model_id(
                    provider.default_model, provider.id
                )
                if existing_model is None:
                    await self.models.create(
                        provider_id=provider.id,
                        name=provider.default_model,
                        model_id=provider.default_model,
                        is_default=True,
                    )
                    models += 1
        if await self.providers.get_default() is None:
            active = await self.providers.get_active()
            if active:
                active[0].is_default = True
        await self.session.flush()
        return {"created": created, "revived": revived, "models": models}


    # ------------------------------------------------------------------
    # Model discovery / health
    # ------------------------------------------------------------------
    async def sync_models(self, provider_id: str, *, refresh: bool = True) -> ServiceResult:
        provider = await self.providers.get(provider_id)
        if provider is None or provider.is_deleted:
            return ServiceResult.error_response("Provider not found", 404)
        runtime = self.runtime_for(provider)
        try:
            discovered = await runtime.get_models()
        except ProviderError as exc:
            provider.health_status = "unhealthy"
            provider.last_error = str(exc)
            provider.last_validated_at = datetime.now(UTC)
            await self.session.flush()
            return ServiceResult.error_response(str(exc), 502)
        except Exception as exc:
            provider.health_status = "unhealthy"
            provider.last_error = str(exc)
            provider.last_validated_at = datetime.now(UTC)
            await self.session.flush()
            return ServiceResult.error_response(
                f"Model discovery failed: {exc}", 502
            )
        finally:
            await runtime.aclose()

        created = 0
        updated = 0
        stored_ids: set[str] = set()
        for item in discovered:
            model_id = str(item.get("id") or "").strip()
            if not model_id:
                continue
            payload = self._model_payload(item)
            existing = await self.models.get_by_model_id(model_id, provider.id)
            if existing is None:
                await self.models.create(provider_id=provider.id, **payload)
                created += 1
            else:
                for key, value in payload.items():
                    setattr(existing, key, value)
                updated += 1
            stored_ids.add(model_id)
        if provider.default_model and provider.default_model not in stored_ids:
            await self.models.upsert(
                provider.id,
                {
                    "name": provider.default_model,
                    "model_id": provider.default_model,
                    "is_default": True,
                },
            )
        provider.health_status = "healthy"
        provider.last_error = None
        provider.last_validated_at = datetime.now(UTC)
        await self.session.flush()
        models = await self.models.get_by_provider(provider.id)
        return ServiceResult.ok(
            {
                "provider_id": provider.id,
                "discovered": len(discovered),
                "created": created,
                "updated": updated,
                "models": [self.serialize_model(model, provider) for model in models],
            }
        )

    def _model_payload(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": item.get("name") or item["id"],
            "model_id": item["id"],
            "description": item.get("description"),
            "task": item.get("pipeline_tag") or item.get("task"),
            "context_length": _int_or_none(
                item.get("context_length")
                or (item.get("metadata") or {}).get("context_length")
            ),
            "max_tokens": _int_or_none(item.get("max_tokens")),
            "supports_streaming": bool(item.get("supports_streaming", True)),
            "supports_tools": bool(item.get("supports_tools", False)),
            "supports_vision": bool(item.get("supports_vision", False)),
            "supports_embeddings": bool(item.get("supports_embeddings", False)),
            "supports_audio": bool(item.get("supports_audio", False)),
            "supports_json_mode": bool(item.get("supports_json_mode", False)),
            "input_price_per_million": _float_or_none(
                item.get("input_price_per_million")
            ),
            "output_price_per_million": _float_or_none(
                item.get("output_price_per_million")
            ),
            "metadata_": item.get("metadata"),
        }

    async def health_check(self, provider_id: str) -> ServiceResult:
        provider = await self.providers.get(provider_id)
        if provider is None or provider.is_deleted:
            return ServiceResult.error_response("Provider not found", 404)
        runtime = self.runtime_for(provider)
        try:
            healthy = await runtime.health_check()
        finally:
            await runtime.aclose()
        provider.health_status = "healthy" if healthy else "unhealthy"
        provider.last_validated_at = datetime.now(UTC)
        provider.last_error = None if healthy else "Health check failed"
        await self.session.flush()
        return ServiceResult.ok(
            {
                "provider_id": provider.id,
                "healthy": healthy,
                "status": provider.health_status,
                "error": provider.last_error,
                "checked_at": provider.last_validated_at,
            }
        )

    async def validate_credentials(self, provider_id: str) -> ServiceResult:
        provider = await self.providers.get(provider_id)
        if provider is None or provider.is_deleted:
            return ServiceResult.error_response("Provider not found", 404)
        runtime = self.runtime_for(provider)
        try:
            if isinstance(runtime, HuggingFaceProvider):
                account = await runtime.validate_token()
                result = {"valid": True, "account": account, "error": None}
            else:
                healthy = await runtime.health_check()
                result = {
                    "valid": healthy,
                    "account": None,
                    "error": None if healthy else "Provider health check failed",
                }
        except ProviderError as exc:
            result = {"valid": False, "account": None, "error": str(exc)}
        finally:
            await runtime.aclose()
        provider.health_status = "healthy" if result["valid"] else "unhealthy"
        provider.last_validated_at = datetime.now(UTC)
        provider.last_error = result["error"]
        await self.session.flush()
        return ServiceResult.ok({"provider_id": provider.id, **result})

    # ------------------------------------------------------------------
    # Model catalogue
    # ------------------------------------------------------------------
    async def list_models(
        self,
        *,
        provider_type: str | None = None,
        supports_vision: bool | None = None,
        supports_tools: bool | None = None,
        supports_embeddings: bool | None = None,
        search: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        models, total = await self.models.list_available(
            provider_type=provider_type,
            supports_vision=supports_vision,
            supports_tools=supports_tools,
            supports_embeddings=supports_embeddings,
            search=search,
            skip=skip,
            limit=limit,
        )
        providers = {p.id: p for p in await self.providers.get_active()}
        return {
            "items": [
                self.serialize_model(model, providers.get(model.provider_id))
                for model in models
            ],
            "total": total,
            "page": skip // limit + 1 if limit else 1,
            "page_size": limit,
        }

    def serialize_model(
        self, model: AIModel, provider: AIProvider | None = None
    ) -> dict[str, Any]:
        return {
            "id": model.id,
            "provider_id": model.provider_id,
            "provider_name": provider.name if provider else None,
            "provider_type": provider.provider_type if provider else None,
            "name": model.name,
            "model_id": model.model_id,
            "version": model.version,
            "description": model.description,
            "task": model.task,
            "max_tokens": model.max_tokens,
            "context_length": model.context_length,
            "supports_streaming": model.supports_streaming,
            "supports_tools": model.supports_tools,
            "supports_vision": model.supports_vision,
            "supports_embeddings": model.supports_embeddings,
            "supports_audio": model.supports_audio,
            "supports_json_mode": model.supports_json_mode,
            "input_price_per_million": model.input_price_per_million,
            "output_price_per_million": model.output_price_per_million,
            "is_active": model.is_active,
            "is_default": model.is_default,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    async def discover_huggingface(
        self,
        provider_id: str | None,
        *,
        query: str | None = None,
        pipeline_tag: str | None = None,
        supports_vision: bool | None = None,
        supports_tools: bool | None = None,
        supports_embeddings: bool | None = None,
        provider: str | None = None,
        limit: int = 50,
        refresh: bool = False,
    ) -> ServiceResult:
        db_provider = None
        if provider_id:
            db_provider = await self.providers.get(provider_id)
        if db_provider is None:
            db_provider = await self.providers.get_by_name("huggingface")
        if db_provider is None:
            db_provider = await self._ensure_env_provider("huggingface")
        if db_provider is None:
            return ServiceResult.error_response(
                "Hugging Face is not configured; set HF_TOKEN first", 400
            )
        runtime = self.runtime_for(db_provider)
        try:
            if not isinstance(runtime, HuggingFaceProvider):
                return ServiceResult.error_response(
                    "Provider is not a Hugging Face provider", 400
                )
            models = await runtime.search_models(
                query,
                pipeline_tag=pipeline_tag,
                supports_vision=supports_vision,
                supports_tools=supports_tools,
                supports_embeddings=supports_embeddings,
                provider=provider,
                limit=limit,
                refresh=refresh,
            )
        except ProviderError as exc:
            return ServiceResult.error_response(str(exc), 502)
        finally:
            await runtime.aclose()
        return ServiceResult.ok({"items": models, "total": len(models)})

    @staticmethod
    def provider_types() -> list[dict[str, Any]]:
        return [
            {
                "type": provider_type,
                "display_name": provider_class.display_name or provider_type,
                "requires_api_key": provider_class.requires_api_key,
                "configured": has_credentials(provider_type),
                "capabilities": asdict(provider_class.capabilities),
            }
            for provider_type, provider_class in PROVIDER_CLASSES.items()
        ]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["ProviderService"]
