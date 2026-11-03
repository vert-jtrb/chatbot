from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user, get_superuser
from app.database.session import get_session
from app.models.user import User
from app.repositories.provider import AIModelRepository, AIProviderRepository
from app.schemas.provider import (
    AIModelCreate,
    AIModelResponse,
    AIProviderCreate,
    AIProviderResponse,
    AIProviderUpdate,
    HuggingFaceDiscoveryResponse,
    ModelListResponse,
    ProviderHealthResponse,
    ProviderSyncResponse,
    ProviderTypeResponse,
    ProviderValidateResponse,
)
from app.services.provider import ProviderService

router = APIRouter(prefix="/providers", tags=["Providers"])
models_router = APIRouter(prefix="/models", tags=["Models"])


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------
@router.get("/types", response_model=list[ProviderTypeResponse])
async def list_provider_types(
    _: User = Depends(get_current_user),
) -> list[dict]:
    return ProviderService.provider_types()


@router.get("/huggingface/discover", response_model=HuggingFaceDiscoveryResponse)
async def discover_huggingface_models(
    provider_id: str | None = Query(None),
    query: str | None = Query(None),
    pipeline_tag: str | None = Query(None),
    supports_vision: bool | None = Query(None),
    supports_tools: bool | None = Query(None),
    supports_embeddings: bool | None = Query(None),
    inference_provider: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    refresh: bool = Query(False),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).discover_huggingface(
        provider_id,
        query=query,
        pipeline_tag=pipeline_tag,
        supports_vision=supports_vision,
        supports_tools=supports_tools,
        supports_embeddings=supports_embeddings,
        provider=inference_provider,
        limit=limit,
        refresh=refresh,
    )
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return {"items": result.data["items"], "total": result.data["total"]}


# ----------------------------------------------------------------------
# Provider CRUD
# ----------------------------------------------------------------------
@router.get("", response_model=list[AIProviderResponse])
async def list_providers(
    active_only: bool = Query(False),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> list[dict]:
    return await ProviderService(session).list_providers(active_only=active_only)


@router.post("", response_model=AIProviderResponse, status_code=201)
async def create_provider(
    payload: AIProviderCreate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).create_provider(payload)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.get("/{provider_id}", response_model=AIProviderResponse)
async def get_provider(
    provider_id: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).get_provider(provider_id)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.patch("/{provider_id}", response_model=AIProviderResponse)
async def update_provider(
    provider_id: str,
    payload: AIProviderUpdate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).update_provider(provider_id, payload)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).delete_provider(provider_id)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.post("/{provider_id}/sync", response_model=ProviderSyncResponse)
async def sync_provider_models(
    provider_id: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).sync_models(provider_id)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.post("/{provider_id}/health", response_model=ProviderHealthResponse)
async def provider_health(
    provider_id: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).health_check(provider_id)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.post("/{provider_id}/validate", response_model=ProviderValidateResponse)
async def validate_provider(
    provider_id: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await ProviderService(session).validate_credentials(provider_id)
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.post(
    "/{provider_id}/models", response_model=AIModelResponse, status_code=201
)
async def create_provider_model(
    provider_id: str,
    payload: AIModelCreate,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    provider = await AIProviderRepository(session).get(provider_id)
    if provider is None or provider.is_deleted:
        raise HTTPException(status_code=404, detail="Provider not found")
    models = AIModelRepository(session)
    if await models.get_by_model_id(payload.model_id, provider_id):
        raise HTTPException(status_code=409, detail="Model already exists for provider")
    if payload.is_default:
        for existing in await models.get_by_provider(provider_id):
            existing.is_default = False
    model = await models.create(
        provider_id=provider_id,
        name=payload.name,
        model_id=payload.model_id,
        version=payload.version,
        description=payload.description,
        task=payload.task,
        max_tokens=payload.max_tokens,
        context_length=payload.context_length,
        supports_streaming=payload.supports_streaming,
        supports_tools=payload.supports_tools,
        supports_vision=payload.supports_vision,
        supports_embeddings=payload.supports_embeddings,
        supports_audio=payload.supports_audio,
        supports_json_mode=payload.supports_json_mode,
        input_price_per_million=payload.input_price_per_million,
        output_price_per_million=payload.output_price_per_million,
        is_default=payload.is_default,
    )
    return ProviderService(session).serialize_model(model, provider)


# ----------------------------------------------------------------------
# Model catalogue (authenticated users)
# ----------------------------------------------------------------------
@models_router.get("", response_model=ModelListResponse)
async def list_models(
    provider_type: str | None = Query(None),
    supports_vision: bool | None = Query(None),
    supports_tools: bool | None = Query(None),
    supports_embeddings: bool | None = Query(None),
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> dict:
    return await ProviderService(session).list_models(
        provider_type=provider_type,
        supports_vision=supports_vision,
        supports_tools=supports_tools,
        supports_embeddings=supports_embeddings,
        search=search,
        skip=(page - 1) * page_size,
        limit=page_size,
    )


@models_router.get("/{model_row_id}", response_model=AIModelResponse)
async def get_model(
    model_row_id: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_current_user),
) -> dict:
    models = AIModelRepository(session)
    model = await models.get_with_provider(model_row_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return ProviderService(session).serialize_model(model, model.provider)


@models_router.delete("/{model_row_id}")
async def delete_model(
    model_row_id: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    models = AIModelRepository(session)
    model = await models.get(model_row_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    model.is_active = False
    await session.flush()
    return {"message": "Model disabled", "id": model_row_id}
