from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.middleware.cors import setup_cors
from app.api.middleware.rate_limit import setup_rate_limiting
from app.api.v1.router import router
from app.bootstrap import initialize
from app.core.config import settings
from app.database.session import async_session_factory, engine
from app.logging.setup import setup_logging
from app.providers.ai.registry import ProviderRegistry
from app.telemetry.setup import setup_telemetry

logger = structlog.get_logger()

HTTP_REQUESTS = Counter(
    "chatbot_http_requests_total",
    "HTTP requests processed",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "chatbot_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    settings.validate_production_secrets()
    await initialize()
    ProviderRegistry.initialize()
    logger.info(
        "application_started",
        environment=str(settings.environment),
        debug=settings.debug,
        providers=list(ProviderRegistry.get_all()),
    )
    try:
        yield
    finally:
        await engine.dispose()
        logger.info("application_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Open-source, multi-provider AI chatbot platform API",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    setup_cors(app)
    setup_rate_limiting(app)
    setup_telemetry(app)

    @app.middleware("http")
    async def observe_requests(request: Request, call_next) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            HTTP_REQUESTS.labels(request.method, request.url.path, "500").inc()
            raise
        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        HTTP_LATENCY.labels(request.method, path).observe(elapsed)
        response.headers["X-Response-Time"] = f"{elapsed * 1000:.1f}ms"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def custom_http_exception_handler(
        _request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "status_code": 500},
        )

    @app.get("/health", tags=["Operations"])
    async def health_check() -> JSONResponse:
        database_ok = True
        try:
            async with async_session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception:
            database_ok = False
        status_code = 200 if database_ok else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if database_ok else "degraded",
                "version": settings.app_version,
                "environment": str(settings.environment),
                "database": "ok" if database_ok else "unavailable",
            },
        )

    @app.get("/ready", tags=["Operations"])
    async def readiness() -> JSONResponse:
        providers = list(ProviderRegistry.get_all())
        return JSONResponse(
            status_code=200,
            content={
                "ready": True,
                "providers": providers,
                "version": settings.app_version,
            },
        )

    @app.get("/metrics", tags=["Operations"])
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)

    return app


app = create_app()
