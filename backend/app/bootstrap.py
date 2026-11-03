"""Startup bootstrap: schema, roles, providers, and optional demo data.

The bootstrap is idempotent and safe to run on every process start.  In
production, set ``AUTO_CREATE_TABLES=false`` and run Alembic migrations; the
bootstrap then only seeds roles and environment-backed providers.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security.password import hash_password
from app.database.base import Base
from app.database.session import async_session_factory, engine
import app.models  # noqa: F401 - imported for metadata registration
from app.models.chat import Conversation
from app.models.role import Role as RoleModel
from app.repositories.user import UserRepository
from app.services.auth import AuthService
from app.services.provider import ProviderService

logger = structlog.get_logger()


async def create_schema() -> None:
    if not settings.auto_create_tables:
        return
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("database_schema_ready", dialect="sqlite" if settings.is_sqlite else "postgres")


async def bootstrap(session: AsyncSession) -> None:
    auth = AuthService(session)
    await auth.ensure_default_roles()
    provider_stats = await ProviderService(session).bootstrap_from_environment()
    admin_created = await ensure_admin(session)
    demo = await ensure_demo_data(session) if settings.bootstrap_demo_data else None
    await session.commit()
    logger.info(
        "bootstrap_complete",
        providers=provider_stats,
        admin_created=admin_created,
        demo_user="created" if demo else "skipped",
    )


async def ensure_admin(session: AsyncSession) -> bool:
    """Create or promote the configured administrator account."""

    if not settings.admin_email or not settings.admin_password:
        return False
    if len(settings.admin_password) < 8:
        raise RuntimeError("ADMIN_PASSWORD must contain at least 8 characters")
    users = UserRepository(session)
    user = await users.get_by_email(settings.admin_email)
    admin_role = (
        await session.execute(select(RoleModel).where(RoleModel.name == "admin"))
    ).scalar_one_or_none()
    if user is None:
        user = await users.create(
            email=settings.admin_email.lower(),
            password_hash=hash_password(settings.admin_password),
            display_name="Administrator",
            role_id=admin_role.id if admin_role else None,
            is_superuser=True,
            email_verified_at=datetime.now(UTC),
        )
        logger.info("admin_bootstrapped", email=user.email)
        return True
    changed = False
    if not user.is_superuser:
        user.is_superuser = True
        changed = True
    if admin_role and user.role_id != admin_role.id:
        user.role_id = admin_role.id
        changed = True
    if changed:
        await session.flush()
        logger.info("admin_promoted", email=user.email)
    return False


async def ensure_demo_data(session: AsyncSession) -> bool:
    """Seed a demo user and welcome conversation in non-production environments.

    The password is generated randomly and logged once, so no credential is
    hardcoded in the repository.
    """

    if settings.is_production:
        logger.warning("demo_data_skipped_in_production")
        return False
    users = UserRepository(session)
    if await users.get_by_email("demo@example.com"):
        return False
    role = (
        await session.execute(select(RoleModel).where(RoleModel.name == "user"))
    ).scalar_one_or_none()
    password = secrets.token_urlsafe(12)
    user = await users.create(
        email="demo@example.com",
        password_hash=hash_password(password),
        display_name="Demo User",
        role_id=role.id if role else None,
        email_verified_at=datetime.now(UTC),
    )
    session.add(
        Conversation(
            user_id=user.id,
            title="Welcome to the platform",
            status="active",
            system_prompt=(
                "You are a helpful assistant. Keep answers clear and concise."
            ),
            message_count=1,
        )
    )
    await session.flush()
    logger.warning(
        "demo_user_created",
        email=user.email,
        generated_password=password,
        note="Change or delete this account before exposing the deployment",
    )
    return True


async def initialize() -> None:
    await create_schema()
    async with async_session_factory() as session:
        try:
            await bootstrap(session)
        except Exception:
            await session.rollback()
            raise


__all__ = ["bootstrap", "create_schema", "ensure_admin", "ensure_demo_data", "initialize"]
