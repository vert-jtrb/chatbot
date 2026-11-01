from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import RefreshToken, User, UserAPIKey
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, User)

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(
            func.lower(User.email) == email.lower(), User.is_deleted.is_(False)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_auth_provider(
        self, provider: str, provider_id: str
    ) -> User | None:
        stmt = select(User).where(
            User.auth_provider == provider,
            User.auth_provider_id == provider_id,
            User.is_deleted.is_(False),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count(self) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(User).where(User.is_deleted.is_(False))
        )
        return int(result.scalar() or 0)

    async def mark_email_verified(self, user_id: str) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(email_verified_at=datetime.now(UTC))
        )
        await self.session.execute(stmt)
        await self.session.flush()

    async def touch_login(self, user_id: str) -> None:
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(last_login_at=datetime.now(UTC))
        )
        await self.session.flush()


class RefreshTokenRepository(BaseRepository[RefreshToken]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, RefreshToken)

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        stmt = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_deleted.is_(False),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def revoke_family(self, family_id: str, replaced_by: str | None = None) -> None:
        await self.session.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == family_id)
            .values(
                is_revoked=True,
                revoked_at=datetime.now(UTC),
                replaced_by_hash=replaced_by,
            )
        )
        await self.session.flush()

    async def revoke(self, token: RefreshToken, replaced_by: str | None = None) -> None:
        token.is_revoked = True
        token.revoked_at = datetime.now(UTC)
        token.replaced_by_hash = replaced_by
        await self.session.flush()

    async def revoke_all_for_user(self, user_id: str) -> None:
        await self.session.execute(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked.is_(False),
            )
            .values(is_revoked=True, revoked_at=datetime.now(UTC))
        )
        await self.session.flush()


class UserAPIKeyRepository(BaseRepository[UserAPIKey]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, UserAPIKey)

    async def get_active_by_user(self, user_id: str) -> list[UserAPIKey]:
        stmt = select(UserAPIKey).where(
            UserAPIKey.user_id == user_id,
            UserAPIKey.is_active.is_(True),
            UserAPIKey.is_deleted.is_(False),
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


__all__ = ["RefreshTokenRepository", "UserAPIKeyRepository", "UserRepository"]
