from fastapi import APIRouter

from app.api.v1 import admin, auth, chats, knowledge, providers
from app.api.v1.providers import models_router

router = APIRouter(prefix="/api/v1")
router.include_router(auth.router)
router.include_router(chats.router)
router.include_router(providers.router)
router.include_router(models_router)
router.include_router(knowledge.router)
router.include_router(admin.router)
