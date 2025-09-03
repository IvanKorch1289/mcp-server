from fastapi import APIRouter

from app.storage.tarantool import TarantoolClient

utility_router = APIRouter(
    prefix="/utility",
    tags=["Утилиты"],
    responses={404: {"description": "Не найдено"}},
)


@utility_router.get("/validate_cache")
async def validate_cache():
    """Инвалидировать кеш."""
    client = await TarantoolClient.get_instance()
    return await client.invalidate_all_keys()
