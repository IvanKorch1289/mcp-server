from fastapi import APIRouter

from app.http_tools.fetch_data import (
    fetch_company_info,
    fetch_from_dadata,
    fetch_from_infosphere,
)

data_router = APIRouter(
    prefix="/data",
    tags=["Внешние данные"],
    responses={404: {"description": "Не найдено"}},
)


@data_router.get("/client/infosphere/{inn}")
async def get_infosphere_data(inn: str):
    """Получить данные по клиенту из Инфосферы (для отладки)."""
    return await fetch_from_infosphere(inn)


@data_router.get("/client/dadata/{inn}")
async def get_dadata_data(inn: str):
    """Получить данные по клиенту из DaData (для отладки)."""
    return await fetch_from_dadata(inn)


@data_router.get("/client/info/{inn}")
async def get_all_client_data(inn: str):
    """Получить данные по клиенту (для отладки)."""
    return await fetch_company_info(inn)
