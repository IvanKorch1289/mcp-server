import asyncio
import logging
from contextlib import asynccontextmanager

import httpx

from app.advanced.logging_client import logger

# Отключаем стандартный логгер httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

# Глобальные объекты
_http_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


@asynccontextmanager
async def http_client_session():
    """Контекстный менеджер для временного клиента (опционально)"""
    client = await get_http_client()
    yield client


async def get_http_client() -> httpx.AsyncClient:
    """
    Единый HTTP-клиент с логированием всех запросов.
    Автоматически логирует запросы и ответы.
    """
    global _http_client

    if _http_client is None:
        async with _client_lock:
            if _http_client is None:
                _http_client = httpx.AsyncClient(
                    http2=True,
                    limits=httpx.Limits(
                        max_connections=30,
                        max_keepalive_connections=15,
                    ),
                    timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=2.0),
                    # Используем кастомные хуки
                    event_hooks={
                        "request": [_on_request],
                        "response": [_on_response],
                    },
                    transport=httpx.AsyncHTTPTransport(retries=2),
                )
    return _http_client


async def _on_request(request: httpx.Request):
    """Обёртка: логируем запрос"""
    # Добавляем метку времени
    request.extensions["start_time"] = asyncio.get_event_loop().time()
    logger.log_request(request)


async def _on_response(response: httpx.Response):
    """Обёртка: логируем ответ с измерением времени"""
    start_time = response.request.extensions.get("start_time", None)
    duration = asyncio.get_event_loop().time() - start_time if start_time else 0.0
    logger.log_response(response, duration=duration)


async def close_http_client():
    """Закрывает HTTP-клиент и освобождает ресурсы"""
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None
