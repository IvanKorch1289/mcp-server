import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.agent.server import run_mcp_server
from app.routes.agent_routes import agent_router
from app.routes.data_routes import data_router
from app.routes.utility_routes import utility_router
from app.services.http_client import AsyncHttpClient
from app.storage.tarantool import TarantoolClient

logger = logging.getLogger(__name__)


# =======================
# Lifespan: управление жизненным циклом
# =======================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("✅ Приложение запущено")
    await AsyncHttpClient.get_instance()
    await TarantoolClient.get_instance()
    yield
    print("🛑 Остановка приложения...")
    await TarantoolClient.close_global()
    await AsyncHttpClient.close_global()
    logger.info("✅ Все соединения закрыты")


# Инициализация FastAPI с lifespan
app = FastAPI(title="GigaChat MCP Server", lifespan=lifespan)

app.include_router(agent_router)
app.include_router(data_router)
app.include_router(utility_router)


# Основная функция запуска
async def main():
    """Main function to run both MCP server and FastAPI"""
    # Создаем папку для заметок если её нет
    os.makedirs("notes", exist_ok=True)

    # Запускаем MCP сервер в фоновой задаче
    mcp_task = asyncio.create_task(run_mcp_server())

    # Запускаем FastAPI сервер
    config = uvicorn.Config(
        app, host="0.0.0.0", port=8000, log_level="info", reload=True
    )
    server = uvicorn.Server(config)

    # Запускаем оба сервера
    await asyncio.gather(server.serve(), mcp_task)


if __name__ == "__main__":
    asyncio.run(main())
