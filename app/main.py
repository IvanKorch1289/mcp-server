from datetime import datetime
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.server import app_graph, run_mcp_server
from app.session import session_manager, update_session_history
from app.models import PromptRequest
from app.storage.tarantool import tarantool_service
import logging
import os
import asyncio


logger = logging.getLogger(__name__)

# =======================
# Lifespan: управление жизненным циклом
# =======================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Асинхронный lifespan для подключения и отключения ресурсов."""
    # Startup
    print("Starting up...")
    await tarantool_service.connect()
    print("✅ Tarantool connected")

    yield

    # Shutdown
    print("Shutting down...")
    await tarantool_service.close()
    print("✅ Tarantool disconnected")


# Инициализация FastAPI с lifespan
app = FastAPI(title="GigaChat MCP Server", lifespan=lifespan)


@app.post("/prompt")
async def process_prompt(request: PromptRequest, bg: BackgroundTasks):
    """
    Основной эндпоинт для обработки пользовательского промпта.
    1. Очищает старые сессии (в фоне)
    2. Получает или создаёт сессию
    3. Добавляет запрос в историю
    4. Запускает LangGraph-агент
    5. Сохраняет ответ
    6. Возвращает результат
    """
    # Фоновая задача: очистка устаревших сессий
    bg.add_task(session_manager.cleanup_sessions)

    # Получаем сессию (создаётся, если не передана или не найдена)
    session_id, session = await session_manager.get_session(request.session_id)

    # Добавляем пользовательский ввод в историю
    update_session_history(session, "user", request.prompt)

    # Начальное состояние для графа
    initial_state = {
        "input": request.prompt,
        "session_id": session_id,
        "session": session,
        "response": None,
        "tool_results": [],
    }

    try:
        # Выполняем граф агента
        final_state = await app_graph.ainvoke(initial_state)

        # Извлекаем финальный ответ
        response_text = final_state.get("response", "Не удалось сгенерировать ответ.")

        # Сохраняем ответ ассистента в историю
        update_session_history(session, "assistant", response_text)

        # Логируем для отладки
        logger.info(f"Session {session_id}: completed prompt processing")

        # Возвращаем ответ клиенту
        return {
            "response": response_text,
            "session_id": session_id,
            "tools_used": len(final_state.get("tool_results", [])) > 0,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in process_prompt: {e}", exc_info=True)
        error_msg = "Произошла ошибка при обработке запроса."

        # Добавляем сообщение об ошибке в историю сессии
        update_session_history(session, "assistant", error_msg)

        raise HTTPException(
            status_code=500,
            detail=error_msg
        )


@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str):
    """
    Получить историю сообщений сессии.
    Полезно для отладки: проверить, попали ли результаты инструментов в историю.
    """
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = session_manager.sessions[session_id]
    session.last_accessed = datetime.now()  # обновляем время доступа

    return {
        "session_id": session_id,
        "created_at": session.created_at,
        "last_accessed": session.last_accessed,
        "history": session.history,
        "total_messages": len(session.history)
    }


# Основная функция запуска
async def main():
    """Main function to run both MCP server and FastAPI"""
    # Создаем папку для заметок если её нет
    os.makedirs("notes", exist_ok=True)

    # Запускаем MCP сервер в фоновой задаче
    mcp_task = asyncio.create_task(run_mcp_server())

    # Запускаем FastAPI сервер
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
    server = uvicorn.Server(config)

    # Запускаем оба сервера
    await asyncio.gather(
        server.serve(),
        mcp_task
    )


if __name__ == "__main__":
    asyncio.run(main())
