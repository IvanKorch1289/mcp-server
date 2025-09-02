from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.advanced.logging_client import logger
from app.agent.models import PromptRequest
from app.agent.server import app_graph
from app.agent.session import session_manager, update_session_history

agent_router = APIRouter(
    prefix="/agent", tags=["Агент"], responses={404: {"description": "Не найдено"}}
)


@agent_router.post("/prompt")
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

        raise HTTPException(status_code=500, detail=error_msg) from e


@agent_router.get("/sessions/{session_id}")
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
        "total_messages": len(session.history),
    }
