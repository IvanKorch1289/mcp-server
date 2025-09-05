import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage, message_to_dict
from langchain_core.messages.human import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from app.advanced.logging_client import logger
from app.agent.models import PromptRequest
from app.agent.server import app_graph
from app.storage.tarantool import TarantoolClient, save_thread_to_tarantool

agent_router = APIRouter(prefix="/agent", tags=["Агент"])


@agent_router.post("/prompt")
async def process_prompt(request: PromptRequest, bg: BackgroundTasks):
    thread_id = request.thread_id or f"thread_{uuid.uuid4().hex}"
    created_at = datetime.now().timestamp()

    config = RunnableConfig(configurable={"thread_id": thread_id})

    initial_state = {
        "input": request.prompt,
        "thread_id": thread_id,
        "messages": [HumanMessage(content=request.prompt)],
        "response": None,
        "tool_results": [],
        "last_tool_call_handled": True,
        "tool_call_count": 0,
    }

    try:
        # Оборачиваем в try-except для перехвата ЛЮБОЙ ошибки
        final_state = await app_graph.ainvoke(initial_state, config=config)

        # Извлекаем ответ
        response_text = "Не удалось сгенерировать ответ."
        messages = final_state.get("messages", [])

        for msg in reversed(messages):
            if (
                isinstance(msg, AIMessage)
                and msg.content
                and not str(msg.content).startswith("❌ Ошибка выполнения")
            ):
                response_text = msg.content
                break

        # Сохраняем в фоне
        bg.add_task(
            save_thread_to_tarantool,
            thread_id,
            {
                "messages": [
                    message_to_dict(msg) if isinstance(msg, BaseMessage) else msg
                    for msg in messages
                ],
                "created_at": created_at,
                "input": request.prompt,
            },
        )

        return {
            "response": response_text,
            "thread_id": thread_id,
            "tools_used": len(final_state.get("tool_results", [])) > 0,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Critical error in process_prompt: {e}", exc_info=True)
        # Возвращаем чёткую ошибку
        raise HTTPException(
            status_code=500, detail=f"Ошибка выполнения агента: {str(e)}"
        ) from e


@agent_router.get("/thread_history/{thread_id}")
async def get_thread_history(thread_id: str):
    # Пробуем разные форматы ключей
    possible_keys = [
        f"thread:{thread_id}",
        f"thread:thread_{thread_id}",
        f"thread:thread_{thread_id}".encode("utf-8").decode("utf-8"),
    ]

    client = await TarantoolClient.get_instance()
    result = None

    # Пробуем все возможные форматы ключей
    for key in possible_keys:
        result = await client.get_persistent(key)
        if result:
            break

    if not result:
        raise HTTPException(status_code=404, detail="Тред не найден")

    # Возвращаем найденные данные
    return result


@agent_router.get("/threads")
async def list_threads() -> Dict[str, Any]:
    """
    Возвращает список всех сохранённых тредов из постоянного хранилища.
    """
    try:
        client = await TarantoolClient.get_instance()
        threads_data = await client.scan_threads()

        threads = []
        for item in threads_data:
            thread_id = item["key"].replace("thread:", "")
            threads.append(
                {
                    "thread_id": thread_id,
                    "user_prompt": (
                        item["input"][:100] + "..."
                        if len(item["input"]) > 100
                        else item["input"]
                    ),
                    "created_at": (
                        datetime.fromtimestamp(item["created_at"]).isoformat()
                        if item["created_at"]
                        else "Неизвестно"
                    ),
                    "message_count": item["message_count"],
                }
            )

        return {
            "total": len(threads),
            "threads": sorted(
                threads, key=lambda x: x.get("created_at", ""), reverse=True
            ),
        }
    except Exception as e:
        logger.error(f"Error listing threads: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Ошибка при получении списка тредов"
        ) from e
