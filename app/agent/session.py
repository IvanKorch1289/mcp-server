import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict

from langchain.schema import (
    AIMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from app.agent.prompts import system_prompt_template


class SessionState(BaseModel):
    history: list = []
    context: dict = {}
    created_at: datetime = datetime.now()
    last_accessed: datetime = datetime.now()

    # Храним system_prompt отдельно
    system_prompt: str = system_prompt_template

    def to_messages(self):
        messages = []
        messages.append(SystemMessage(content=self.system_prompt))

        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                if msg.get("tool_calls"):
                    tool_calls = []
                    for tc in msg["tool_calls"]:
                        try:
                            tool_calls.append(
                                {
                                    "name": tc["function"]["name"],
                                    "args": json.loads(tc["function"]["arguments"]),
                                    "id": tc["id"],
                                    "type": tc["type"],
                                }
                            )
                        except (KeyError, json.JSONDecodeError):
                            continue
                    messages.append(AIMessage(content="", tool_calls=tool_calls))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "function":
                messages.append(
                    FunctionMessage(name=msg["name"], content=msg["content"])
                )

        return messages


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get_session(self, session_id: str = None) -> tuple[str, SessionState]:
        async with self._lock:
            if not session_id or session_id not in self.sessions:
                session_id = f"session_{uuid.uuid4().hex}"
                self.sessions[session_id] = SessionState()
            session = self.sessions[session_id]
            session.last_accessed = datetime.now()
            return session_id, session

    async def cleanup_sessions(self):
        async with self._lock:
            now = datetime.now()
            expired = [
                sid
                for sid, s in self.sessions.items()
                if (now - s.last_accessed).total_seconds() > 30 * 60
            ]
            for sid in expired:
                del self.sessions[sid]


def update_session_history(
    session: SessionState, role: str, content: str, tool_calls=None
):
    if role not in ("user", "assistant"):
        raise ValueError("Only 'user' and 'assistant' roles allowed in history")

    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }

    if tool_calls:
        # Приводим к формату, который ожидает LangChain
        formatted_tool_calls = []
        for tc in tool_calls:
            formatted_tool_calls.append(
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",  # или "tool_call" — зависит от версии
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    },
                }
            )
        msg["tool_calls"] = formatted_tool_calls

    session.history.append(msg)

    if len(session.history) > 20:
        session.history = session.history[-20:]


def add_tool_result(session: SessionState, tool_name: str, result: dict):
    """
    Добавляет результат инструмента как function-сообщение.
    """
    session.history.append(
        {
            "role": "function",
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False, indent=2),
            "timestamp": datetime.now().isoformat(),
        }
    )

    if len(session.history) > 20:
        session.history = session.history[-20:]


session_manager = SessionManager()
