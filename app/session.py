import asyncio
import uuid
from datetime import datetime
from typing import Dict
from pydantic import BaseModel
from app.prompts import system_prompt_template


class SessionState(BaseModel):
    history: list = []
    context: dict = {}
    created_at: datetime = datetime.now()
    last_accessed: datetime = datetime.now()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Добавляем system-сообщение один раз
        if not self.history or not any(m["role"] == "system" for m in self.history):
            self.history.append({
                "role": "system",
                "content": system_prompt_template,
                "timestamp": datetime.now().isoformat()
            })

    def to_messages(self):
        from langchain.schema import HumanMessage, AIMessage, SystemMessage
        messages = []

        # Только ПЕРВОЕ system-сообщение — как system
        first_system = next((msg for msg in self.history if msg["role"] == "system"), None)
        if first_system:
            messages.append(SystemMessage(content=first_system["content"]))

        # Все остальные — как user/assistant
        for msg in self.history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            # Остальные system (например, результаты) — как user
            elif msg["role"] == "system":
                messages.append(HumanMessage(content=f"[СИСТЕМА] {msg['content']}"))

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
                sid for sid, s in self.sessions.items()
                if (now - s.last_accessed).total_seconds() > 30 * 60
            ]
            for sid in expired:
                del self.sessions[sid]


def update_session_history(session: SessionState, role: str, content: str):
    """
    Добавляет сообщение в историю сессии.
    """
    session.history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    # Ограничиваем длину истории
    if len(session.history) > 20:
        session.history = session.history[-20:]


session_manager = SessionManager()
