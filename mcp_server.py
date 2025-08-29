import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import aiofiles
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langchain_gigachat.chat_models.gigachat import GigaChat
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, Field


load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализируем FastAPI приложение
app = FastAPI(title="GigaChat MCP Server")

# Инициализируем MCP сервер
mcp_server = Server("gigachat-mcp-server")

# Конфигурация
MAX_HISTORY_LENGTH = 20
SESSION_TIMEOUT_MINUTES = 30
GIGA_API_KEY = os.getenv('GIGA_API_KEY')

# Проверяем наличие API ключа
if not GIGA_API_KEY:
    logger.warning("GIGA_API_KEY not found in environment variables")


# Модели Pydantic для валидации
class CountFilesInput(BaseModel):
    directory_path: str = Field(..., description="Path to the directory")


class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file")


class CreateNoteInput(BaseModel):
    note_content: str = Field(..., description="Content of the note")
    note_name: Optional[str] = Field(None, description="Name of the note file")


class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class SessionState(BaseModel):
    history: List[Dict[str, str]] = []
    context: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    last_accessed: datetime = datetime.now()

    def to_messages(self):
        """
        Convert history to LangChain messages,
        ensuring system message is first
        """
        messages = []

        # Сначала добавляем system message если оно есть
        system_messages = []
        other_messages = []

        for msg in self.history:
            if msg["role"] == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Добавляем system messages первыми
        for msg in system_messages:
            messages.append(SystemMessage(content=msg["content"]))

        # Затем добавляем остальные сообщения
        for msg in other_messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        return messages


# Глобальное хранилище сессий с автоматической очисткой
class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get_session(
        self,
        session_id: Optional[str] = None
    ) -> tuple[str, SessionState]:
        """Get or create a session"""
        async with self._lock:
            if not session_id or session_id not in self.sessions:
                session_id = f"session_{uuid.uuid4().hex}"
                self.sessions[session_id] = SessionState()
                logger.info(f"Created new session: {session_id}")

            session = self.sessions[session_id]
            session.last_accessed = datetime.now()
            return session_id, session

    async def cleanup_sessions(self):
        """Clean up expired sessions"""
        async with self._lock:
            now = datetime.now()
            expired_keys = []

            for session_id, session in self.sessions.items():
                timeout_delta = now - session.last_accessed
                if timeout_delta.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
                    expired_keys.append(session_id)

            for session_id in expired_keys:
                del self.sessions[session_id]
                logger.info(f"Removed expired session: {session_id}")


# Инициализация менеджера сессий
session_manager = SessionManager()


# Определение состояния для LangGraph
# Определение состояния для LangGraph
class AgentState(TypedDict):
    input: str
    session_id: str
    session: SessionState
    response: Optional[str] = None
    tool_results: List[Dict[str, Any]] = []


# Инструменты как функции LangChain
async def count_files_in_directory(directory_path: str) -> Dict[str, Any]:
    """Count files in a directory"""
    try:
        if not os.path.exists(directory_path):
            return {"error": f"Directory {directory_path} does not exist"}

        if not os.path.isdir(directory_path):
            return {"error": f"{directory_path} is not a directory"}

        items = os.listdir(directory_path)
        files = []
        directories = []

        for item in items:
            full_path = os.path.join(directory_path, item)
            if os.path.isfile(full_path):
                files.append(item)
            elif os.path.isdir(full_path):
                directories.append(item)

        total_size = sum(
            os.path.getsize(os.path.join(directory_path, f))
            for f in files
        )

        return {
            "directory": os.path.abspath(directory_path),
            "file_count": len(files),
            "directory_count": len(directories),
            "total_size_bytes": total_size,
            "files": files,
            "directories": directories
        }
    except PermissionError:
        return {"error": f"Permission denied accessing {directory_path}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


async def get_current_time() -> Dict[str, Any]:
    """Get current date and time"""
    now = datetime.now()
    return {
        "iso_format": now.isoformat(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": now.timestamp(),
        "timezone": str(now.astimezone().tzinfo)
    }


async def read_file_content(file_path: str) -> Dict[str, Any]:
    """Read content of a file"""
    try:
        if not os.path.exists(file_path):
            return {"error": f"File {file_path} does not exist"}

        if not os.path.isfile(file_path):
            return {"error": f"{file_path} is not a file"}

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        return {
            "file_path": os.path.abspath(file_path),
            "content": content,
            "size_bytes": len(content.encode('utf-8')),
            "line_count": content.count('\n') + 1,
            "encoding": "utf-8"
        }
    except UnicodeDecodeError:
        return {"error": "Cannot decode file content as UTF-8 text"}
    except PermissionError:
        return {"error": f"Permission denied reading {file_path}"}
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}


async def create_note(
    note_content: str,
    note_name: Optional[str] = None
) -> Dict[str, Any]:
    """Create a text note with the given content"""
    try:
        if not note_name:
            note_name = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        notes_dir = os.path.join(os.getcwd(), "notes")
        os.makedirs(notes_dir, exist_ok=True)

        filepath = os.path.join(notes_dir, note_name)

        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(note_content)

        return {
            "status": "success",
            "file_path": filepath,
            "filename": note_name,
            "size_bytes": len(note_content.encode('utf-8')),
            "message": f"Note created successfully at {filepath}"
        }
    except Exception as e:
        return {"error": f"Error creating note: {str(e)}"}


# Создаем инструменты LangChain
tools = [
    StructuredTool.from_function(
        func=count_files_in_directory,
        name="count_files",
        description="Count files and directories in a specified directory",
        args_schema=CountFilesInput
    ),
    StructuredTool.from_function(
        func=get_current_time,
        name="get_current_time",
        description="Get current date and time"
    ),
    StructuredTool.from_function(
        func=read_file_content,
        name="read_file",
        description="Read content of a file",
        args_schema=ReadFileInput
    ),
    StructuredTool.from_function(
        func=create_note,
        name="create_note",
        description="Create a text note with the given content",
        args_schema=CreateNoteInput
    )
]

# Инициализация LLM
try:
    llm = GigaChat(
        credentials=GIGA_API_KEY,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        temperature=0.1,
        timeout=300,
        max_retries=2,
        streaming=False
    )
    logger.info("GigaChat initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize GigaChat: {e}")
    raise

# Привязываем инструменты к модели
try:
    llm_with_tools = llm.bind_tools(tools)
    logger.info("Tools bound successfully")
except Exception as e:
    logger.error(f"Failed to bind tools: {e}")
    # Fallback: используем модель без инструментов
    llm_with_tools = llm


# Промпт для агента
system_prompt_template = """Ты - AI-ассистент с доступом к инструментам.
У тебя есть доступ к следующим инструментам:
- count_files: посчитать файлы в директории
- get_current_time: получить текущее время
- read_file: прочитать содержимое файла
- create_note: создать текстовую заметку

Когда тебе нужно использовать инструмент, явно вызови его.
После получения результатов инструментов, обработай их и предоставь ответ пользователю.

Всегда отвечай на русском языке."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Создаем агента
agent = prompt | llm_with_tools


# Определяем узлы для LangGraph
async def agent_node(state: AgentState) -> Dict[str, Any]:
    """Узел выполнения агента"""
    config = RunnableConfig()

    # Получаем историю сообщений
    messages = state["session"].to_messages()

    # Вызываем агента с правильными переменными
    try:
        response = await agent.ainvoke({
            "input": state["input"],
            "chat_history": messages,
            "agent_scratchpad": []
        }, config=config)

        # Логируем ответ для отладки
        logger.info(f"Model response: {response}")
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response attributes: {dir(response)}")

        # Обрабатываем разные форматы ответов
        response_text = ""

        # Проверяем наличие tool_calls (новый формат)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"Found tool_calls: {response.tool_calls}")
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('args', {})
                response_text = f"ИНСТРУМЕНТ:{tool_name} ПАРАМЕТРЫ:{json.dumps(tool_args)}"
                logger.info(f"Generated tool call response: {response_text}")
                break  # берем первый вызов инструмента

        # Проверяем наличие function_call (старый формат)
        elif (hasattr(response, 'additional_kwargs') and
              response.additional_kwargs.get('function_call')):
            logger.info(
                f"Found function_call: {response.additional_kwargs['function_call']}"
            )
            tool_call = response.additional_kwargs['function_call']
            tool_name = tool_call.get('name', '')
            tool_args_str = tool_call.get('arguments', '{}')
            try:
                tool_args = json.loads(tool_args_str)
                response_text = f"ИНСТРУМЕНТ:{tool_name} ПАРАМЕТРЫ:{json.dumps(tool_args)}"
            except json.JSONDecodeError:
                response_text = f"ИНСТРУМЕНТ:{tool_name} ПАРАМЕТРЫ:{{}}"

        # Если есть обычный текстовый контент
        elif hasattr(response, 'content') and response.content:
            response_text = response.content

        # Если все еще пусто, используем fallback
        if not response_text:
            response_text = "Мне нужно использовать инструмент для выполнения этого запроса."
            logger.warning("No response text generated, using fallback")

        logger.info(f"Final response text: {response_text}")
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        return {"response": f"Error: {str(e)}"}


async def tool_node(state: AgentState) -> Dict[str, Any]:
    """Узел выполнения инструментов"""
    tool_results = []

    # Парсим вызовы инструментов из ответа
    tool_calls = parse_tool_calls(state["response"])

    for tool_name, tool_params in tool_calls:
        logger.info(f"Executing tool: {tool_name} with params: {tool_params}")

        # Находим инструмент
        tool = next((t for t in tools if t.name == tool_name), None)
        if not tool:
            result = {"error": f"Unknown tool: {tool_name}"}
            tool_results.append({"tool": tool_name, "result": result})
            continue

        # Вызываем инструмент
        try:
            if tool_name == "get_current_time":
                result = await tool.func()
            else:
                result = await tool.func(**tool_params)

            tool_results.append({"tool": tool_name, "result": result})

            # Вместо добавления как system message, сохраняем результаты в контексте
            # или формируем специальное сообщение для пользователя
            tool_result_str = f"Результат выполнения {tool_name}: {json.dumps(result, ensure_ascii=False)}"

            # Добавляем как user message для следующего шага
            update_session_history(
                state["session"],
                "user",
                f"Результат выполнения инструмента {tool_name}: {tool_result_str}"
            )

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            tool_results.append({
                "tool": tool_name,
                "result": {"error": error_msg}
            })

            # Добавляем ошибку как user message
            update_session_history(
                state["session"],
                "user",
                f"Ошибка выполнения инструмента {tool_name}: {error_msg}"
            )

    return {"tool_results": tool_results}

# Создаем граф workflow
workflow = StateGraph(AgentState)

# Добавляем узлы
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Устанавливаем начальные точки
workflow.set_entry_point("agent")


# Добавляем условные переходы
def should_use_tools(state: AgentState) -> str:
    """Определяем, нужно ли использовать инструменты"""
    logger.info(f"State in should_use_tools: {state}")

    if not state.get("response"):
        logger.info("No response in state, going to end")
        return "end"

    logger.info(
        f"Checking if should use tools for response: {state['response']}"
    )
    tool_calls = parse_tool_calls(state["response"])

    if tool_calls:
        logger.info(f"Found tool calls, going to tools node: {tool_calls}")
        return "tools"
    else:
        logger.info("No tool calls found, going to end")
        return "end"


workflow.add_conditional_edges(
    "agent",
    should_use_tools,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

# Компилируем граф
app_graph = workflow.compile()


# Вспомогательные функции
def parse_tool_calls(response: str) -> List[tuple]:
    """Parse tool calls from model response"""
    tool_calls = []

    if not response:
        logger.info("Empty response, no tool calls")
        return tool_calls

    logger.info(f"Parsing response: {response}")

    try:
        # Более надежный парсинг с учетом разных вариантов
        tool_patterns = [
            r'ИНСТРУМЕНТ:([a-zA-Z_][a-zA-Z0-9_]*)\s*ПАРАМЕТРЫ:\s*(\{.*\})',
            r'ИНСТРУМЕНТ[:\s]+([a-zA-Z_][a-zA-Z0-9_]*)[\s\S]*?ПАРАМЕТРЫ[:\s]*(\{[\s\S]*\})',
            r'TOOL:([a-zA-Z_][a-zA-Z0-9_]*)\s*PARAMS:\s*(\{.*\})',
        ]

        for pattern in tool_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                for tool_name, params_str in matches:
                    try:
                        # Очищаем строку параметров
                        params_str = params_str.strip()
                        # Убираем возможные trailing commas
                        if params_str.endswith(','):
                            params_str = params_str[:-1]
                        params = json.loads(params_str)
                        tool_calls.append((tool_name, params))
                        logger.info(
                            f"Successfully parsed tool call: {tool_name} with params: {params}"
                        )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse JSON for {tool_name}: {params_str}, error: {e}"
                        )
                        # Используем пустые параметры
                        tool_calls.append((tool_name, {}))
                break

    except Exception as e:
        logger.error(f"Error parsing tool calls: {e}")

    logger.info(f"Final tool calls: {tool_calls}")
    return tool_calls


def update_session_history(session: SessionState, role: str, content: str):
    """Обновляем историю сессии"""
    session.history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    # Ограничиваем историю
    if len(session.history) > MAX_HISTORY_LENGTH:
        session.history = session.history[-MAX_HISTORY_LENGTH:]


# Регистрируем инструменты в MCP сервере
@mcp_server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """Return the list of available tools."""
    return [
        Tool(
            name=tool.name,
            description=tool.description,
            inputSchema=(
                tool.args_schema.model_json_schema()
                if tool.args_schema
                else {}
            )
        )
        for tool in tools
    ]


@mcp_server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: Dict[str, Any]
) -> List[TextContent]:
    """Handle tool calls."""
    try:
        # Находим инструмент
        tool = next((t for t in tools if t.name == name), None)
        if not tool:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}"
            }, ensure_ascii=False))]

        # Вызываем инструмент
        if name == "get_current_time":
            result = await tool.func()
        else:
            result = await tool.func(**arguments)

        return [
            TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False)
            )
        ]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Tool execution error: {str(e)}"
        }, ensure_ascii=False))]


# FastAPI endpoints
@app.post("/prompt")
async def process_prompt(
    request: PromptRequest,
    background_tasks: BackgroundTasks
):
    """Process user prompt with context management using LangGraph"""
    # Очищаем старые сессии в фоне
    background_tasks.add_task(session_manager.cleanup_sessions)

    # Получаем или создаем сессию - используем переданный session_id если есть
    session_id, session = await session_manager.get_session(request.session_id)

    # Обновляем историю
    update_session_history(session, "user", request.prompt)

    # Выполняем граф
    initial_state = AgentState(
        input=request.prompt,
        session_id=session_id,
        session=session,
        response=None,
        tool_results=[]
    )

    try:
        final_state = await app_graph.ainvoke(initial_state)

        # Обновляем историю с ответом
        if final_state["response"]:
            update_session_history(
                session,
                "assistant",
                final_state["response"]
            )

        return {
            "response": final_state["response"],
            "session_id": session_id,
            "tools_used": len(final_state.get("tool_results", [])) > 0
        }
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        # Добавляем сообщение об ошибке в историю
        error_msg = f"Ошибка обработки запроса: {str(e)}"
        update_session_history(session, "assistant", error_msg)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing prompt: {str(e)}"
        )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session state"""
    if session_id in session_manager.sessions:
        session = session_manager.sessions[session_id]
        session.last_accessed = datetime.now()
        return {
            "session_id": session_id,
            "history": session.history,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
        return {"message": f"Session {session_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "message": "GigaChat MCP Server is running",
        "endpoints": {
            "POST /prompt": "Send a prompt to the AI model",
            "GET /sessions/{session_id}": "Get session state",
            "DELETE /sessions/{session_id}": "Delete session"
        }
    }


# Запуск MCP сервера
async def run_mcp_server():
    """Run the MCP server over stdio"""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream=read_stream,
            write_stream=write_stream,
            initialization_options=None,
        )


# Основная функция запуска
async def main():
    """Main function to run both MCP server and FastAPI"""
    # Создаем папку для заметок если её нет
    os.makedirs("notes", exist_ok=True)

    # Запускаем MCP сервер в фоновой задаче
    mcp_task = asyncio.create_task(run_mcp_server())

    # Запускаем FastAPI сервер
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", reload=True)
    server = uvicorn.Server(config)

    # Запускаем оба сервера
    await asyncio.gather(
        server.serve(),
        mcp_task
    )


if __name__ == "__main__":
    asyncio.run(main())
