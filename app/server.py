from typing import Dict, Any, List, TypedDict, Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_gigachat.chat_models.gigachat import GigaChat
from langgraph.graph import StateGraph, END
from mcp.server import Server
from mcp.types import Tool, TextContent
import json
import re
from langchain.tools import StructuredTool

from app.settings import settings
from app.prompts import system_prompt_template
from app.tools import (
    count_files_in_directory,
    get_current_time,
    read_file_content,
    create_note,
    fetch_company_info
)
from app.models import (
    CountFilesInput,
    ReadFileInput,
    CreateNoteInput,
    FetchCompanyInfoInput,
)
from mcp.server.stdio import stdio_server

# Инициализация MCP-сервера
mcp_server = Server("gigachat-mcp-server")

# =======================
# 1. Определение инструментов
# =======================

tools = [
    # Файловые инструменты
    {
        "func": count_files_in_directory,
        "schema": CountFilesInput,
        "name": "count_files",
        "desc": "Count files and directories in a specified directory",
    },
    {
        "func": get_current_time,
        "schema": None,
        "name": "get_current_time",
        "desc": "Get current date and time",
    },
    {
        "func": read_file_content,
        "schema": ReadFileInput,
        "name": "read_file",
        "desc": "Read content of a file",
    },
    {
        "func": create_note,
        "schema": CreateNoteInput,
        "name": "create_note",
        "desc": "Create a text note with the given content",
    },
    # Инструмент анализа клиентов
    {
        "func": fetch_company_info,
        "schema": FetchCompanyInfoInput,
        "name": "fetch_company_info",
        "desc": "Get comprehensive company info from DaData and InfoSphere by INN",
    },
]


# Конвертируем в LangChain StructuredTool
structured_tools = []
for tool in tools:
    if tool["schema"]:
        structured_tools.append(
            StructuredTool.from_function(
                func=tool["func"],
                name=tool["name"],
                description=tool["desc"],
                args_schema=tool["schema"],
            )
        )
    else:
        structured_tools.append(
            StructuredTool.from_function(
                func=tool["func"],
                name=tool["name"],
                description=tool["desc"],
            )
        )

# =======================
# 2. Инициализация LLM
# =======================

llm = GigaChat(
    credentials=settings.giga_api_key,
    verify_ssl_certs=False
)

try:
    llm_with_tools = llm.bind_tools(structured_tools)
except Exception as e:
    print(f"Warning: could not bind tools, using raw LLM: {e}")
    llm_with_tools = llm

# =======================
# 3. Промпт и агент
# =======================

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = prompt | llm_with_tools


# =======================
# 4. Состояние графа
# =======================

class AgentState(TypedDict):
    input: str
    session_id: str
    session: Any  # SessionState from session.py
    response: Optional[str]
    tool_results: List[Dict[str, Any]]


# =======================
# 5. Вспомогательные функции
# =======================

def parse_tool_calls(response: str) -> List[tuple]:
    """
    Парсит строку вида "ИНСТРУМЕНТ:имя ПАРАМЕТРЫ:{...}"
    в список кортежей (имя_инструмента, параметры).
    Поддерживает JSON и разные форматы.
    """
    if not response:
        return []

    patterns = [
        r'ИНСТРУМЕНТ:([a-zA-Z_][a-zA-Z0-9_]*)\s*ПАРАМЕТРЫ:\s*(\{.*\})',
        r'TOOL:([a-zA-Z_][a-zA-Z0-9_]*)\s*PARAMS:\s*(\{.*\})',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            result = []
            for tool_name, params_str in matches:
                try:
                    params_str = params_str.strip()
                    if params_str.endswith(','):
                        params_str = params_str[:-1]
                    params = json.loads(params_str)
                    result.append((tool_name, params))
                except json.JSONDecodeError:
                    result.append((tool_name, {}))
            return result
    return []


def update_session_history(session, role: str, content: str):
    """Добавляет сообщение в историю сессии."""
    session.history.append({
        "role": role,
        "content": content,
        "timestamp": session.last_accessed.isoformat(),
    })
    # Ограничиваем длину истории
    if len(session.history) > 20:
        session.history = session.history[-20:]


# =======================
# 6. Узлы графа
# =======================

async def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Узел агента: вызывает LLM для генерации ответа.
    Может вернуть текст или вызов инструмента.
    """
    messages = state["session"].to_messages()
    try:
        response = await agent.ainvoke({
            "input": state["input"],
            "chat_history": messages,
            "agent_scratchpad": [],
        }, config=RunnableConfig())

        response_text = ""

        # Новый формат: tool_calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            call = response.tool_calls[0]
            tool_name = call['name']
            tool_args = call.get('args', {})
            response_text = f"ИНСТРУМЕНТ:{tool_name} ПАРАМЕТРЫ:{json.dumps(tool_args)}"

        # Старый формат: function_call
        elif (hasattr(response, 'additional_kwargs') and
              response.additional_kwargs.get('function_call')):
            fc = response.additional_kwargs['function_call']
            tool_name = fc.get('name', '')
            try:
                tool_args = json.loads(fc.get('arguments', '{}'))
            except json.JSONDecodeError:
                tool_args = {}
            response_text = f"ИНСТРУМЕНТ:{tool_name} ПАРАМЕТРЫ:{json.dumps(tool_args)}"

        # Простой текст
        elif hasattr(response, 'content') and response.content:
            response_text = response.content

        else:
            response_text = "Не удалось обработать запрос."

        return {"response": response_text}
    except Exception as e:
        return {"response": f"Ошибка агента: {str(e)}"}


async def tool_node(state: AgentState) -> Dict[str, Any]:
    tool_calls = parse_tool_calls(state["response"])
    results = []

    for tool_name, tool_args in tool_calls:
        tool = next((t for t in structured_tools if t.name == tool_name), None)
        if not tool:
            error = {"error": f"Unknown tool: {tool_name}"}
            results.append({"tool": tool_name, "result": error})
            update_session_history(
                state["session"],
                "user",
                f"❌ Неизвестный инструмент: {tool_name}"
            )
            continue

        try:
            if tool_name == "get_current_time":
                result = await tool.func()
            else:
                result = await tool.func(**tool_args)

            results.append({"tool": tool_name, "result": result})

            # ✅ ВАЖНО: добавляем как user, но с префиксом
            update_session_history(
                state["session"],
                "user",
                f"[СИСТЕМА] fetch_company_info({tool_args.get('inn')}) выполнен:\n"
                f"{json.dumps(result, ensure_ascii=False, indent=2)}"
            )

        except Exception as e:
            error = {"error": str(e)}
            results.append({"tool": tool_name, "result": error})
            update_session_history(
                state["session"],
                "user",
                f"[СИСТЕМА] Ошибка в {tool_name}: {str(e)}"
            )

    # 🔥 Сбрасываем response, чтобы agent_node вызвался снова
    return {
        "tool_results": results,
        "response": None
    }

# =======================
# 7. Построение графа
# =======================

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")


def should_use_tools(state: AgentState) -> str:
    """Определяет, нужно ли вызывать инструменты."""
    if not state.get("response"):
        # Если нет response — это начало или после tool_node
        # → завершаем (финальный ответ уже сгенерирован ранее)
        return "end"

    # Проверяем, есть ли вызов инструмента
    if parse_tool_calls(state["response"]):
        return "tools"

    # Если response есть, но нет вызова — это финальный ответ
    return "end"


workflow.add_conditional_edges(
    "agent",
    should_use_tools,
    {
        "tools": "tools",
        "agent": "agent",  # ← после tool_node снова в agent
        "end": END
    }
)

# После tool_node — всегда возвращаемся к agent
workflow.add_edge("tools", "agent")

app_graph = workflow.compile()


# =======================
# 8. Регистрация в MCP
# =======================

@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    """Возвращает список всех доступных инструментов."""
    return [
        Tool(
            name=tool["name"],
            description=tool["desc"],
            inputSchema=(
                tool["schema"].model_json_schema() if tool["schema"] else {}
            ),
        )
        for tool in tools
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Вызывает инструмент по имени."""
    tool = next((t for t in structured_tools if t.name == name), None)
    if not tool:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False))]

    try:
        if name == "get_current_time":
            result = await tool.func()
        else:
            result = await tool.func(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


# Запуск MCP сервера
async def run_mcp_server():
    """Run the MCP server over stdio"""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream=read_stream,
            write_stream=write_stream,
            initialization_options=None,
        )
