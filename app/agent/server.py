import json
from typing import Any, Dict, List, Optional, TypedDict

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_gigachat.chat_models.gigachat import GigaChat
from langgraph.graph import END, StateGraph
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from app.agent.models import (
    CountFilesInput,
    CreateNoteInput,
    FetchCompanyInfoInput,
    ReadFileInput,
)
from app.agent.session import add_tool_result, update_session_history
from app.agent.tools import (
    count_files_in_directory,
    create_note,
    get_current_time,
    read_file_content,
)
from app.http_tools.fetch_data import fetch_company_info
from app.settings import settings

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

llm = GigaChat(credentials=settings.giga_api_key, verify_ssl_certs=False)

try:
    llm_with_tools = llm.bind_tools(structured_tools)
except Exception as e:
    print(f"Warning: could not bind tools, using raw LLM: {e}")
    llm_with_tools = llm

# =======================
# 3. Промпт и агент
# =======================

prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
    ]
)

agent = prompt | llm_with_tools


# =======================
# 4. Состояние графа
# =======================


class AgentState(TypedDict):
    input: str
    session_id: str
    session: Any  # SessionState
    response: Optional[str]
    tool_results: List[Dict[str, Any]]
    # Добавим флаг, чтобы избежать зацикливания
    last_tool_call_handled: bool


# =======================
# 5. Узлы графа
# =======================


async def agent_node(state: AgentState) -> Dict[str, Any]:
    messages = state["session"].to_messages()

    try:
        response = await agent.ainvoke(
            {
                "chat_history": messages,
            },
            config=RunnableConfig(),
        )

        if hasattr(response, "tool_calls") and response.tool_calls:
            last_msg = (
                state["session"].history[-1] if state["session"].history else None
            )
            if (
                last_msg
                and last_msg["role"] == "function"
                and last_msg["name"] == response.tool_calls[0]["name"]
            ):
                pass  # уже был вызов
            else:
                # Форматируем tool_calls для сохранения
                formatted_calls = [
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": tc["args"],
                        "type": tc.get("type", "function"),
                    }
                    for tc in response.tool_calls
                ]
                # Сохраняем через update_session_history
                update_session_history(
                    state["session"],
                    "assistant",
                    "",  # content пустой при tool_calls
                    tool_calls=formatted_calls,
                )

            return {
                "response": None,
                "tool_results": [],
                "last_tool_call_handled": False,
            }

        if hasattr(response, "content") and response.content:
            update_session_history(state["session"], "assistant", response.content)
            return {
                "response": response.content,
                "last_tool_call_handled": True,
            }

        return {
            "response": "Не удалось сгенерировать ответ.",
            "last_tool_call_handled": True,
        }

    except Exception as e:
        error_msg = f"Ошибка агента: {str(e)}"
        update_session_history(state["session"], "assistant", error_msg)
        return {"response": error_msg, "last_tool_call_handled": True}


async def tool_node(state: AgentState) -> Dict[str, Any]:
    history = state["session"].history
    if not history:
        return {
            "tool_results": [],
            "response": None,
            "last_tool_call_handled": True,
        }

    last_msg = history[-1]
    if last_msg["role"] != "assistant" or not last_msg.get("tool_calls"):
        return {
            "tool_results": [],
            "response": None,
            "last_tool_call_handled": True,
        }

    tool_calls = last_msg["tool_calls"]
    results = []

    for call in tool_calls:
        # ✅ Исправлено: name внутри function
        try:
            tool_name = call["function"]["name"]
            # Аргументы — распарсим из JSON
            args = json.loads(call["function"]["arguments"])
        except (KeyError, json.JSONDecodeError) as e:
            error = {"error": f"Invalid tool call format: {str(e)}"}
            results.append({"tool": "unknown", "result": error})
            continue

        tool = next((t for t in structured_tools if t.name == tool_name), None)
        if not tool:
            error = {"error": f"Unknown tool: {tool_name}"}
            results.append({"tool": tool_name, "result": error})
            update_session_history(
                state["session"],
                "user",
                f"❌ Неизвестный инструмент: {tool_name}",
            )
            continue

        try:
            result = await tool.func(**args)
            results.append({"tool": tool_name, "result": result})
            add_tool_result(state["session"], tool_name, result)

        except Exception as e:
            error = {"error": str(e)}
            results.append({"tool": tool_name, "result": error})
            update_session_history(
                state["session"],
                "user",
                f"[СИСТЕМА] Ошибка в {tool_name}: {str(e)}",
            )

    return {
        "tool_results": results,
        "response": None,
        "last_tool_call_handled": True,
    }


# =======================
# 6. Построение графа
# =======================

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")


# Условие: нужно ли вызывать инструмент?
def should_use_tools(state: AgentState) -> str:
    # Если в последнем ответе были tool_calls, и они ещё не обработаны
    history = state["session"].history
    if not history:
        return "end"

    last_msg = history[-1]
    if last_msg["role"] == "assistant" and last_msg.get("tool_calls"):
        return "tools"

    return "end"


# Условные рёбра из agent
workflow.add_conditional_edges(
    "agent", should_use_tools, {"tools": "tools", "end": END}
)

# 🔁 ВАЖНО: после tools — снова к agent
workflow.add_edge("tools", "agent")

app_graph = workflow.compile()


# =======================
# 7. Регистрация в MCP
# =======================


@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    """Возвращает список всех доступных инструментов."""
    return [
        Tool(
            name=tool["name"],
            description=tool["desc"],
            inputSchema=(tool["schema"].model_json_schema() if tool["schema"] else {}),
        )
        for tool in tools
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Вызывает инструмент по имени."""
    tool = next((t for t in structured_tools if t.name == name), None)
    if not tool:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False),
            )
        ]

    try:
        if name == "get_current_time":
            result = await tool.func()
        else:
            result = await tool.func(**arguments)
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, ensure_ascii=False),
            )
        ]


# Запуск MCP сервера
async def run_mcp_server():
    """Run the MCP server over stdio"""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream=read_stream,
            write_stream=write_stream,
            initialization_options=None,
        )
