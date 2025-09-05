import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_gigachat.chat_models import GigaChat
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from app.advanced.logging_client import logger
from app.agent.models import (
    CountFilesInput,
    CreateNoteInput,
    FetchCompanyInfoInput,
    ReadFileInput,
)
from app.agent.prompts import system_prompt_template
from app.agent.tools import (
    count_files_in_directory,
    create_note,
    fetch_company_info_for_analyze,
    get_current_time,
    read_file_content,
)
from app.settings import settings

# =======================
# 1. Состояние: TypedDict + add_messages
# =======================


class AgentState(TypedDict):
    input: str
    thread_id: str
    messages: Annotated[List[dict], add_messages]
    response: Optional[str]
    tool_results: List[Dict[str, Any]]
    last_tool_call_handled: bool


# =======================
# 2. Инструменты → StructuredTool
# =======================

tools = [
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
    {
        "func": fetch_company_info_for_analyze,
        "schema": FetchCompanyInfoInput,
        "name": "fetch_company_info",
        "desc": "Get comprehensive company info from DaData and InfoSphere by INN",
    },
]

structured_tools = []

for t in tools:
    if t["schema"]:
        tool = StructuredTool(
            name=t["name"],
            description=t["desc"],
            args_schema=t["schema"],
            coroutine=t["func"],
        )
    else:
        tool = StructuredTool.from_function(
            name=t["name"],
            description=t["desc"],
            coroutine=t["func"],
        )
    structured_tools.append(tool)


# =======================
# 3. LLM и промпт
# =======================

llm = GigaChat(credentials=settings.giga_api_key, verify_ssl_certs=False)
llm_with_tools = llm.bind_tools(structured_tools)


prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt_template), ("placeholder", "{messages}")]
)
agent_runnable = prompt | llm_with_tools


# =======================
# 4. Узел агента
# =======================


async def agent_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    try:
        response = await agent_runnable.ainvoke(
            {"messages": state["messages"]},
            config=config,
        )
        # Добавляем обработку для случаев, когда ответ пустой
        if isinstance(response, AIMessage) and not response.content:
            # Если content пустой, но есть tool_calls, это нормально
            if not getattr(response, "tool_calls", None):
                # Для "чистых" ответов без содержимого используем placeholder
                response.content = "Генерация ответа завершена."
        return {"messages": [response]}
    except Exception as e:
        error_msg = f"Ошибка LLM: {str(e)[:500]}"
        logger.error(error_msg, exc_info=True)
        return {"messages": [AIMessage(content=error_msg)]}


# =======================
# 5. Узел инструментов
# =======================

tool_node = ToolNode(structured_tools)


# =======================
# 6. Условия перехода
# =======================


def should_use_tools(state: AgentState) -> str:
    # Проверяем лимит вызовов
    if state.get("tool_call_count", 0) >= 5:
        return "end"

    # Проверяем, есть ли новые вызовы инструментов
    messages = state["messages"]
    if not messages:
        return "end"

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "tools"

    return "end"


# =======================
# 7. Построение графа
# =======================

workflow = StateGraph(AgentState)
memory_saver = MemorySaver()

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_use_tools, {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app_graph = workflow.compile(checkpointer=memory_saver)


# =======================
# 8. MCP-сервер
# =======================

mcp_server = Server("gigachat-mcp-server")


@mcp_server.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name=t["name"],
            description=t["desc"],
            inputSchema=t["schema"].model_json_schema() if t["schema"] else {},
        )
        for t in tools
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    tool = next((t for t in structured_tools if t.name == name), None)
    if not tool:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False),
            )
        ]

    try:
        result = await tool.ainvoke(arguments)
        return [
            TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2, default=str),
            )
        ]
    except Exception as e:
        return [
            TextContent(
                type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False)
            )
        ]


# =======================
# 9. Запуск MCP
# =======================


async def run_mcp_server():
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, None)
