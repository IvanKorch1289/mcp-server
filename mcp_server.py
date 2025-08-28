import os
import json
import requests
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import uvicorn

# Инициализируем FastAPI приложение
app = FastAPI(title="Qwen MCP Server")

# Инициализируем MCP сервер
mcp_server = Server("qwen-mcp-server")

# Модель для запросов
class PromptRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None

class SessionState(BaseModel):
    history: List[Dict[str, str]] = []
    context: Dict[str, Any] = {}
    created_at: datetime = datetime.now()

# Глобальное хранилище сессий
sessions: Dict[str, SessionState] = {}

# Базовый URL для Ollama
OLLAMA_URL = "http://localhost:11434"

# Регистрируем инструменты в MCP сервере
@mcp_server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """Return the list of available tools."""
    return [
        Tool(
            name="count_files",
            description="Count files and directories in a specified directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Path to the directory"
                    }
                },
                "required": ["directory_path"]
            }
        ),
        Tool(
            name="get_current_time",
            description="Get current date and time",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="read_file",
            description="Read content of a file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="create_note",
            description="Create a text note with the given content",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_content": {
                        "type": "string",
                        "description": "Content of the note"
                    },
                    "note_name": {
                        "type": "string",
                        "description": "Name of the note file (optional)"
                    }
                },
                "required": ["note_content"]
            }
        )
    ]

# Реализация инструментов
@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "count_files":
            directory_path = arguments.get("directory_path", ".")
            result = count_files_in_directory(directory_path)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        elif name == "get_current_time":
            result = get_current_time()
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        elif name == "read_file":
            file_path = arguments.get("file_path")
            if not file_path:
                return [TextContent(type="text", text=json.dumps({
                    "error": "File path is required"
                }, ensure_ascii=False))]

            result = read_file_content(file_path)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        elif name == "create_note":
            note_content = arguments.get("note_content", "")
            note_name = arguments.get(
                "note_name",
                f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            result = create_note(note_content, note_name)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

        else:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}"
            }, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Tool execution error: {str(e)}"
        }, ensure_ascii=False))]

# Функции инструментов
def count_files_in_directory(directory_path: str) -> Dict[str, Any]:
    """Count files in a directory"""
    try:
        if not os.path.exists(directory_path):
            return {"error": f"Directory {directory_path} does not exist"}

        if not os.path.isdir(directory_path):
            return {"error": f"{directory_path} is not a directory"}

        # Получаем список всех элементов
        items = os.listdir(directory_path)

        # Разделяем на файлы и директории
        files = []
        directories = []

        for item in items:
            full_path = os.path.join(directory_path, item)
            if os.path.isfile(full_path):
                files.append(item)
            elif os.path.isdir(full_path):
                directories.append(item)

        # Получаем общий размер файлов
        total_size = sum(os.path.getsize(os.path.join(directory_path, f)) for f in files)

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

def get_current_time() -> Dict[str, Any]:
    """Get current date and time"""
    now = datetime.now()
    return {
        "iso_format": now.isoformat(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": now.timestamp(),
        "timezone": str(now.astimezone().tzinfo)
    }

def read_file_content(file_path: str) -> Dict[str, Any]:
    """Read content of a file"""
    try:
        if not os.path.exists(file_path):
            return {"error": f"File {file_path} does not exist"}

        if not os.path.isfile(file_path):
            return {"error": f"{file_path} is not a file"}

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "file_path": os.path.abspath(file_path),
            "content": content,
            "size_bytes": len(content),
            "line_count": content.count('\n') + 1,
            "encoding": "utf-8"
        }
    except UnicodeDecodeError:
        return {"error": "Cannot decode file content as UTF-8 text"}
    except PermissionError:
        return {"error": f"Permission denied reading {file_path}"}
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}

def create_note(content: str, filename: str = None) -> Dict[str, Any]:
    """Create a text note with the given content"""
    try:
        if not filename:
            filename = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        # Создаем папку notes если её нет
        notes_dir = os.path.join(os.getcwd(), "notes")
        os.makedirs(notes_dir, exist_ok=True)

        filepath = os.path.join(notes_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return {
            "status": "success",
            "file_path": filepath,
            "filename": filename,
            "size_bytes": len(content),
            "message": f"Note created successfully at {filepath}"
        }
    except Exception as e:
        return {"error": f"Error creating note: {str(e)}"}

# Функция для отправки промпта модели
def send_to_model(prompt: str, system_prompt: str = None) -> str:
    """Send prompt to Qwen model"""
    url = f"{OLLAMA_URL}/api/generate"

    payload = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 500
        }
    }

    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received from model")
    except requests.exceptions.Timeout:
        return "Model request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to model: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# FastAPI endpoints
@app.post("/prompt")
async def process_prompt(request: PromptRequest):
    """Process user prompt with context management"""
    # Получаем или создаем сессию
    session_id = request.session_id or f"session_{datetime.now().timestamp()}"
    if session_id not in sessions:
        sessions[session_id] = SessionState()

    session = sessions[session_id]

    # Системный промпт с контекстом
    system_prompt = f"""Ты - AI-ассистент с доступом к инструментам.
    У тебя есть доступ к следующим инструментам:
    - count_files: посчитать файлы в директории
    - get_current_time: получить текущее время
    - read_file: прочитать содержимое файла
    - create_note: создать текстовую заметку
    
    История диалога (последние 5 сообщений):
    {json.dumps(session.history[-5:], ensure_ascii=False, indent=2)}
    
    Контекст сессии:
    {json.dumps(session.context, ensure_ascii=False, indent=2)}
    
    Всегда отвечай на русском языке."""

    # Формируем промпт с учетом истории
    full_prompt = f"""
    Пользователь: {request.prompt}
    
    Проанализируй запрос и определи, нужно ли использовать инструменты.
    Если нужно использовать инструмент, ответь в формате:
    ИНСТРУМЕНТ:название_инструмента
    ПАРАМЕТРЫ:json_с_параметрами
    
    Если инструменты не нужны, просто ответь на вопрос.
    """

    # Отправляем модели
    response = send_to_model(full_prompt, system_prompt)

    # Парсим ответ на предмет вызовов инструментов
    tool_calls = parse_tool_calls(response)
    final_response = response

    if tool_calls:
        tool_results = []
        for tool_name, tool_params in tool_calls:
            # Вызываем инструмент
            if tool_name == "count_files":
                result = count_files_in_directory(tool_params.get("directory_path", "."))
            elif tool_name == "get_current_time":
                result = get_current_time()
            elif tool_name == "read_file":
                result = read_file_content(tool_params.get("file_path", ""))
            elif tool_name == "create_note":
                result = create_note(
                    tool_params.get("note_content", ""),
                    tool_params.get("note_name", None)
                )
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            tool_results.append({"tool": tool_name, "result": result})

        # Формируем промпт с результатами инструментов
        tools_prompt = f"""
        Результаты выполнения инструментов:
        {json.dumps(tool_results, ensure_ascii=False, indent=2)}
        
        Сформулируй итоговый ответ пользователю на основе этих результатов.
        """
        final_response = send_to_model(tools_prompt, system_prompt)

    # Обновляем историюW
    session.history.append({
        "role": "user",
        "content": request.prompt,
        "timestamp": datetime.now().isoformat()
    })
    session.history.append({
        "role": "assistant",
        "content": final_response,
        "timestamp": datetime.now().isoformat()
    })

    # Ограничиваем историю 20 сообщениями
    if len(session.history) > 20:
        session.history = session.history[-20:]

    # Сохраняем сессию
    sessions[session_id] = session

    return {
        "response": final_response,
        "session_id": session_id,
        "tools_used": bool(tool_calls)
    }

def parse_tool_calls(response: str) -> List[tuple]:
    """Parse tool calls from model response"""
    tool_calls = []
    lines = response.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ИНСТРУМЕНТ:"):
            tool_name = line.replace("ИНСТРУМЕНТ:", "").strip()
            i += 1
            if i < len(lines) and lines[i].strip().startswith("ПАРАМЕТРЫ:"):
                params_line = lines[i].strip().replace("ПАРАМЕТРЫ:", "").strip()
                try:
                    params = json.loads(params_line)
                    tool_calls.append((tool_name, params))
                except json.JSONDecodeError:
                    # Если JSON невалидный, пропускаем
                    pass
        i += 1

    return tool_calls

@app.get("/tools/count_files")
async def api_count_files(directory_path: str = "."):
    """API endpoint to count files"""
    return count_files_in_directory(directory_path)

@app.get("/tools/current_time")
async def api_current_time():
    """API endpoint to get current time"""
    return get_current_time()

@app.get("/tools/read_file")
async def api_read_file(file_path: str):
    """API endpoint to read file content"""
    return read_file_content(file_path)

@app.post("/tools/create_note")
async def api_create_note(note_content: str, note_name: Optional[str] = None):
    """API endpoint to create a note"""
    return create_note(note_content, note_name)

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session state"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session deleted"}

@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "message": "Qwen MCP Server is running",
        "endpoints": {
            "POST /prompt": "Send a prompt to the AI model",
            "GET /tools/count_files": "Count files in a directory",
            "GET /tools/current_time": "Get current time",
            "GET /tools/read_file": "Read file content",
            "POST /tools/create_note": "Create a text note",
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
    # Запускаем MCP сервер в фоновой задаче
    mcp_task = asyncio.create_task(run_mcp_server())

    # Запускаем FastAPI сервер
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Запускаем оба сервера
    await asyncio.gather(
        server.serve(),
        mcp_task
    )

if __name__ == "__main__":
    # Создаем папку для заметок если её нет
    os.makedirs("notes", exist_ok=True)

    # Запускаем приложение
    asyncio.run(main())
