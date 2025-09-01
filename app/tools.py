import os
from datetime import datetime
from typing import Dict, Any, Optional
import aiofiles
import asyncio
import re


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


async def fetch_from_dadata(inn: str) -> Dict[str, Any]:
    try:
        await asyncio.sleep(0.1)
        return {
            "source": "dadata",
            "status": "success",
            "data": {
                "name": "ООО Ромашка",
                "inn": inn,
                "ogrn": "1234567890123",
                "address": "г. Москва, ул. Ленина, д. 1",
                "kpp": "770101001",
                "management": {"name": "Иванов И.И.", "post": "Генеральный директор"},
                "state": {"status": "ACTIVE", "registration_date": 1234567890}
            }
        }
    except Exception as e:
        return {"source": "dadata", "status": "error", "error": str(e)}


async def fetch_from_infosphere(inn: str) -> Dict[str, Any]:
    try:
        await asyncio.sleep(0.2)
        return {
            "source": "infosphere",
            "status": "success",
            "data": {
                "inn": inn,
                "risk_score": 0.3,
                "sanctions": False,
                "negative_news": ["Нет"],
                "affiliations": ["ООО Ягода", "АО Солнце"],
                "financial_stability": "medium"
            }
        }
    except Exception as e:
        return {"source": "infosphere", "status": "error", "error": str(e)}


async def fetch_company_info(inn: str) -> Dict[str, Any]:
    if not re.fullmatch(r"\d{10}|\d{12}", inn):
        return {"error": "Invalid INN format. Must be 10 or 12 digits."}

    dadata_task = asyncio.create_task(fetch_from_dadata(inn))
    infosphere_task = asyncio.create_task(fetch_from_infosphere(inn))
    dadata_result, infosphere_result = await asyncio.gather(dadata_task, infosphere_task, return_exceptions=True)

    if isinstance(dadata_result, Exception):
        dadata_result = {"source": "dadata", "status": "error", "error": str(dadata_result)}
    if isinstance(infosphere_result, Exception):
        infosphere_result = {"source": "infosphere", "status": "error", "error": str(infosphere_result)}

    return {
        "request_inn": inn,
        "sources": {"dadata": dadata_result, "infosphere": infosphere_result},
        "summary": {
            "is_active": (dadata_result.get("data", {}).get("state", {}).get("status") == "ACTIVE") if dadata_result.get("status") == "success" else None,
            "risk_level": infosphere_result.get("data", {}).get("risk_score") if infosphere_result.get("status") == "success" else None,
            "has_sanctions": infosphere_result.get("data", {}).get("sanctions") if infosphere_result.get("status") == "success" else None,
            "company_name": dadata_result.get("data", {}).get("name") if dadata_result.get("status") == "success" else None,
        }
    }
