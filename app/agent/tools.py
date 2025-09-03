import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles

from app.services.fetch_data import fetch_company_info


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
            os.path.getsize(os.path.join(directory_path, f)) for f in files
        )

        return {
            "directory": os.path.abspath(directory_path),
            "file_count": len(files),
            "directory_count": len(directories),
            "total_size_bytes": total_size,
            "files": files,
            "directories": directories,
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
        "timezone": str(now.astimezone().tzinfo),
    }


async def read_file_content(file_path: str) -> Dict[str, Any]:
    """Read content of a file"""
    try:
        if not os.path.exists(file_path):
            return {"error": f"File {file_path} does not exist"}

        if not os.path.isfile(file_path):
            return {"error": f"{file_path} is not a file"}

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        return {
            "file_path": os.path.abspath(file_path),
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": content.count("\n") + 1,
            "encoding": "utf-8",
        }
    except UnicodeDecodeError:
        return {"error": "Cannot decode file content as UTF-8 text"}
    except PermissionError:
        return {"error": f"Permission denied reading {file_path}"}
    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}


async def create_note(
    note_content: str, note_name: Optional[str] = None
) -> Dict[str, Any]:
    try:
        if not note_name:
            note_name = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        else:
            # Очистка имени
            note_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", note_name)

        notes_dir = os.path.join(os.getcwd(), "notes")
        os.makedirs(notes_dir, exist_ok=True)
        filepath = os.path.join(notes_dir, note_name)

        # Избегаем перезаписи
        counter = 1
        original_name, ext = os.path.splitext(note_name)
        while os.path.exists(filepath):
            note_name = f"{original_name}_{counter}{ext}"
            filepath = os.path.join(notes_dir, note_name)
            counter += 1

        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(note_content)

        return {
            "status": "success",
            "file_path": filepath,
            "filename": note_name,
            "size_bytes": len(note_content.encode("utf-8")),
            "message": f"Note created successfully at {filepath}",
        }
    except PermissionError:
        return {"error": "Permission denied: cannot create note file"}
    except Exception as e:
        return {"error": f"Error creating note: {str(e)}"}


async def fetch_company_info_for_analyze(inn: str) -> Dict[str, Any]:
    if not re.fullmatch(r"\d{10}|\d{12}", inn):
        return {"error": "Invalid INN format. Must be 10 or 12 digits."}

    return await fetch_company_info(inn)
