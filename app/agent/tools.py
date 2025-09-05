# app/agent/tools.py
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles

from app.services.fetch_data import fetch_company_info


async def count_files_in_directory(directory_path: str) -> Dict[str, Any]:
    """Подсчитывает количество файлов и папок в указанной директории.

    Используй, когда пользователь спрашивает о содержимом папки: сколько файлов, папок, их размер.
    Функция проверяет существование пути, права доступа и возвращает детальную статистику.

    Аргументы:
        directory_path (str): путь к директории (обязательный)

    Возвращает:
        Словарь с:
        - file_count: количество файлов
        - directory_count: количество подпапок
        - total_size_bytes: общий размер в байтах
        - files: список имён файлов
        - directories: список имён папок
        При ошибке — объект с полем 'error'.
    """
    try:
        if not os.path.exists(directory_path):
            return {
                "error": {
                    "type": "directory_not_found",
                    "message": f"Directory {directory_path} does not exist",
                }
            }

        if not os.path.isdir(directory_path):
            return {
                "error": {
                    "type": "not_a_directory",
                    "message": f"{directory_path} is not a directory",
                }
            }

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
            "status": "success",
            "directory": os.path.abspath(directory_path),
            "file_count": len(files),
            "directory_count": len(directories),
            "total_size_bytes": total_size,
            "files": files,
            "directories": directories,
            "summary": f"Found {len(files)} files and {len(directories)} directories",
        }
    except PermissionError:
        return {
            "error": {
                "type": "permission_denied",
                "message": f"Permission denied accessing {directory_path}",
            }
        }
    except Exception as e:
        return {
            "error": {
                "type": "unexpected_error",
                "message": f"Unexpected error: {str(e)}",
            }
        }


async def get_current_time() -> str:
    """Возвращает текущую дату и время в формате 'ГГГГ-ММ-ДД ЧЧ:ММ:СС'.

    Используй, когда пользователь спрашивает 'который час', 'сегодняшнюю дату' или 'текущее время'.
    Не используй, если в запросе есть 'не нужно время' или 'без даты'.

    Аргументы: отсутствуют.

    Возвращает:
       Структурированный словарь с датой и временем.
    """
    now = datetime.now()
    return {
        "status": "success",
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "UTC+3",
        "summary": f"Текущее время: {now.strftime('%H:%M')}",
    }


async def read_file_content(file_path: str) -> Dict[str, Any]:
    """Читает содержимое текстового файла и возвращает его как строку.

    Используй, когда нужно получить текст из файла для анализа, цитирования или проверки.
    Поддерживает только UTF-8. Проверяет существование файла и права доступа.

    Аргументы:
        file_path (str): путь к файлу (обязательный)

    Возвращает:
        Словарь с:
        - content: текст файла
        - size_bytes: размер в байтах
        - line_count: количество строк
        При ошибке — объект с полем 'error'.
    """
    try:
        if not os.path.exists(file_path):
            return {
                "error": {
                    "type": "file_not_found",
                    "message": f"File {file_path} does not exist",
                }
            }

        if not os.path.isfile(file_path):
            return {
                "error": {"type": "not_a_file", "message": f"{file_path} is not a file"}
            }

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        return {
            "status": "success",
            "file_path": os.path.abspath(file_path),
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
            "line_count": content.count("\n") + 1,
            "encoding": "utf-8",
            "summary": f"Successfully read {len(content)} characters",
        }
    except UnicodeDecodeError:
        return {
            "error": {
                "type": "encoding_error",
                "message": "Cannot decode file content as UTF-8 text",
            }
        }
    except PermissionError:
        return {
            "error": {
                "type": "permission_denied",
                "message": f"Permission denied reading {file_path}",
            }
        }
    except Exception as e:
        return {
            "error": {"type": "read_error", "message": f"Error reading file: {str(e)}"}
        }


async def create_note(
    note_content: str, note_name: Optional[str] = None
) -> Dict[str, Any]:
    """Создаёт текстовый файл (.txt) с указанным содержимым.

    Используй, только если пользователь явно просит 'сохранить', 'записать в файл', 'создать заметку'.
    Не используй, если в запросе есть 'заметку не создавай' или подобное ограничение.

    Аргументы:
        note_content (str): текст для сохранения (обязательный)
        note_name (str, опционально): имя файла. Если не задано — генерируется автоматически.

    Поведение:
        - Генерирует имя по названию компании и времени, если не задано.
        - Очищает имя от недопустимых символов.
        - Предотвращает перезапись, добавляя суффикс _1, _2 и т.д.
        - Создаёт папку 'notes', если её нет.

    Возвращает:
        Словарь с:
        - file_path: полный путь к файлу
        - filename: имя файла
        - size_bytes: размер в байтах
        При ошибке — объект с полем 'error'.
    """
    try:
        # Если имя содержит шаблоны, код или недопустимые символы — игнорируем
        if note_name and (
            "{" in note_name or "}" in note_name or "(" in note_name or ")" in note_name
        ):
            note_name = None

        if not note_name:
            # Генерируем имя на основе контента
            lines = note_content.splitlines()
            company_line = next(
                (
                    line
                    for line in lines
                    if "Наименование:" in line
                    or "Общество с ограниченной ответственностью" in line
                ),
                None,
            )
            if company_line:
                match = re.search(r'"([^"]+)"', company_line)
                company_name = match.group(1).replace(" ", "_") if match else "client"
            else:
                company_name = "client"

            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            note_name = f"{company_name}-{timestamp}.txt"
        else:
            note_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", note_name)
            if not note_name.endswith((".txt", ".md")):
                note_name += ".txt"

        notes_dir = os.path.join(os.getcwd(), "notes")
        os.makedirs(notes_dir, exist_ok=True)
        filepath = os.path.join(notes_dir, note_name)

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
            "summary": f"Note '{note_name}' saved with {len(note_content)} characters",
        }
    except PermissionError:
        return {
            "error": {
                "type": "permission_denied",
                "message": "Permission denied: cannot create note file",
            }
        }
    except Exception as e:
        return {
            "error": {
                "type": "creation_error",
                "message": f"Error creating note: {str(e)}",
            }
        }


async def fetch_company_info_for_analyze(inn: str) -> Dict[str, Any]:
    """Получает полную информацию о компании по ИНН из DaData и InfoSphere.

    Используй, когда требуется анализ клиента, проверка юрлица, оценка рисков.
    Возвращает данные о статусе, руководителе, судебных делах, исполнительных производствах, поддержке.

    Аргументы:
        inn (str): ИНН компании (10 или 12 цифр, обязательный)

    Проверки:
        - Формат ИНН (только цифры, длина 10 или 12)
        - Существование компании

    Возвращает:
        Словарь с полями:
        - company_info: данные из DaData, InfoSphere, Casebook
        - inn: переданный ИНН
        При ошибке — объект с полем 'error'.
    """
    if not re.fullmatch(r"\d{10}|\d{12}", inn):
        return {
            "error": {
                "type": "invalid_inn",
                "message": "Invalid INN format. Must be 10 or 12 digits.",
            }
        }

    try:
        result = await fetch_company_info(inn)
        if "error" in result:
            return {
                "error": {"type": "company_fetch_failed", "message": result["error"]}
            }

        return {
            "status": "success",
            "company_info": result,
            "inn": inn,
            "summary": f"Successfully fetched data for INN {inn}",
        }
    except Exception as e:
        return {
            "error": {
                "type": "external_api_error",
                "message": f"Failed to fetch company info: {str(e)}",
            }
        }
