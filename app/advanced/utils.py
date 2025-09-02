import json
import re
from functools import wraps
from typing import List

from app.storage.tarantool import tarantool_service


def cache_response(ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            cached = await tarantool_service.get(key)
            if cached is not None:
                return cached

            result = await func(*args, **kwargs)

            if isinstance(result, (dict, list)):
                if not (isinstance(result, dict) and "error" in result):
                    await tarantool_service.set(key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


def clean_xml_dict(data):
    """
    Рекурсивно удаляет префикс '@' из ключей словаря,
    полученного с помощью xmltodict.parse().
    Также обрабатывает списки и вложенные структуры.
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Убираем @ в начале ключа
            new_key = key.lstrip("@") if isinstance(key, str) else key
            cleaned[new_key] = clean_xml_dict(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_xml_dict(item) for item in data]
    else:
        return data


def parse_tool_calls(content: str) -> List[tuple]:
    if not content:
        return []

    # Поддержка нескольких форматов
    patterns = [
        r'ИНСТРУМЕНТ:\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*(?:ПАРАМЕТРЫ:)?\s*(\{.*\})?',
        r'инструмент:\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2) or "{}"
            try:
                args = json.loads(args_str) if args_str.strip() else {}
            except Exception:
                args = {}
            return [(tool_name, args)]
    return []
