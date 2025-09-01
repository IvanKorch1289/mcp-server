from functools import wraps

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

            if isinstance(result, (dict, list)) and not (
                isinstance(result, dict) and result.get("error")
            ):
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
            new_key = key.lstrip('@') if isinstance(key, str) else key
            cleaned[new_key] = clean_xml_dict(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_xml_dict(item) for item in data]
    else:
        return data
