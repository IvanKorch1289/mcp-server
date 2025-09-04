import hashlib
from functools import wraps
from typing import Any, Callable

from app.advanced.logging_client import logger
from app.storage.tarantool import TarantoolClient


def cache_response(ttl: int = 3600):
    """
    Декоратор для кэширования результата асинхронной функции в Tarantool.
    Генерирует ключ на основе имени функции и аргументов.
    Автоматически сериализует/десериализует через msgpack/json.
    Не кэширует ошибки.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Получаем клиент
            cache = await TarantoolClient.get_instance()

            # Генерируем ключ
            key_parts = [func.__module__, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

            # Хешируем, чтобы избежать длинных ключей (> 255 символов)
            raw_key = ":".join(key_parts)
            key = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
            cache_key = f"cache:{key}"

            try:
                # Попробуем получить из кэша
                cached = await cache.get(cache_key)
                if cached is not None:
                    logger.debug(
                        f"Cache HIT: {raw_key} → {cache_key}", component="cache"
                    )
                    return cached
                logger.debug(f"Cache MISS: {raw_key} → {cache_key}", component="cache")
            except Exception as e:
                logger.warning(
                    f"Cache GET error for {cache_key}: {e}", component="cache"
                )

            # Выполняем функцию
            try:
                result = await func(*args, **kwargs)
            except Exception:
                logger.warning(
                    f"Function {func.__name__} raised exception, not caching",
                    component="cache",
                )
                raise

            # Кэшируем только если результат — dict/list и не содержит "error"
            if isinstance(result, (dict, list)):
                if not (isinstance(result, dict) and "error" in result):
                    try:
                        await cache.set(cache_key, result, ttl=ttl)
                        logger.debug(
                            f"Cache SET: {cache_key}, ttl={ttl}", component="cache"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Cache SET failed for {cache_key}: {e}", component="cache"
                        )
                else:
                    logger.debug(
                        f"Skip caching error response: {result}", component="cache"
                    )
            else:
                logger.debug(
                    f"Skip caching non-serializable result: {type(result)}",
                    component="cache",
                )

            return result

        return wrapper

    return decorator


def clean_xml_dict(data):
    """
    Рекурсивно удаляет префикс '@' и '#' из ключей словаря.
    """
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            new_key = key
            if isinstance(key, str):
                new_key = key.lstrip("@#")
            cleaned[new_key] = clean_xml_dict(value)
        return cleaned
    elif isinstance(data, list):
        return [clean_xml_dict(item) for item in data]
    else:
        return data
