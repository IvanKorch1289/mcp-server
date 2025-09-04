import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import msgpack

import tarantool
from app.advanced.logging_client import logger
from app.settings import settings

# Пул для blocking-операций
_executor = ThreadPoolExecutor(max_workers=3)


class TarantoolClient:
    """
    Асинхронный клиент для Tarantool с поддержкой кэширования и TTL.
    Поддерживает:
      - get/set/delete
      - массовую инвалидацию кэша
      - автоматическое переподключение
      - логирование
    """

    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def get_instance(cls) -> "TarantoolClient":
        """Асинхронный Singleton с lazy-init"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = cls()
                    await instance._connect()
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        self._connection: Optional[tarantool.Connection] = None
        self._connected: bool = False
        self._space = "cache"

    async def _connect(self):
        """Асинхронное подключение через пул"""
        if self._connected and self._connection:
            return

        def connect_fn():
            try:
                conn = tarantool.connect(
                    host=settings.tarantool_host,
                    port=settings.tarantool_port,
                    user=settings.tarantool_user,
                    password=settings.tarantool_password,
                )
                return conn
            except Exception as e:
                logger.error(f"Tarantool connection failed: {e}", component="tarantool")
                raise ConnectionError(f"Cannot connect to Tarantool: {e}") from e

        loop = asyncio.get_event_loop()
        self._connection = await loop.run_in_executor(_executor, connect_fn)
        self._connected = True
        logger.info("Tarantool connected successfully", component="tarantool")

    async def _ensure_connection(self):
        """Проверяет соединение, переподключается при необходимости"""
        if not self._connected:
            await self._connect()

    async def get(self, key: str) -> Optional[Dict[Any, Any]]:
        """
        Получает значение по ключу.
        Если просрочено — удаляет и возвращает None.
        """
        await self._ensure_connection()

        def do_get():
            logger.debug(
                f"Tarantool.get() called for key='{key}'", component="tarantool_debug"
            )
            try:
                if not self._connection:
                    return None

                result = self._connection.select(self._space, key)
                if not result:
                    return None

                row = result[0]
                if len(row) < 3:
                    logger.warning(
                        f"Invalid row format for key '{key}': expected 3 fields, got {len(row)}",
                        component="tarantool",
                    )
                    return None

                value_packed, expires_at = row[1], row[2]

                # Проверка типа
                if not isinstance(value_packed, (bytes, bytearray)):
                    logger.warning(
                        f"Invalid value type for key '{key}': {type(value_packed)}, value={repr(value_packed)}",
                        component="tarantool",
                    )
                    return None

                now = time.time()
                if now > expires_at:
                    self._connection.delete(self._space, key)
                    logger.debug(
                        f"Cache expired and deleted: {key}", component="tarantool"
                    )
                    return None
                logger.debug(
                    f"Tarantool.get() unpacking value for key='{key}': type={type(value_packed)}, len={len(value_packed)}",
                    component="tarantool_debug",
                )
                # Распаковываем
                unpacked = msgpack.unpackb(
                    value_packed,
                    raw=False,
                    max_str_len=100_000,
                    max_bin_len=100_000,
                    max_array_len=1000,
                    max_map_len=1000,
                    max_ext_len=100_000,
                )

                # Убедимся, что результат — словарь или список
                if not isinstance(unpacked, (dict, list)):
                    logger.warning(
                        f"Unpacked value for key '{key}' is not dict/list: {type(unpacked)}",
                        component="tarantool",
                    )
                    return None

                return unpacked

            except tarantool.DatabaseError as e:
                logger.error(
                    f"Tarantool DB error on GET {key}: {e}", component="tarantool"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error on GET {key}: {e}", component="tarantool"
                )
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, do_get)

    async def set(self, key: str, value: Any, ttl: int = 3600):
        await self._ensure_connection()

        # 🔥 Логируем ТИП и значение
        logger.debug(
            f"Tarantool.set() called for key='{key}', type(value)={type(value)}, value_preview={str(value)[:200]}",
            component="tarantool_debug",
        )

        if not isinstance(value, (dict, list)):
            logger.warning(
                f"Tarantool.set() value is not dict/list (type={type(value)}). Wrapping.",
                component="tarantool",
            )
            value = {
                "value": str(value),
                "type": type(value).__name__,
                "warning": "Automatically converted non-dict/list value",
                "timestamp": time.time(),
            }

        def do_set():
            try:
                packed = msgpack.packb(value, use_bin_type=True, strict_types=False)
                expires_at = time.time() + ttl
                self._connection.replace(self._space, (key, packed, expires_at))
                logger.debug(f"Cache set: {key}, ttl={ttl}", component="tarantool")
            except Exception as e:
                logger.error(
                    f"Unexpected error on SET {key}: {e}", component="tarantool"
                )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, do_set)

        def do_set():
            try:
                packed = msgpack.packb(value, use_bin_type=True, strict_types=False)
                expires_at = time.time() + ttl
                self._connection.replace(self._space, (key, packed, expires_at))
                logger.debug(f"Cache set: {key}, ttl={ttl}", component="tarantool")
            except tarantool.DatabaseError as e:
                logger.error(
                    f"Tarantool DB error on SET {key}: {e}", component="tarantool"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error on SET {key}: {e}", component="tarantool"
                )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, do_set)

    async def delete(self, key: str):
        """Удаляет ключ"""
        await self._ensure_connection()

        def do_delete():
            try:
                self._connection.delete(self._space, key)
                logger.debug(f"Cache deleted: {key}", component="tarantool")
            except Exception as e:
                logger.error(f"Error deleting key {key}: {e}", component="tarantool")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, do_delete)

    async def invalidate_all_keys(self, confirm: bool = False):
        """
        Полная инвалидация всех ключей в пространстве 'cache'.
        ⚠️ Опасная операция! Требует подтверждения.
        """
        if not confirm:
            logger.warning(
                "invalidate_all_keys() called without confirm=True. Aborted.",
                component="tarantool",
            )
            return

        await self._ensure_connection()

        def do_truncate():
            try:
                self._connection.call("box.space.cache:truncate")
                logger.warning(
                    "All cache keys invalidated (space 'cache' truncated)",
                    component="tarantool",
                )
            except Exception as e:
                logger.error(
                    f"Failed to invalidate all keys: {e}", component="tarantool"
                )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, do_truncate)

    async def close(self):
        """Закрывает соединение"""
        if self._connection:

            def close_fn():
                try:
                    self._connection.close()
                    logger.info("Tarantool connection closed", component="tarantool")
                except Exception as e:
                    logger.error(f"Error closing Tarantool: {e}", component="tarantool")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_executor, close_fn)
        self._connected = False
        self._connection = None

    @classmethod
    async def close_global(cls):
        """Закрывает глобальный экземпляр"""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None
