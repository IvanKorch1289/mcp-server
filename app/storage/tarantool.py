import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import msgpack

import tarantool
from app.settings import settings

executor = ThreadPoolExecutor(max_workers=2)


class TarantoolService:
    def __init__(self):
        self.connection: Optional[tarantool.Connection] = None

    def connect_sync(self):
        try:
            self.connection = tarantool.connect(
                host=settings.tarantool_host,
                port=settings.tarantool_port,
                user=settings.tarantool_user,
                password=settings.tarantool_password,
            )
            return "OK"
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Tarantool: {e}") from e

    async def connect(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.connect_sync)

    def get_sync(self, key: str):
        try:
            if not self.connection:
                return None
            result = self.connection.select("cache", key)
            if not result or len(result) == 0:
                return None

            row = result[0]
            if len(row) < 3:
                print("Tarantool GET error: row has less than 3 fields")
                return None

            value_packed = row[1]
            expires_at = row[2]

            if not isinstance(value_packed, (bytes, bytearray)):
                print(f"Tarantool GET error: expected bytes, got {type(value_packed)}")
                return None

            if time.time() > expires_at:
                self.delete_sync(key)
                return None

            return msgpack.unpackb(
                value_packed,
                raw=False,
                strict_map_key=False,
                max_map_len=1000,
                max_list_len=1000,
            )

        except Exception as e:
            print(f"Tarantool GET error: {e}")
            return None

    async def get(self, key: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self.get_sync, key)

    def set_sync(self, key: str, value: Dict[Any, Any], ttl: int = 3600):
        try:
            if not self.connection:
                return

            if not isinstance(value, (dict, list)):
                print(f"Tarantool SET error: value is not dict/list, got {type(value)}")
                return

            packed = msgpack.packb(
                value,
                use_bin_type=True,
                strict_types=False,  # разрешает не-JSON типы
            )
            expires_at = time.time() + ttl
            self.connection.replace("cache", (key, packed, expires_at))
        except Exception as e:
            print(f"Tarantool SET error: {e}")

    async def set(self, key: str, value: Dict[Any, Any], ttl: int = 3600):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.set_sync, key, value, ttl)

    def delete_sync(self, key: str):
        try:
            if self.connection:
                self.connection.delete("cache", key)
        except Exception as e:
            print(f"Tarantool DELETE error: {e}")

    async def delete(self, key: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.delete_sync, key)

    def close_sync(self):
        if self.connection:
            self.connection.close()

    async def close(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.close_sync)


# Глобальный экземпляр
tarantool_service = TarantoolService()
