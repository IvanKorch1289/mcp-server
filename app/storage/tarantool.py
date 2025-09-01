import tarantool
import msgpack
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio
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
            raise ConnectionError(f"Failed to connect to Tarantool: {e}")

    async def connect(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, self.connect_sync)

    def get_sync(self, key: str):
        try:
            if not self.connection:
                return None
            result = self.connection.select("cache", key)
            if result and len(result) > 0:
                value_packed, expires_at = result[0][1], result[0][2]
                if expires_at > time.time():
                    return msgpack.unpackb(value_packed, raw=False)
                else:
                    self.delete_sync(key)
            return None
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
            packed = msgpack.packb(value, use_bin_type=True)
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
