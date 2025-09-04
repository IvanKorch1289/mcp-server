import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

from app.advanced.logging_client import logger

logging.getLogger("httpx").setLevel(logging.WARNING)


class AsyncHttpClient:
    _instance: Optional["AsyncHttpClient"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "AsyncHttpClient":
        # Разрешаем создание, но не гарантируем инициализацию
        # Проверку делегируем get_instance()
        return super().__new__(cls)

    @classmethod
    async def get_instance(cls) -> "AsyncHttpClient":
        """
        Асинхронный Singleton. Гарантирует, что клиент будет создан и проинициализирован.
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.__init_once()  # Только один раз
                    await instance._initialize()
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        # Запрещаем инициализацию напрямую
        raise RuntimeError(
            f"Нельзя создавать экземпляр {self.__class__.__name__} напрямую. "
            f"Используйте {self.__class__.__name__}.get_instance()"
        )

    def __init_once(self):
        """
        Вызывается только один раз при создании экземпляра.
        Аналог __init__, но безопасный для Singleton.
        """
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized: bool = False

    async def _initialize(self):
        """Инициализация клиента (асинхронная)"""
        if self._initialized:
            return

        transport = httpx.AsyncHTTPTransport(retries=0)
        self._client = httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(
                max_connections=30,
                max_keepalive_connections=15,
            ),
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=2.0),
            event_hooks={
                "request": [self._on_request],
                "response": [self._on_response],
            },
            transport=transport,
        )
        self._initialized = True

    async def _on_request(self, request: httpx.Request):
        request.extensions["start_time"] = asyncio.get_event_loop().time()
        logger.log_request(request)

    async def _on_response(self, response: httpx.Response):
        start_time = response.request.extensions.get("start_time", None)
        duration = asyncio.get_event_loop().time() - start_time if start_time else 0.0
        logger.log_response(response, duration=duration)

    async def _retry_request(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        **kwargs,
    ) -> httpx.Response:
        client = self._client
        if not client:
            raise RuntimeError("HTTP-клиент не инициализирован.")

        for attempt in range(max_retries):
            try:
                response = await client.request(method, url, **kwargs)

                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt < max_retries - 1:
                        delay = backoff_factor * (2**attempt)
                        await asyncio.sleep(delay)
                        continue
                response.raise_for_status()
                return response
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Request failed after {max_retries} attempts: {str(exc)}",
                        component="_retry_request",
                    )
                    raise
                delay = backoff_factor * (2**attempt)
                logger.warning(
                    f"Request error: {str(exc)}. Retrying in {delay} seconds...",
                    component="_retry_request",
                )
                await asyncio.sleep(delay)
        raise RuntimeError("Unreachable: failed after retries")

    async def fetch_all_pages(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data_extractor: Optional[Callable[[Dict[str, Any]], List[Any]]] = None,
        page_extractor: Optional[Callable[[Dict[str, Any]], Optional[int]]] = None,
        **kwargs,
    ) -> List[Any]:
        if not self._client:
            raise RuntimeError("Клиент не инициализирован.")

        all_data: List[Any] = []
        params = params or {}
        current_page = params.get("page", 1)
        seen_pages = set()

        def default_data_extractor(data: Dict[str, Any]) -> List[Any]:
            for key in ("data", "results", "items", "entries"):
                if isinstance(data.get(key), list):
                    return data[key]
            return []

        def default_page_extractor(data: Dict[str, Any]) -> Optional[int]:
            return (
                data.get("total_pages")
                or data.get("pagination", {}).get("total_pages")
                or data.get("meta", {}).get("total_pages")
            )

        extract_data = data_extractor or default_data_extractor
        extract_total_pages = page_extractor or default_page_extractor

        while True:
            if current_page in seen_pages:
                logger.warning(
                    "Обнаружена зацикленная пагинация на странице %d", current_page
                )
                break
            seen_pages.add(current_page)

            request_params = {**params, "page": current_page}

            try:
                response = await self._retry_request(
                    method=method,
                    url=url,
                    params=request_params,
                    **kwargs,
                )
                json_data = response.json()

                page_items = extract_data(json_data)

                all_data.extend(page_items)

                total_pages = extract_total_pages(json_data)

                if total_pages is None:
                    logger.info(
                        f"Пагинация не обнаружена. Останавливаемся после страницы {current_page}.",
                        component="fetch_all_pages",
                    )
                    break

                if current_page >= total_pages:
                    break

                current_page += 1

            except Exception as e:
                logger.error(
                    f"Ошибка при запросе страницы {current_page}: {str(e)}",
                    component="http_client",
                )
                break

        return all_data

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        if not self._client:
            raise RuntimeError("Клиент не инициализирован.")
        return await self._retry_request(method=method, url=url, **kwargs)

    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    @classmethod
    async def close_global(cls):
        if cls._instance is not None:
            await cls._instance.aclose()
            cls._instance = None
