import logging
from datetime import datetime
from pathlib import Path

import httpx
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

# Настройка папки для логов
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Глобальный логгер приложения
app_logger = logging.getLogger("mcp-server")
app_logger.setLevel(logging.DEBUG)
app_logger.handlers.clear()


class AppLogger:
    """
    Универсальный логгер с цветным выводом в терминал и ежедневными файлами.
    Может использоваться в любом месте приложения.
    """

    _instance = None
    _console = Console()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_handlers()
        return cls._instance

    def _setup_handlers(self):
        """Настройка: цветной терминал + файлы по дням"""
        # 1. Терминал
        rich_handler = RichHandler(
            console=self._console,
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        app_logger.addHandler(rich_handler)

        # 2. Файл (ежедневный)
        self._add_timed_file_handler()

    def _add_timed_file_handler(self):
        """Добавляет хэндлер для записи в файл с именем по дате"""
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = LOGS_DIR / f"{today}.log"

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        app_logger.addHandler(file_handler)

    def _renew_file_handler(self):
        """Обновляет файловый хэндлер при смене дня"""
        for handler in app_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                app_logger.removeHandler(handler)
        self._add_timed_file_handler()

    def _ensure_daily_log(self):
        """Проверяет, нужно ли обновить файл лога (смена дня)"""
        today = datetime.now().strftime("%Y-%m-%d")
        current_file = LOGS_DIR / f"{today}.log"
        if not any(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == current_file
            for h in app_logger.handlers
        ):
            self._renew_file_handler()

    def log_request(self, request: httpx.Request):
        """Логирует HTTP-запрос"""
        self._ensure_daily_log()

        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", width=10)
        table.add_column()

        table.add_row("➡️ ЗАПРОС", "")
        table.add_row("Метод", request.method)
        table.add_row("URL", str(request.url))
        table.add_row("Заголовки", self._format_headers(request.headers))

        if request.content:
            body = request.content.decode("utf-8", errors="replace")
            table.add_row("Тело", self._truncate(body, 500))

        app_logger.info("")
        self._console.print(table)

    def log_response(self, response: httpx.Response, duration: float = 0.0):
        """Логирует HTTP-ответ (с обязательным duration)"""
        self._ensure_daily_log()

        request = response.request
        status_color = "red" if response.is_error else "green"

        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold cyan", width=12)
        table.add_column()

        table.add_row("⬅️ ОТВЕТ", "")
        table.add_row("Метод", request.method)
        table.add_row("URL", str(request.url))
        table.add_row("Статус", Text(str(response.status_code), style=status_color))
        table.add_row("Время", f"{duration:.2f} с")

        try:
            body = response.text
            if body:
                table.add_row("Тело", self._truncate(body, 500))
        except Exception:
            pass

        app_logger.info("")
        self._console.print(table)

    def _format_headers(self, headers: httpx.Headers) -> str:
        return "\n".join(
            f"{k}: {v}"
            for k, v in headers.items()
            if k.lower() not in ["authorization", "cookie"]
        )

    def _truncate(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    # === Удобные методы для общего логирования ===

    def info(self, message: str, component: str = "app"):
        self._ensure_daily_log()
        app_logger.info(f"[{component.upper()}] {message}")

    def error(self, message: str, component: str = "app", exc_info=False):
        self._ensure_daily_log()
        app_logger.error(f"[{component.upper()}] {message}", exc_info=exc_info)

    def debug(self, message: str, component: str = "app"):
        self._ensure_daily_log()
        app_logger.debug(f"[{component.upper()}] {message}")

    def warning(self, message: str, component: str = "app"):
        self._ensure_daily_log()
        app_logger.warning(f"[{component.upper()}] {message}")

    def exception(self, message: str, component: str = "app"):
        self._ensure_daily_log()
        app_logger.exception(f"[{component.upper()}] {message}")


# Единый экземпляр логгера (можно импортировать где угодно)
logger = AppLogger()
