Вот полный и красиво оформленный README для вашего проекта 🚀  

```markdown
# 🤖 GigaChat MCP AI-Агент с инструментами и кэшированием

Умный ассистент на базе **GigaChat** с поддержкой вызова внешних инструментов, анализа данных и кэширования через **Tarantool**.  
Проект построен с использованием `LangGraph`, `LangChain`, `MCP (Model Context Protocol)` и `FastAPI`.

---

## 🔧 Основные возможности

- ✅ **Многошаговые агенты** с поддержкой вызова инструментов  
- ✅ **Анализ юридических лиц по ИНН** через DaData и InfoSphere  
- ✅ **Встроенные инструменты**: работа с файлами, время, заметки  
- ✅ **Кэширование запросов** через Tarantool (с TTL)  
- ✅ **MCP-совместимый сервер** для интеграции с IDE и агентами  
- ✅ **Сессии пользователей** с историей диалога  
- ✅ **Русский язык** по умолчанию  

---

## 📦 Технологии

| Технология   | Назначение |
|--------------|------------|
| `GigaChat`   | Языковая модель от Сбера |
| `LangGraph`  | Оркестрация агентов и графов |
| `FastAPI`    | HTTP API |
| `Tarantool`  | Быстрое кэширование с TTL |
| `MCP Server` | Поддержка Model Context Protocol |
| `aiohttp`    | Асинхронные HTTP-запросы |
| `xmltodict`  | Парсинг XML от InfoSphere |

---

## ⚙️ Установка и настройка

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```
> Требуется **Python 3.10+**

---

### 2. Настройка переменных окружения
Создайте файл **.env**:

```env
# GigaChat
GIGA_API_KEY=your_gigachat_api_key

# DaData
DADATA_API_KEY=your_dadata_api_key
DADATA_URL=https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party

# InfoSphere
INFOSPHERE_LOGIN=your_login
INFOSPHERE_PASSWORD=your_password
INFOSPHERE_URL=https://api.infosphere.ru/xml

# Tarantool
TARANTOOL_HOST=localhost
TARANTOOL_PORT=3301
TARANTOOL_USER=guest
TARANTOOL_PASSWORD=
```

🔐 Все ключи загружаются из `app/settings.py` через `pydantic.BaseSettings`.

---

## 🚀 Запуск

```bash
python -m app.main
```

Сервер запустится на:  
- **HTTP API:** [http://localhost:8000](http://localhost:8000)  
- **MCP Server:** через `stdio` (для IDE)  

---

## 🌐 API Эндпоинты

### 🔹 Основной запрос к агенту
`POST /prompt`

```json
{
  "prompt": "Проанализируй клиента с ИНН 7702302345",
  "session_id": "session_123"
}
```

**Ответ:**
```json
{
  "response": "Компания активна, риск низкий...",
  "session_id": "session_123",
  "tools_used": true,
  "timestamp": "2025-09-01T12:00:00"
}
```

---

### 🔹 История сессии
`GET /sessions/{session_id}` — Просмотр истории диалога  

### 🔹 Получение данных по ИНН
`GET /client/info/{inn}` — Отладка запросов  

---

## 🧰 Доступные инструменты

| Инструмент           | Назначение |
|----------------------|------------|
| `get_current_time`   | Получить текущее время |
| `count_files`        | Посчитать файлы в директории |
| `read_file`          | Прочитать содержимое файла |
| `create_note`        | Создать текстовую заметку |
| `fetch_company_info` | Получить данные о компании по ИНН |

📌 При анализе по ИНН агент автоматически вызывает `fetch_company_info`, обрабатывает результат и возвращает структурированный вывод.  

---

## 🔄 Логика работы агента

1. Пользователь отправляет запрос  
2. Агент определяет, нужен ли инструмент  
3. Если да — вызывает инструмент и сохраняет результат  
4. Модель анализирует результат  
5. Возвращает итоговый ответ  

🔁 Граф работы:
```
agent → tools → agent → ответ
```

---

## 🗃️ Кэширование

- Все тяжёлые запросы (**DaData**, **InfoSphere**) кэшируются в **Tarantool**  
- TTL: **1–2 часа** (зависит от инструмента)  
- Используется **msgpack** для сериализации  
- Обработка ошибок `max_map_len`, некорректных типов и повреждённых данных  

---

## 🛠️ Отладка

### Проверка Tarantool
```bash
tarantoolctl connect localhost:3301
> box.space.cache:select()
```

### Логи
- **INFO** — основные события  
- **ERROR** — ошибки Tarantool, вызовов, парсинга  

---

## 🔗 MCP-интеграция

Сервер совместим с **MCP-клиентами** (например, VS Code или JetBrains).  

---

## 🧪 Примеры запросов

### Узнать текущее время
```bash
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Который час?",
    "session_id": "test1"
  }'
```

### Анализ клиента по ИНН
```bash
curl -X POST http://localhost:8000/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Проанализируй клиента с ИНН 7702302345",
    "session_id": "client_analysis_1"
  }'
```

---

## 📁 Структура проекта

```
app/
├── main.py            # Точка входа
├── server.py          # Граф агента и MCP-сервер
├── session.py         # Управление сессиями
├── tools.py           # Инструменты
├── fetch_data.py      # Запросы к DaData и InfoSphere
├── storage/
│   └── tarantool.py   # Кэширование
├── prompts.py         # Системный промпт
├── settings.py        # Настройки
└── utils.py           # Вспомогательные функции
```

---

## 📄 Лицензия

Проект разработан для внутреннего использования.  
Распространяется **без лицензии**.

---

💡 Разработчик: *Korch Ivan*  
📅 Последнее обновление: *Сентябрь 2025*