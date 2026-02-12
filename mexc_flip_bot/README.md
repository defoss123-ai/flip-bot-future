# mexc_flip_bot

Каркас проекта для будущего торгового бота MEXC Futures (Python 3.11+).

## Структура

```text
mexc_flip_bot/
  app/
  web/
  data/
  logs/
  config.yml.example
  requirements.txt
  README.md
```

## Установка

1. Перейдите в каталог проекта:
   ```bash
   cd mexc_flip_bot
   ```
2. Создайте и активируйте виртуальное окружение:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
4. Создайте рабочий конфиг:
   ```bash
   cp config.yml.example config.yml
   ```

## Запуск

Запуск каркаса бота:

```bash
python -m app.main
```

После запуска автоматически создаётся SQLite база `data/bot.db` с таблицами:
- `signals`
- `orders`
- `trades`

Ожидаемый вывод в логах: `Bot started`.

## Хранилище

В `app/database.py` реализован async-класс `Database` (SQLAlchemy + aiosqlite) с методами:
- `init_db()`
- `add_signal()`, `mark_signal_status()`
- `add_order()`, `update_order_status()`
- `add_trade()`, `update_trade()`
- `list_recent_trades()`, `list_recent_orders()`
- `stats()`

`stats()` возвращает агрегаты:
- `total_trades`
- `winrate`
- `pnl_day`
- `pnl_week`
- `pnl_month`

## Signal parser

Добавлен `app/signal_parser.py`:
- парсит поля `Price DEX`, `Price MEXC`, `Spread`, `Liquidity`, `Avg Align Time`;
- применяет фильтры по `min_spread_percent`, `min_liquidity_usd`, `max_align_age_sec`;
- строит анти-дубль `sha256(normalized_text)`;
- воркер в `app/main.py` сохраняет сигнал в SQLite со статусом `new` или `filtered`.

## MEXC paper client

Добавлен `app/mexc_client.py` с интерфейсом `MexcFuturesClient` (без реальных HTTP-запросов):
- `test_connection()`, `get_contracts()`, `set_leverage()`
- `place_order()`, `get_order()`, `cancel_order()`
- `place_tp_sl()`, `apply_market_price()`

Paper-mode поведение:
- ордера хранятся в памяти и записываются в таблицу `orders`;
- `market` заполняется сразу по последней `mexc_price` (обновляется из входящего сигнала);
- `limit` заполняется по простой симуляции (`buy`: `current<=limit`, `sell`: `current>=limit`).

## Trade engine

Добавлен `app/trade_engine.py`:
- каждые 2 секунды берет один `signals.status=new`;
- проверяет `running`, `max_positions` и cooldown по символу;
- определяет направление (`dex_price > mexc_price => long`, иначе short);
- открывает сделку через `MexcFuturesClient` (`market/limit`), ждёт fill до `entry_timeout_sec`;
- при timeout применяет `retry_mode` (`cancel`, `cancel_then_market`, `cancel_then_relimit`);
- после fill ставит `TP/SL` и переводит trade в `open`;
- все действия логируются и пишутся в SQLite (`signals/orders/trades`).

## Telegram listener

Добавлен `app/telegram_listener.py` на базе Telethon.

Поведение в `app/main.py`:
- создаётся `asyncio.Queue` для входящих сообщений;
- listener и TradeEngine запускаются только если в `data/state.yml` установлено `running: true` (кнопки Start/Stop в Web UI);
- каждое сообщение логируется (первые 120 символов);
- если Telegram/Telethon не настроены, приложение выводит понятную ошибку и не падает.

## Live MEXC keys

Для `live` режима задайте ключи в `config.yml` (или `data/config_runtime.yml`):

```yaml
mexc:
  api_key: your_trade_only_key
  api_secret: your_secret
```

Важно:
- используйте ключ **только с trade-permissions**, без withdraw;
- сначала отладьте стратегию в `paper` режиме.

Есть минимальный smoke-test:

```bash
python -m app.test_mexc
```

Он печатает:
- `OK`/`FAIL` по `test_connection()`
- первые 10 контрактов
- last price для BTC.

## Web-панель управления

```bash
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
```

Маршруты панели:
- `GET /` — Dashboard (статус бота + карточки статистики + последние 10 сигналов)
- `GET /settings` — форма runtime-настроек
- `POST /settings` — сохранение в `data/config_runtime.yml`
- `GET /trades` — последние 200 сделок
- `GET /orders` — последние 200 ордеров
- `POST /trades/{id}/close` — кнопка `Close now` (пока заглушка)
- `POST /bot/start`, `POST /bot/stop`
- `POST /bot/paper`, `POST /bot/live`
- `WS /ws/dashboard` — live-обновления Dashboard (каждые ~2 сек): pnl/winrate/trades/open_positions/exposure/system_health
- `GET /api/equity?range=day|week|month|all` — данные equity curve (кумулятивный pnl закрытых сделок)

## Важно

- Реальная торговля и интеграция с Telegram пока не реализованы.
- Реальные запросы к MEXC пока не выполняются, используется paper-mode симуляция.


## Stability improvements

- `TelegramListener` now reconnects automatically with exponential backoff (1s, 2s, 4s ... up to 30s) and handles `FloodWaitError` without crashing.
- `MexcFuturesClientReal` wraps all requests in `safe_request()` with retries for timeout/5xx (up to 3 attempts), and **no retry** on signature-related errors.
- Added `PositionMonitor` (`app/position_monitor.py`) that keeps syncing open exchange positions with local `trades` every 3 seconds, warns if API sync takes >10s, and never stops the loop on temporary failures.
- Added `HealthMonitor` (`app/health.py`) that tracks API/TELEGRAM/DB health. If API fails 3 checks in a row, new entries are temporarily paused (without force-closing positions).
- Dashboard now includes **System Health** block: `API`, `TELEGRAM`, `DB`, and entries state (`RUNNING`/`PAUSED`).


## Recovery on restart

On bot startup (`python -m app.main`) the app performs recovery sync:
- loads open exchange positions and reconciles them with local `trades`;
- creates missing `open` trades if position exists without DB trade;
- closes stale `open` trades with `close_reason=unknown` if position no longer exists;
- checks `orders.status=placed` against exchange status and updates canceled/rejected/filled rows;
- sets `data/state.yml -> recovered: true` on successful recovery.

Dashboard shows **Recovery complete** when `state.recovered=true`.


Dashboard подключается к WebSocket автоматически и пытается переподключиться каждые 3 секунды при обрыве.

Equity Curve на Dashboard обновляется каждые 5 секунд и поддерживает фильтры Day/Week/Month/All.


## Direction Engine

Добавлен модуль `app/direction_engine.py` с `DirectionEngine`, который определяет направление входа на основе:
- spread/deviation фильтров;
- funding-rate фильтра;
- confidence score (0..100) по spread/liquidity/deviation;
- опциональной коррекции плеча при повышенной 1m волатильности (`adjusted_leverage`).

`TradeEngine` теперь использует `DirectionEngine` перед входом: если направление `None`, сигнал помечается `filtered`; если рассчитано `adjusted_leverage`, оно используется для расчёта qty и открытия сделки.


## Strategy mode and effective settings

Added strategy profiles with runtime resolver:

- `strategy.mode`: `conservative | aggressive`
- each profile defines: `min_confidence`, `tp/sl`, `cooldown`, `max_positions`, `entry_type`, `entry_timeout_sec`, `retry_mode`, `leverage_mode`, `max_leverage`, and profile `filters`.

`ConfigResolver.get_effective_settings()` builds effective runtime settings from selected strategy profile. TradeEngine now uses only effective settings for:
- `tp/sl`
- cooldown / max_positions
- entry type / timeout / retry mode
- filters (`min_spread`, `min_liquidity`, `max_align`)
- leverage control (`leverage_mode`, `max_leverage`)

Adaptive leverage in TradeEngine:
- base = `trading.leverage`
- if `spread > 2 * min_spread` => `*1.3`
- if `liquidity < 2 * min_liquidity` => `*0.7`
- if high volatility => `*0.5`
- clamp to `[1, max_leverage]`

Web `/settings` now includes:
- `strategy_mode` selector
- full editable fields for both `conservative` and `aggressive` profiles
- effective summary block (computed settings) applied without restart.


## Strategy behavior differences

`StrategyController` adds behavioral differences between profiles:

- **Conservative**
  - strict pre-entry guard: spread/liquidity/align-age and confidence
  - limit price improvement (`-0.05%` for long, `+0.05%` for short)
  - timeout handling with `cancel_then_relimit` (one retry)
  - loss streak guard: 3 consecutive losing closed trades (24h window) pause new entries for 30 minutes

- **Aggressive**
  - allows market-style fast entries (per effective profile)
  - `fast_exit`: after 45s, if pnl < `+0.05%`, force market reduce-only close with `close_reason=time_exit`
  - panic on API errors: 5 order-related errors in 2 minutes pause new entries for 10 minutes

Dashboard now shows active safety locks and strategy mode.


## Strategy auto-revert

In `/settings` strategy switching is managed only via fields:
- `strategy_mode` (`conservative`/`aggressive`)
- `auto_revert_enabled`
- `auto_revert_minutes` (1..240)

When switching to `aggressive` with auto-revert enabled, app stores revert target in state and a background async task checks every 5 seconds. On expiry it switches strategy back (usually to `conservative`), clears revert state, updates runtime config immediately, and sends Telegram Bot API notification if `telegram_notifications.enabled=true` in runtime config.

Dashboard shows current strategy mode and countdown `Auto revert in: MM:SS` when timer is active.
