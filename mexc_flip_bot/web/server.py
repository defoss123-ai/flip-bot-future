"""Web control panel for bot runtime state, settings and analytics."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import yaml
from fastapi import FastAPI, Form, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import AppConfig, load_config
from app.config_resolver import ConfigResolver
from app.database import Database
from app.health import HealthMonitor
from app.mexc_client import MexcFuturesClient
from app.risk_manager import RiskManager
from app.state import load_state, save_state
from app.trade_engine import TradeEngine


ROOT_DIR = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
RUNTIME_CONFIG_PATH = ROOT_DIR / "data" / "config_runtime.yml"



def _profile_summary_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    strategy = runtime.get("strategy") or {}
    mode = str(strategy.get("mode") or "conservative")
    selected = strategy.get(mode) or {}
    trading = runtime.get("trading") or {}
    risk = runtime.get("risk") or {}
    return {
        "entry_type": selected.get("entry_type", trading.get("default_entry_type", "market")),
        "leverage": trading.get("leverage", 1),
        "usdt_amount": trading.get("usdt_amount", 0.0),
        "tp_percent": selected.get("tp_percent", risk.get("tp_percent", 0.0)),
        "sl_percent": selected.get("sl_percent", risk.get("sl_percent", 0.0)),
        "mode": mode,
    }


def _merge_runtime_with_defaults(data: dict[str, Any]) -> dict[str, Any]:
    default = _default_runtime_config()
    merged: dict[str, Any] = {**default, **(data or {})}
    for nested in ["trading", "orders", "risk", "filters"]:
        merged[nested] = {**default[nested], **((data.get(nested) if isinstance(data, dict) else {}) or {})}

    strategy_data = (data.get("strategy") if isinstance(data, dict) else {}) or {}
    merged["strategy"] = {**default["strategy"], **strategy_data}
    for profile in ["conservative", "aggressive"]:
        prof_default = default["strategy"][profile]
        prof_data = strategy_data.get(profile) or {}
        merged_profile = {**prof_default, **prof_data}
        merged_profile["filters"] = {**prof_default["filters"], **((prof_data.get("filters") or {}))}
        merged["strategy"][profile] = merged_profile
    return merged


def _runtime_to_json(runtime: dict[str, Any]) -> str:
    return json.dumps(runtime, ensure_ascii=False)


def _runtime_from_json(data_json: str) -> dict[str, Any]:
    try:
        loaded = json.loads(data_json or "{}")
    except Exception:
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    return _merge_runtime_with_defaults(loaded)


def _profile_record_to_view(profile) -> dict[str, Any]:
    runtime = _runtime_from_json(profile.data_json)
    summary = _profile_summary_runtime(runtime)
    return {
        "id": profile.id,
        "name": profile.name,
        "updated_at": profile.updated_at.isoformat(),
        "runtime": runtime,
        "mode": summary["mode"],
        "entry_type": summary["entry_type"],
        "leverage": summary["leverage"],
        "usdt_amount": summary["usdt_amount"],
        "tp_percent": summary["tp_percent"],
        "sl_percent": summary["sl_percent"],
        "is_active": bool(profile.is_active),
    }


async def _ensure_default_profile() -> None:
    profiles = await database.list_profiles()
    if profiles:
        active = await database.get_active_profile()
        if active is None:
            await database.set_active_profile(profiles[0].id)
        return
    runtime = _merge_runtime_with_defaults(_default_runtime_config())
    profile_id = await database.create_profile("Default", _runtime_to_json(runtime))
    await database.set_active_profile(profile_id)


async def _active_profile_runtime() -> tuple[Any, dict[str, Any]]:
    profile = await database.get_active_profile()
    if profile is None:
        await _ensure_default_profile()
        profile = await database.get_active_profile()
    if profile is None:
        runtime = _merge_runtime_with_defaults(_default_runtime_config())
        return None, runtime
    runtime = _runtime_from_json(profile.data_json)
    return profile, runtime


def _base_config() -> AppConfig:
    config_path = ROOT_DIR / "config.yml"
    if config_path.exists():
        return load_config(config_path)
    return load_config(ROOT_DIR / "config.yml.example")


def _default_runtime_config() -> dict[str, Any]:
    base = _base_config()
    return {
        "running": False,
        "mode": base.mode,
        "trading": {
            "usdt_amount": base.trading.usdt_amount,
            "leverage": base.trading.leverage,
            "max_positions": base.trading.max_positions,
            "cooldown_sec": base.trading.cooldown_sec,
            "default_entry_type": base.trading.default_entry_type,
            "leverage_mode": base.trading.leverage_mode,
            "max_leverage": base.trading.max_leverage,
        },
        "orders": {
            "entry_timeout_sec": base.orders.entry_timeout_sec,
            "retry_mode": base.orders.retry_mode,
        },
        "risk": {
            "tp_percent": base.risk.tp_percent,
            "sl_percent": base.risk.sl_percent,
            "max_total_exposure_usdt": base.risk.max_total_exposure_usdt,
            "max_funding_rate_percent": base.risk.max_funding_rate_percent,
            "spread_anomaly_multiplier": base.risk.spread_anomaly_multiplier,
        },
        "filters": {
            "min_spread_percent": base.filters.min_spread_percent,
            "min_liquidity_usd": base.filters.min_liquidity_usd,
            "max_align_age_sec": base.filters.max_align_age_sec,
        },
        "strategy": {
            "mode": base.strategy.mode,
            "auto_revert_enabled": False,
            "auto_revert_minutes": 30,
            "conservative": {
                "min_confidence": base.strategy.conservative.min_confidence,
                "tp_percent": base.strategy.conservative.tp_percent,
                "sl_percent": base.strategy.conservative.sl_percent,
                "cooldown_sec": base.strategy.conservative.cooldown_sec,
                "max_positions": base.strategy.conservative.max_positions,
                "entry_type": base.strategy.conservative.entry_type,
                "entry_timeout_sec": base.strategy.conservative.entry_timeout_sec,
                "retry_mode": base.strategy.conservative.retry_mode,
                "leverage_mode": base.strategy.conservative.leverage_mode,
                "max_leverage": base.strategy.conservative.max_leverage,
                "filters": {
                    "min_spread_percent": base.strategy.conservative.filters.min_spread_percent,
                    "min_liquidity_usd": base.strategy.conservative.filters.min_liquidity_usd,
                    "max_align_age_sec": base.strategy.conservative.filters.max_align_age_sec,
                },
            },
            "aggressive": {
                "min_confidence": base.strategy.aggressive.min_confidence,
                "tp_percent": base.strategy.aggressive.tp_percent,
                "sl_percent": base.strategy.aggressive.sl_percent,
                "cooldown_sec": base.strategy.aggressive.cooldown_sec,
                "max_positions": base.strategy.aggressive.max_positions,
                "entry_type": base.strategy.aggressive.entry_type,
                "entry_timeout_sec": base.strategy.aggressive.entry_timeout_sec,
                "retry_mode": base.strategy.aggressive.retry_mode,
                "leverage_mode": base.strategy.aggressive.leverage_mode,
                "max_leverage": base.strategy.aggressive.max_leverage,
                "filters": {
                    "min_spread_percent": base.strategy.aggressive.filters.min_spread_percent,
                    "min_liquidity_usd": base.strategy.aggressive.filters.min_liquidity_usd,
                    "max_align_age_sec": base.strategy.aggressive.filters.max_align_age_sec,
                },
            },
        },
    }


def _default_runtime_config_from_example() -> dict[str, Any]:
    """Load defaults strictly from config.yml.example when possible."""
    example_path = ROOT_DIR / "config.yml.example"
    if example_path.exists():
        base = load_config(example_path)
        return {
            "running": False,
            "mode": base.mode,
            "trading": {
                "usdt_amount": base.trading.usdt_amount,
                "leverage": base.trading.leverage,
                "max_positions": base.trading.max_positions,
                "cooldown_sec": base.trading.cooldown_sec,
                "default_entry_type": base.trading.default_entry_type,
                "leverage_mode": base.trading.leverage_mode,
                "max_leverage": base.trading.max_leverage,
            },
            "orders": {
                "entry_timeout_sec": base.orders.entry_timeout_sec,
                "retry_mode": base.orders.retry_mode,
            },
            "risk": {
                "tp_percent": base.risk.tp_percent,
                "sl_percent": base.risk.sl_percent,
                "max_total_exposure_usdt": base.risk.max_total_exposure_usdt,
                "max_funding_rate_percent": base.risk.max_funding_rate_percent,
                "spread_anomaly_multiplier": base.risk.spread_anomaly_multiplier,
            },
            "filters": {
                "min_spread_percent": base.filters.min_spread_percent,
                "min_liquidity_usd": base.filters.min_liquidity_usd,
                "max_align_age_sec": base.filters.max_align_age_sec,
            },
            "strategy": {
                "mode": base.strategy.mode,
                "auto_revert_enabled": False,
                "auto_revert_minutes": 30,
                "conservative": {
                    "min_confidence": base.strategy.conservative.min_confidence,
                    "tp_percent": base.strategy.conservative.tp_percent,
                    "sl_percent": base.strategy.conservative.sl_percent,
                    "cooldown_sec": base.strategy.conservative.cooldown_sec,
                    "max_positions": base.strategy.conservative.max_positions,
                    "entry_type": base.strategy.conservative.entry_type,
                    "entry_timeout_sec": base.strategy.conservative.entry_timeout_sec,
                    "retry_mode": base.strategy.conservative.retry_mode,
                    "leverage_mode": base.strategy.conservative.leverage_mode,
                    "max_leverage": base.strategy.conservative.max_leverage,
                    "filters": {
                        "min_spread_percent": base.strategy.conservative.filters.min_spread_percent,
                        "min_liquidity_usd": base.strategy.conservative.filters.min_liquidity_usd,
                        "max_align_age_sec": base.strategy.conservative.filters.max_align_age_sec,
                    },
                },
                "aggressive": {
                    "min_confidence": base.strategy.aggressive.min_confidence,
                    "tp_percent": base.strategy.aggressive.tp_percent,
                    "sl_percent": base.strategy.aggressive.sl_percent,
                    "cooldown_sec": base.strategy.aggressive.cooldown_sec,
                    "max_positions": base.strategy.aggressive.max_positions,
                    "entry_type": base.strategy.aggressive.entry_type,
                    "entry_timeout_sec": base.strategy.aggressive.entry_timeout_sec,
                    "retry_mode": base.strategy.aggressive.retry_mode,
                    "leverage_mode": base.strategy.aggressive.leverage_mode,
                    "max_leverage": base.strategy.aggressive.max_leverage,
                    "filters": {
                        "min_spread_percent": base.strategy.aggressive.filters.min_spread_percent,
                        "min_liquidity_usd": base.strategy.aggressive.filters.min_liquidity_usd,
                        "max_align_age_sec": base.strategy.aggressive.filters.max_align_age_sec,
                    },
                },
            },
        }
    return _default_runtime_config()


def _load_runtime_config() -> dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        default = _default_runtime_config()
        _save_runtime_config(default)
        return default

    data = yaml.safe_load(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    default = _default_runtime_config()

    for key in ["running", "mode", "trading", "orders", "risk", "filters", "strategy"]:
        if key not in data:
            data[key] = default[key]

    for nested in ["trading", "orders", "risk", "filters"]:
        merged = {**default[nested], **(data.get(nested) or {})}
        data[nested] = merged

    data["strategy"] = {**default["strategy"], **(data.get("strategy") or {})}
    for profile in ["conservative", "aggressive"]:
        prof_default = default["strategy"][profile]
        prof_data = data["strategy"].get(profile) or {}
        merged_profile = {**prof_default, **prof_data}
        merged_profile["filters"] = {**prof_default["filters"], **((prof_data.get("filters") or {}))}
        data["strategy"][profile] = merged_profile

    return data


def _save_runtime_config(config_data: dict[str, Any]) -> None:
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG_PATH.write_text(yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _apply_runtime_to_base(base: AppConfig, runtime: dict[str, Any]) -> AppConfig:
    payload = base.model_dump()
    payload["mode"] = runtime.get("mode", base.mode)

    for nested in ["trading", "orders", "risk", "filters", "strategy"]:
        payload[nested].update(runtime.get(nested) or {})

    mexc_runtime = runtime.get("mexc") or {}
    if mexc_runtime:
        payload["mexc"].update(mexc_runtime)

    return AppConfig.model_validate(payload)


def _redirect_with_msg(path: str, msg: str) -> RedirectResponse:
    return RedirectResponse(url=f"{path}?msg={quote_plus(msg)}", status_code=303)


def _validate_settings(
    default_entry_type: str,
    entry_timeout_sec: int,
    retry_mode: str,
    leverage: int,
    usdt_amount: float,
    max_total_exposure_usdt: float,
    max_funding_rate_percent: float,
    spread_anomaly_multiplier: float,
) -> str | None:
    if default_entry_type not in {"market", "limit"}:
        return "Invalid entry type"
    if retry_mode not in {"cancel", "cancel_then_market", "cancel_then_relimit"}:
        return "Invalid retry mode"
    if leverage < 1 or leverage > 200:
        return "Leverage must be in range 1..200"
    if usdt_amount <= 0:
        return "USDT amount must be > 0"
    if entry_timeout_sec < 5 or entry_timeout_sec > 600:
        return "Entry timeout must be in range 5..600"
    if default_entry_type == "limit" and entry_timeout_sec <= 0:
        return "For limit entry timeout must be > 0"
    if max_total_exposure_usdt <= 0:
        return "max_total_exposure_usdt must be > 0"
    if max_funding_rate_percent < 0:
        return "max_funding_rate_percent must be >= 0"
    if spread_anomaly_multiplier <= 0:
        return "spread_anomaly_multiplier must be > 0"
    return None


LOGGER = logging.getLogger(__name__)


class _UILogger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, data: dict[str, Any]) -> None:
        encoded = json.dumps(data)
        async with self._lock:
            targets = list(self.active_connections)

        dead: list[WebSocket] = []
        for connection in targets:
            try:
                await connection.send_text(encoded)
            except Exception:
                dead.append(connection)

        if dead:
            async with self._lock:
                for connection in dead:
                    self.active_connections.discard(connection)


app = FastAPI(title="mexc_flip_bot web")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
database = Database(ROOT_DIR / "data" / "bot.db")
BASE_CONFIG = _base_config()
RUNTIME_CONFIG = _load_runtime_config()
APP_CONFIG = _apply_runtime_to_base(BASE_CONFIG, RUNTIME_CONFIG)
mexc_client = MexcFuturesClient(
    database=database,
    mode=APP_CONFIG.mode,
    api_key=APP_CONFIG.mexc.api_key,
    api_secret=APP_CONFIG.mexc.api_secret,
)

risk_manager = RiskManager(config=APP_CONFIG, mexc_client=mexc_client, logger=_UILogger())
health_monitor = HealthMonitor(database=database, mexc_client=mexc_client, logger=_UILogger())
trade_engine = TradeEngine(
    config=APP_CONFIG,
    database=database,
    mexc_client=mexc_client,
    logger=_UILogger(),
    is_running=lambda: load_state(ROOT_DIR).running,
    risk_manager=risk_manager,
    allow_new_entries=health_monitor.allow_new_entries,
)

ws_manager = ConnectionManager()
ws_broadcast_task: asyncio.Task[None] | None = None
auto_revert_task: asyncio.Task[None] | None = None


def _refresh_in_memory_runtime(runtime: dict[str, Any]) -> AppConfig:
    global RUNTIME_CONFIG, APP_CONFIG
    RUNTIME_CONFIG = runtime
    APP_CONFIG = _apply_runtime_to_base(BASE_CONFIG, runtime)
    trade_engine.config = APP_CONFIG
    trade_engine.risk_manager.config = APP_CONFIG
    trade_engine.config_resolver.config = APP_CONFIG
    health_monitor.set_telegram_ok(bool(APP_CONFIG.telegram.use_telethon))
    return APP_CONFIG


async def _build_dashboard_payload() -> dict[str, Any]:
    stats = await database.stats()
    await health_monitor.check_once()
    positions = await mexc_client.get_open_positions()

    total_exposure = 0.0
    for position in positions:
        size = abs(float(position.get("size") or position.get("signed_size") or 0.0))
        entry_price = float(position.get("entry_price") or 0.0)
        total_exposure += size * entry_price

    return {
        "pnl_day": stats.pnl_day,
        "pnl_week": stats.pnl_week,
        "pnl_month": stats.pnl_month,
        "total_trades": stats.total_trades,
        "winrate": stats.winrate,
        "open_positions": len(positions),
        "total_exposure": round(total_exposure, 6),
        "system_health": health_monitor.snapshot(),
    }




def _since_for_range(period: str) -> datetime | None:
    now = datetime.now(UTC)
    if period == "day":
        return now - timedelta(days=1)
    if period == "week":
        return now - timedelta(days=7)
    if period == "month":
        return now - timedelta(days=30)
    return None


async def _build_equity_points(period: str = "all") -> list[dict[str, float | str]]:
    since = _since_for_range(period)
    trades = await database.list_closed_trades(since=since)
    if not trades:
        ts = datetime.now(UTC).isoformat()
        return [{"timestamp": ts, "equity": 0.0}]

    cumulative = 0.0
    points: list[dict[str, float | str]] = []
    for trade in trades:
        cumulative += float(trade.pnl_usdt or 0.0)
        points.append({"timestamp": trade.created_at.isoformat(), "equity": round(cumulative, 6)})
    return points


async def _ws_dashboard_broadcaster() -> None:
    while True:
        try:
            payload = await _build_dashboard_payload()
            await ws_manager.broadcast(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("WS dashboard broadcast error: %s", exc)
        await asyncio.sleep(2)




def _revert_countdown(state) -> str | None:
    if state.revert_at is None:
        return None
    now = datetime.now(UTC)
    if state.revert_at <= now:
        return "00:00"
    left = int((state.revert_at - now).total_seconds())
    mm = left // 60
    ss = left % 60
    return f"{mm:02d}:{ss:02d}"


async def _send_telegram_notification(text: str) -> None:
    runtime = _load_runtime_config()
    tg = runtime.get("telegram_notifications") or {}
    if not tg.get("enabled"):
        return
    token = str(tg.get("bot_token") or "").strip()
    chat_id = str(tg.get("chat_id") or "").strip()
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = Request(url=url, data=payload, method="POST", headers={"Content-Type": "application/json"})
        await asyncio.to_thread(urlopen, req, 8)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Telegram notify failed: %s", exc)


async def _auto_revert_loop() -> None:
    while True:
        try:
            active_profile, runtime = await _active_profile_runtime()
            state = load_state(ROOT_DIR)
            revert_at = state.revert_at
            revert_to = state.revert_to
            enabled = bool((runtime.get("strategy") or {}).get("auto_revert_enabled", False))
            if enabled and revert_at is not None and revert_to:
                if datetime.now(UTC) >= revert_at:
                    runtime.setdefault("strategy", {})
                    runtime["strategy"]["mode"] = revert_to
                    runtime["strategy"]["auto_revert_enabled"] = False
                    _save_runtime_config(runtime)
                    _refresh_in_memory_runtime(runtime)
                    if active_profile is not None:
                        await database.update_profile(active_profile.id, _runtime_to_json(runtime))
                    save_state(running=bool(runtime.get("running", False)), root_dir=ROOT_DIR, active_profile_id=(active_profile.id if active_profile else 0), revert_at="", revert_to="")
                    await _send_telegram_notification(f"Auto revert: strategy switched back to {revert_to}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Auto revert loop error: %s", exc)
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event() -> None:
    global ws_broadcast_task, auto_revert_task
    await database.init_db()
    await _ensure_default_profile()
    _active_profile, runtime = await _active_profile_runtime()
    _save_runtime_config(runtime)
    _refresh_in_memory_runtime(runtime)
    save_state(bool(runtime.get("running", False)), ROOT_DIR, active_profile_id=(_active_profile.id if _active_profile else 0))
    health_monitor.set_telegram_ok(bool(APP_CONFIG.telegram.use_telethon))
    await health_monitor.check_once()
    ws_broadcast_task = asyncio.create_task(_ws_dashboard_broadcaster(), name="ws-dashboard-broadcast")
    auto_revert_task = asyncio.create_task(_auto_revert_loop(), name="strategy-auto-revert")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global ws_broadcast_task, auto_revert_task
    if ws_broadcast_task is not None:
        ws_broadcast_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ws_broadcast_task
        ws_broadcast_task = None
    if auto_revert_task is not None:
        auto_revert_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await auto_revert_task
        auto_revert_task = None


@app.get("/")
async def dashboard(request: Request):
    state = load_state(ROOT_DIR)
    revert_countdown = _revert_countdown(state)
    active_profile, active_runtime = await _active_profile_runtime()
    active_summary = _profile_summary_runtime(active_runtime)
    stats = await database.stats()
    signals = await database.list_recent_signals(limit=10)
    await health_monitor.check_once()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "runtime": RUNTIME_CONFIG,
            "app_config": APP_CONFIG,
            "state": state,
            "stats": stats,
            "signals": signals,
            "msg": request.query_params.get("msg", ""),
            "health": health_monitor.snapshot(),
            "open_positions": len(await mexc_client.get_open_positions()),
            "total_exposure": (await _build_dashboard_payload())["total_exposure"],
            "effective": ConfigResolver(APP_CONFIG).get_effective_settings(),
            "safety": trade_engine.get_safety_status(),
            "revert_countdown": revert_countdown,
            "active_profile_name": (active_profile.name if active_profile else "N/A"),
            "active_profile_summary": active_summary,
        },
    )


@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json(await _build_dashboard_payload())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("WS client error: %s", exc)
        await ws_manager.disconnect(websocket)




@app.get("/api/equity")
async def api_equity(range: str = Query(default="all", pattern="^(day|week|month|all)$")):
    try:
        points = await _build_equity_points(range)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Equity endpoint error: %s", exc)
        return JSONResponse(status_code=500, content={"error": "equity_failed"})
    return points

@app.get("/settings")
async def settings_page(request: Request, profile_id: int | None = Query(default=None)):
    _active, runtime_view = await _active_profile_runtime()
    if profile_id is not None:
        selected = await database.get_profile(profile_id)
        if selected is not None:
            runtime_view = _runtime_from_json(selected.data_json)

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "runtime": runtime_view,
            "entry_types": ["market", "limit"],
            "retry_modes": ["cancel", "cancel_then_market", "cancel_then_relimit"],
            "strategy_modes": ["conservative", "aggressive"],
            "leverage_modes": ["fixed", "adaptive"],
            "modes": ["paper", "live"],
            "msg": request.query_params.get("msg", ""),
            "health": health_monitor.snapshot(),
            "effective": ConfigResolver(APP_CONFIG).get_effective_settings(),
        },
    )


@app.post("/settings")
async def save_settings(
    mode: str = Form(...),
    strategy_mode: str = Form(...),
    auto_revert_enabled: bool = Form(False),
    auto_revert_minutes: int = Form(30),
    usdt_amount: float = Form(...),
    leverage: int = Form(...),
    max_positions: int = Form(...),
    cooldown_sec: int = Form(...),
    default_entry_type: str = Form(...),
    leverage_mode: str = Form(...),
    max_leverage: int = Form(...),
    min_spread_percent: float = Form(...),
    min_liquidity_usd: float = Form(...),
    max_align_age_sec: int = Form(...),
    tp_percent: float = Form(...),
    sl_percent: float = Form(...),
    max_total_exposure_usdt: float = Form(...),
    max_funding_rate_percent: float = Form(...),
    spread_anomaly_multiplier: float = Form(...),
    entry_timeout_sec: int = Form(...),
    retry_mode: str = Form(...),
    conservative_min_confidence: int = Form(...),
    conservative_tp_percent: float = Form(...),
    conservative_sl_percent: float = Form(...),
    conservative_cooldown_sec: int = Form(...),
    conservative_max_positions: int = Form(...),
    conservative_entry_type: str = Form(...),
    conservative_entry_timeout_sec: int = Form(...),
    conservative_retry_mode: str = Form(...),
    conservative_leverage_mode: str = Form(...),
    conservative_max_leverage: int = Form(...),
    conservative_min_spread_percent: float = Form(...),
    conservative_min_liquidity_usd: float = Form(...),
    conservative_max_align_age_sec: int = Form(...),
    aggressive_min_confidence: int = Form(...),
    aggressive_tp_percent: float = Form(...),
    aggressive_sl_percent: float = Form(...),
    aggressive_cooldown_sec: int = Form(...),
    aggressive_max_positions: int = Form(...),
    aggressive_entry_type: str = Form(...),
    aggressive_entry_timeout_sec: int = Form(...),
    aggressive_retry_mode: str = Form(...),
    aggressive_leverage_mode: str = Form(...),
    aggressive_max_leverage: int = Form(...),
    aggressive_min_spread_percent: float = Form(...),
    aggressive_min_liquidity_usd: float = Form(...),
    aggressive_max_align_age_sec: int = Form(...),
):
    try:
        err = _validate_settings(
            default_entry_type,
            entry_timeout_sec,
            retry_mode,
            leverage,
            usdt_amount,
            max_total_exposure_usdt,
            max_funding_rate_percent,
            spread_anomaly_multiplier,
        )
        if err:
            return _redirect_with_msg("/settings", err)

        if strategy_mode not in {"conservative", "aggressive"}:
            return _redirect_with_msg("/settings", "Invalid strategy_mode")
        if leverage_mode not in {"fixed", "adaptive"}:
            return _redirect_with_msg("/settings", "Invalid leverage_mode")
        if auto_revert_minutes < 1 or auto_revert_minutes > 240:
            return _redirect_with_msg("/settings", "auto_revert_minutes must be 1..240")

        _active, runtime = await _active_profile_runtime()
        runtime["mode"] = mode
        runtime["trading"].update(
            {
                "usdt_amount": usdt_amount,
                "leverage": leverage,
                "max_positions": max_positions,
                "cooldown_sec": cooldown_sec,
                "default_entry_type": default_entry_type,
                "leverage_mode": leverage_mode,
                "max_leverage": max_leverage,
            }
        )
        runtime["filters"].update(
            {
                "min_spread_percent": min_spread_percent,
                "min_liquidity_usd": min_liquidity_usd,
                "max_align_age_sec": max_align_age_sec,
            }
        )
        runtime["risk"].update(
            {
                "tp_percent": tp_percent,
                "sl_percent": sl_percent,
                "max_total_exposure_usdt": max_total_exposure_usdt,
                "max_funding_rate_percent": max_funding_rate_percent,
                "spread_anomaly_multiplier": spread_anomaly_multiplier,
            }
        )
        runtime["orders"].update({"entry_timeout_sec": entry_timeout_sec, "retry_mode": retry_mode})

        runtime.setdefault("strategy", {})
        runtime["strategy"]["mode"] = strategy_mode
        runtime["strategy"]["auto_revert_enabled"] = bool(auto_revert_enabled)
        runtime["strategy"]["auto_revert_minutes"] = int(auto_revert_minutes)
        runtime["strategy"]["conservative"] = {
            "min_confidence": conservative_min_confidence,
            "tp_percent": conservative_tp_percent,
            "sl_percent": conservative_sl_percent,
            "cooldown_sec": conservative_cooldown_sec,
            "max_positions": conservative_max_positions,
            "entry_type": conservative_entry_type,
            "entry_timeout_sec": conservative_entry_timeout_sec,
            "retry_mode": conservative_retry_mode,
            "leverage_mode": conservative_leverage_mode,
            "max_leverage": conservative_max_leverage,
            "filters": {
                "min_spread_percent": conservative_min_spread_percent,
                "min_liquidity_usd": conservative_min_liquidity_usd,
                "max_align_age_sec": conservative_max_align_age_sec,
            },
        }
        runtime["strategy"]["aggressive"] = {
            "min_confidence": aggressive_min_confidence,
            "tp_percent": aggressive_tp_percent,
            "sl_percent": aggressive_sl_percent,
            "cooldown_sec": aggressive_cooldown_sec,
            "max_positions": aggressive_max_positions,
            "entry_type": aggressive_entry_type,
            "entry_timeout_sec": aggressive_entry_timeout_sec,
            "retry_mode": aggressive_retry_mode,
            "leverage_mode": aggressive_leverage_mode,
            "max_leverage": aggressive_max_leverage,
            "filters": {
                "min_spread_percent": aggressive_min_spread_percent,
                "min_liquidity_usd": aggressive_min_liquidity_usd,
                "max_align_age_sec": aggressive_max_align_age_sec,
            },
        }

        _save_runtime_config(runtime)
        _refresh_in_memory_runtime(runtime)

        if strategy_mode == "aggressive" and auto_revert_enabled:
            revert_at = datetime.now(UTC) + timedelta(minutes=int(auto_revert_minutes))
            active_profile = await database.get_active_profile()
            save_state(bool(runtime.get("running", False)), ROOT_DIR, active_profile_id=(active_profile.id if active_profile else 0), revert_at=revert_at, revert_to="conservative")
        else:
            # Any manual strategy selection or disabling auto-revert should clear stale timers.
            active_profile = await database.get_active_profile()
            save_state(bool(runtime.get("running", False)), ROOT_DIR, active_profile_id=(active_profile.id if active_profile else 0), revert_at="", revert_to="")

        active_profile = await database.get_active_profile()
        if active_profile is not None:
            await database.update_profile(active_profile.id, _runtime_to_json(runtime))

        return _redirect_with_msg("/settings", "Saved")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Save settings failed: %s", exc)
        return _redirect_with_msg("/settings", f"Save failed: {exc}")


@app.post("/settings/reset")
async def reset_settings():
    try:
        runtime = _default_runtime_config_from_example()
        _save_runtime_config(runtime)
        _refresh_in_memory_runtime(runtime)
        active_profile = await database.get_active_profile()
        save_state(bool(runtime.get("running", False)), ROOT_DIR, active_profile_id=(active_profile.id if active_profile else 0), revert_at="", revert_to="")

        active_profile = await database.get_active_profile()
        if active_profile is not None:
            await database.update_profile(active_profile.id, _runtime_to_json(runtime))

        return _redirect_with_msg("/settings", "Reset to defaults")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Reset settings failed: %s", exc)
        return _redirect_with_msg("/settings", f"Reset failed: {exc}")


@app.get("/profiles")
async def profiles_page(request: Request):
    profiles = await database.list_profiles()
    rows = [_profile_record_to_view(profile) for profile in profiles]
    active_profile = next((row for row in rows if row["is_active"]), None)
    return templates.TemplateResponse(
        "profiles.html",
        {
            "request": request,
            "profiles": rows,
            "active_profile": active_profile,
            "msg": request.query_params.get("msg", ""),
        },
    )


@app.post("/profiles/create")
async def profiles_create(name: str = Form(...)):
    try:
        trimmed = name.strip()
        if not trimmed:
            return _redirect_with_msg("/profiles", "Name is required")
        _active, active_runtime = await _active_profile_runtime()
        await database.create_profile(trimmed, _runtime_to_json(active_runtime))
        return _redirect_with_msg("/profiles", "Profile created")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Create profile failed: %s", exc)
        return _redirect_with_msg("/profiles", f"Create failed: {exc}")


@app.post("/profiles/activate/{profile_id}")
async def profiles_activate(profile_id: int):
    try:
        profile = await database.get_profile(profile_id)
        if profile is None:
            return _redirect_with_msg("/profiles", "Profile not found")
        ok = await database.set_active_profile(profile_id)
        if not ok:
            return _redirect_with_msg("/profiles", "Profile not found")
        runtime = _runtime_from_json(profile.data_json)
        runtime["running"] = bool(RUNTIME_CONFIG.get("running", False))
        _save_runtime_config(runtime)
        _refresh_in_memory_runtime(runtime)
        save_state(bool(runtime.get("running", False)), ROOT_DIR, active_profile_id=profile_id)
        await database.update_profile(profile_id, _runtime_to_json(runtime))
        return _redirect_with_msg("/profiles", "Profile activated")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Activate profile failed: %s", exc)
        return _redirect_with_msg("/profiles", f"Activate failed: {exc}")


@app.post("/profiles/duplicate/{profile_id}")
async def profiles_duplicate(profile_id: int):
    try:
        source = await database.get_profile(profile_id)
        if source is None:
            return _redirect_with_msg("/profiles", "Profile not found")
        await database.create_profile(f"{source.name} Copy", source.data_json)
        return _redirect_with_msg("/profiles", "Profile duplicated")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Duplicate profile failed: %s", exc)
        return _redirect_with_msg("/profiles", f"Duplicate failed: {exc}")


@app.post("/profiles/delete/{profile_id}")
async def profiles_delete(profile_id: int):
    try:
        profile = await database.get_profile(profile_id)
        if profile is None:
            return _redirect_with_msg("/profiles", "Profile not found")
        active = await database.get_active_profile()
        await database.delete_profile(profile_id)
        profiles = await database.list_profiles()
        if not profiles:
            await _ensure_default_profile()
            _active, runtime = await _active_profile_runtime()
            _save_runtime_config(runtime)
            _refresh_in_memory_runtime(runtime)
            save_state(bool(runtime.get("running", False)), ROOT_DIR, active_profile_id=profile_id)
            return _redirect_with_msg("/profiles", "Profile deleted")
        if active is not None and active.id == profile_id:
            await database.set_active_profile(profiles[0].id)
            selected = await database.get_active_profile()
            if selected is not None:
                runtime = _runtime_from_json(selected.data_json)
                runtime["running"] = bool(RUNTIME_CONFIG.get("running", False))
                _save_runtime_config(runtime)
                _refresh_in_memory_runtime(runtime)
                save_state(bool(runtime.get("running", False)), ROOT_DIR, active_profile_id=selected.id)
                await database.update_profile(selected.id, _runtime_to_json(runtime))
        return _redirect_with_msg("/profiles", "Profile deleted")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Delete profile failed: %s", exc)
        return _redirect_with_msg("/profiles", f"Delete failed: {exc}")


@app.get("/trades")
async def trades_page(request: Request):
    trades = await database.list_recent_trades(limit=200)
    return templates.TemplateResponse(
        "trades.html",
        {
            "request": request,
            "trades": trades,
            "msg": request.query_params.get("msg", ""),
        },
    )


@app.post("/trades/{trade_id}/close")
async def close_trade_now(trade_id: int):
    try:
        ok, msg = await trade_engine.close_trade(trade_id)
    except Exception as exc:  # noqa: BLE001
        return _redirect_with_msg("/trades", f"Close failed: {exc}")

    if ok:
        return _redirect_with_msg("/trades", f"Trade {trade_id} closed")
    return _redirect_with_msg("/trades", f"Close skipped: {msg}")


@app.get("/orders")
async def orders_page(request: Request):
    orders = await database.list_recent_orders(limit=200)
    return templates.TemplateResponse(
        "orders.html",
        {
            "request": request,
            "orders": orders,
            "msg": request.query_params.get("msg", ""),
        },
    )


@app.post("/bot/start")
async def bot_start():
    runtime = _load_runtime_config()
    runtime["running"] = True
    _save_runtime_config(runtime)
    _refresh_in_memory_runtime(runtime)
    state = load_state(ROOT_DIR)
    save_state(True, ROOT_DIR, active_profile_id=state.active_profile_id)
    return _redirect_with_msg("/", "Bot started")


@app.post("/bot/stop")
async def bot_stop():
    runtime = _load_runtime_config()
    runtime["running"] = False
    _save_runtime_config(runtime)
    _refresh_in_memory_runtime(runtime)
    state = load_state(ROOT_DIR)
    save_state(False, ROOT_DIR, active_profile_id=state.active_profile_id)
    return _redirect_with_msg("/", "Bot stopped")


@app.post("/bot/panic")
async def bot_panic():
    runtime = _load_runtime_config()
    runtime["running"] = False
    _save_runtime_config(runtime)
    _refresh_in_memory_runtime(runtime)
    state = load_state(ROOT_DIR)
    save_state(False, ROOT_DIR, active_profile_id=state.active_profile_id)

    closed, errors = await trade_engine.close_all_positions(reason="panic")
    if errors > 0:
        return _redirect_with_msg("/", f"PANIC EXECUTED: closed={closed}, errors={errors}")
    return _redirect_with_msg("/", f"PANIC EXECUTED: closed={closed}")


@app.post("/bot/paper")
async def bot_paper():
    runtime = _load_runtime_config()
    runtime["mode"] = "paper"
    _save_runtime_config(runtime)
    _refresh_in_memory_runtime(runtime)
    return _redirect_with_msg("/", "Mode set to paper")


@app.post("/bot/live")
async def bot_live():
    runtime = _load_runtime_config()
    runtime["mode"] = "live"
    _save_runtime_config(runtime)
    _refresh_in_memory_runtime(runtime)
    return _redirect_with_msg("/", "Mode set to live")
