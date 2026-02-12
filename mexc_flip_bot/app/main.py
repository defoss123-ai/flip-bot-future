"""Application entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import yaml

from app.config import AppConfig, load_config
from app.database import Database, TradeRecord
from app.health import HealthMonitor
from app.logger import setup_logger
from app.mexc_client import MexcFuturesClient
from app.position_monitor import PositionMonitor
from app.risk_manager import RiskManager
from app.signal_parser import SignalParser
from app.state import load_state, save_state
from app.telegram_listener import TelegramListener
from app.trade_engine import TradeEngine


async def _signal_worker(
    message_queue: asyncio.Queue[str],
    parser: SignalParser,
    database: Database,
    logger,
    mexc_client: MexcFuturesClient,
) -> None:
    while True:
        raw_text = await message_queue.get()
        parsed = parser.parse(raw_text)
        signal_hash = parser.compute_hash(raw_text)

        symbol = parsed.get("symbol")
        mexc_price = parsed.get("mexc_price")
        if symbol and mexc_price is not None:
            await mexc_client.apply_market_price(str(symbol), float(mexc_price))

        if await database.signal_hash_exists(signal_hash):
            logger.info("Skip duplicate signal hash={}", signal_hash[:12])
            continue

        passed = parser.passes_filters(parsed)
        status = "new" if passed else "filtered"

        signal_id = await database.add_signal(
            raw_text=raw_text,
            symbol=symbol,
            dex_price=parsed.get("dex_price"),
            mexc_price=mexc_price,
            spread_percent=parsed.get("spread_percent"),
            liquidity_usd=parsed.get("liquidity_usd"),
            align_age_sec=parsed.get("align_age_sec"),
            direction=parsed.get("direction"),
            status=status,
            hash=signal_hash,
        )

        if signal_id is None:
            logger.info("Skip duplicate signal (unique conflict) hash={}", signal_hash[:12])
            continue

        logger.info("Signal saved id={} status={} symbol={}", signal_id, status, symbol)


def _load_runtime_mode_and_keys(root_dir: Path, config: AppConfig) -> tuple[str, str, str]:
    runtime_path = root_dir / "data" / "config_runtime.yml"
    runtime = {}
    if runtime_path.exists():
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}

    mode = str(runtime.get("mode") or config.mode)
    mexc_cfg = runtime.get("mexc") or {}
    api_key = str(mexc_cfg.get("api_key") or config.mexc.api_key)
    api_secret = str(mexc_cfg.get("api_secret") or config.mexc.api_secret)
    return mode, api_key, api_secret


def _pnl_for_closed_without_position(trade: TradeRecord, exit_price: float | None) -> float | None:
    if (
        exit_price is None
        or exit_price <= 0
        or trade.entry_price is None
        or trade.entry_price <= 0
    ):
        return trade.pnl_usdt

    change = (exit_price - trade.entry_price) / trade.entry_price
    if trade.direction == "short":
        change = -change
    return trade.usdt_amount * trade.leverage * change


async def _recover_state(database: Database, mexc_client: MexcFuturesClient, config: AppConfig, logger) -> None:
    logger.info("Recovery: started")

    # 1) Sync open positions <-> open trades.
    positions = await mexc_client.get_open_positions()
    positions_by_symbol: dict[str, dict] = {}
    for position in positions:
        symbol = mexc_client.normalize_symbol(str(position.get("symbol") or ""))
        if not symbol:
            continue
        positions_by_symbol[symbol] = position

    open_trades = await database.list_open_trades()
    open_trades_by_symbol: dict[str, list[TradeRecord]] = {}
    for trade in open_trades:
        open_trades_by_symbol.setdefault(trade.symbol, []).append(trade)

    created = 0
    closed_missing = 0
    for symbol, position in positions_by_symbol.items():
        if symbol in open_trades_by_symbol and open_trades_by_symbol[symbol]:
            continue

        signed_size = float(position.get("signed_size") or 0.0)
        direction = "short" if signed_size < 0 else "long"
        entry_price = float(position.get("entry_price") or 0.0) or None
        size = abs(float(position.get("size") or 0.0))
        notional = size * (entry_price or 0.0)
        usdt_amount = max(notional / max(config.trading.leverage, 1), config.trading.usdt_amount)

        await database.add_trade(
            symbol=symbol,
            direction=direction,
            usdt_amount=usdt_amount,
            leverage=config.trading.leverage,
            entry_price=entry_price,
            status="open",
            close_reason=None,
        )
        created += 1

    for trade in open_trades:
        symbol = mexc_client.normalize_symbol(trade.symbol)
        if symbol in positions_by_symbol:
            continue

        exit_price = mexc_client.get_market_price(symbol)
        pnl = _pnl_for_closed_without_position(trade, exit_price)
        await database.update_trade(
            trade.id,
            status="closed",
            close_reason="unknown",
            exit_price=exit_price,
            pnl_usdt=pnl,
        )
        closed_missing += 1

    # 2) Sync hanging orders in `placed` status.
    updated_orders = 0
    placed_orders = await database.list_orders_by_status(status="placed", limit=1000)
    for order in placed_orders:
        if not order.exchange_order_id:
            continue
        try:
            order_info = await mexc_client.get_order(order.symbol, order.exchange_order_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Recovery: failed to check order db_id={} exchange_id={} err={}",
                order.id,
                order.exchange_order_id,
                exc,
            )
            continue

        status = str(order_info.get("status") or "placed")
        if status in {"canceled", "rejected", "filled", "error"}:
            await database.update_order_status(order.id, status=status)
            updated_orders += 1

    logger.info(
        "Recovery: completed created_open_trades={} closed_missing_positions={} updated_orders={}",
        created,
        closed_missing,
        updated_orders,
    )


async def run() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    logger = setup_logger(root_dir / "logs")

    config: AppConfig = load_config(root_dir / "config.yml")
    database = Database(root_dir / "data" / "bot.db")
    await database.init_db()

    parser = SignalParser(
        min_spread_percent=config.filters.min_spread_percent,
        min_liquidity_usd=config.filters.min_liquidity_usd,
        max_align_age_sec=config.filters.max_align_age_sec,
    )
    runtime_mode, api_key, api_secret = _load_runtime_mode_and_keys(root_dir, config)
    mexc_client = MexcFuturesClient(
        database=database,
        mode=runtime_mode,
        api_key=api_key,
        api_secret=api_secret,
        logger=logger,
    )
    health_monitor = HealthMonitor(database=database, mexc_client=mexc_client, logger=logger)
    message_queue: asyncio.Queue[str] = asyncio.Queue()

    risk_manager = RiskManager(config=config, mexc_client=mexc_client, logger=logger)

    trade_engine = TradeEngine(
        config=config,
        database=database,
        mexc_client=mexc_client,
        logger=logger,
        is_running=lambda: load_state(root_dir).running,
        risk_manager=risk_manager,
        allow_new_entries=health_monitor.allow_new_entries,
    )

    position_monitor = PositionMonitor(database=database, mexc_client=mexc_client, logger=logger)

    bot_state = load_state(root_dir)

    logger.info("Bot started")
    logger.info("Database initialized at {}", root_dir / "data" / "bot.db")
    logger.info(
        "Initialized components: parser={}, mexc_client={}, trade_engine={}, database={}, mode={}",
        parser.__class__.__name__,
        mexc_client.__class__.__name__,
        trade_engine.__class__.__name__,
        database.__class__.__name__,
        runtime_mode,
    )
    logger.info(
        "State: running={} recovered={} started_at={}",
        bot_state.running,
        bot_state.recovered,
        bot_state.started_at.isoformat(),
    )

    bot_state = save_state(root_dir=root_dir, recovered=False)
    try:
        await _recover_state(database, mexc_client, config, logger)
        bot_state = save_state(root_dir=root_dir, recovered=True)
        logger.info("Recovery complete: recovered={}", bot_state.recovered)
    except Exception as exc:  # noqa: BLE001
        bot_state = save_state(root_dir=root_dir, recovered=False)
        logger.error("Recovery failed: {}", exc)

    if not bot_state.running:
        logger.info("State running=false. Telegram listener/worker/engine are not started.")
        return

    worker_task = asyncio.create_task(_signal_worker(message_queue, parser, database, logger, mexc_client), name="signal-worker")
    engine_task = asyncio.create_task(trade_engine.run_loop(), name="trade-engine")
    position_task = asyncio.create_task(position_monitor.run_loop(), name="position-monitor")
    health_task = asyncio.create_task(health_monitor.run_loop(), name="health-monitor")

    tg_cfg = config.telegram
    listener = None
    listener_task = None
    if tg_cfg.use_telethon:
        listener = TelegramListener(
            api_id=tg_cfg.api_id,
            api_hash=tg_cfg.api_hash,
            session_name=tg_cfg.session_name,
            source_chat=tg_cfg.source_chat,
            message_queue=message_queue,
            logger=logger,
        )
        listener_task = asyncio.create_task(listener.start(), name="telegram-listener")
        logger.info("Telegram listener starting for source_chat={}", tg_cfg.source_chat)
    else:
        health_monitor.set_telegram_ok(False)
        logger.error("Telegram listener disabled: telegram.use_telethon is false.")

    try:
        while True:
            await asyncio.sleep(1)
            if listener_task is not None:
                alive = not listener_task.done()
                health_monitor.set_telegram_ok(alive)
                if listener_task.done() and (exc := listener_task.exception()) is not None:
                    logger.error("Telegram listener crashed: {}", exc)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    finally:
        if listener is not None:
            await listener.stop()
        for task in [listener_task, worker_task, engine_task, position_task, health_task]:
            if task is not None:
                task.cancel()
        for task in [listener_task, worker_task, engine_task, position_task, health_task]:
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task


if __name__ == "__main__":
    asyncio.run(run())
