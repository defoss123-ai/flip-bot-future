"""Trade engine for paper/live modes with unified behavior."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Callable

from app.config import AppConfig
from app.config_resolver import ConfigResolver
from app.database import Database, SignalRecord
from app.direction_engine import DirectionEngine, DirectionSignal, MarketData
from app.mexc_client import MexcFuturesClient
from app.risk_manager import RiskManager
from app.strategy_controller import StrategyController


class TradeEngine:
    """Executes one signal at a time and opens trades."""

    def __init__(
        self,
        config: AppConfig,
        database: Database,
        mexc_client: MexcFuturesClient,
        logger,
        is_running: Callable[[], bool],
        risk_manager: RiskManager,
        allow_new_entries: Callable[[], bool] | None = None,
        notifier=None,
    ) -> None:
        self.config = config
        self.database = database
        self.mexc_client = mexc_client
        self.logger = logger
        self.is_running = is_running
        self.risk_manager = risk_manager
        self.allow_new_entries = allow_new_entries
        self.notifier = notifier
        self._last_entry_by_symbol: dict[str, datetime] = {}
        self._contracts_cache: set[str] = set()
        self._contracts_last_loaded: datetime | None = None
        self.config_resolver = ConfigResolver(self.config)
        self.strategy_controller = StrategyController()
        self._loss_guard_until: datetime | None = None
        self._api_guard_until: datetime | None = None
        self._order_error_times: deque[datetime] = deque(maxlen=100)
        self.direction_engine = DirectionEngine(
            min_spread_percent=self.config.filters.min_spread_percent,
            min_liquidity_usd=self.config.filters.min_liquidity_usd,
        )


    async def _active_runtime_settings(self) -> dict[str, object] | None:
        profile = await self.database.get_active_profile()
        if profile is None:
            return None
        try:
            data = json.loads(profile.data_json or "{}")
        except Exception:  # noqa: BLE001
            data = {}
        return data if isinstance(data, dict) else None

    async def _effective_settings(self):
        settings = await self._active_runtime_settings()
        self.config_resolver.config = self.config
        return self.config_resolver.get_effective_settings(settings)

    async def run_loop(self) -> None:
        while True:
            try:
                if self.is_running():
                    effective = await self._effective_settings()
                    await self._refresh_loss_streak_guard(effective)
                    await self._process_time_exits(effective)
                    signal = await self.database.get_next_signal(status="new")
                    if signal is not None:
                        await self._handle_signal(signal)
            except Exception as exc:  # noqa: BLE001
                self.logger.error("TradeEngine loop error: {}", exc)
            await asyncio.sleep(2)

    async def close_trade(self, trade_id: int) -> tuple[bool, str]:
        trade = await self.database.get_trade_by_id(trade_id)
        if trade is None:
            return False, "trade_not_found"

        if trade.status in {"closed", "canceled", "error"}:
            return False, f"trade_not_closable:{trade.status}"

        await self.database.update_trade(trade_id, status="closing")
        self.logger.info("TradeEngine: trade id={} marked as closing", trade_id)

        current_price = self.mexc_client.get_market_price(trade.symbol)
        if current_price is None:
            current_price = await self.database.get_latest_signal_price(trade.symbol)

        if current_price is None or current_price <= 0:
            await self.database.update_trade(trade_id, status="error", close_reason="error")
            return False, "missing_market_price"

        pnl = self._calculate_pnl(trade.direction, trade.entry_price, current_price, trade.usdt_amount, trade.leverage)
        await self.database.update_trade(
            trade_id,
            exit_price=current_price,
            pnl_usdt=pnl,
            status="closed",
            close_reason="manual",
        )
        self.logger.info("TradeEngine: trade id={} closed manually at {} pnl={}", trade_id, current_price, pnl)
        return True, "closed"

    async def close_all_positions(self, reason: str = "panic") -> tuple[int, int]:
        positions = await self.mexc_client.get_open_positions()
        if not positions:
            self.logger.info("TradeEngine: panic close requested, no open positions")
            return 0, 0

        open_trades = await self.database.list_open_trades()
        trades_by_symbol: dict[str, list] = {}
        for trade in open_trades:
            trades_by_symbol.setdefault(trade.symbol, []).append(trade)

        closed_count = 0
        error_count = 0
        for position in positions:
            symbol = self.normalize_symbol(str(position.get("symbol") or ""))
            if symbol is None:
                error_count += 1
                continue

            qty = abs(float(position.get("signed_size") or position.get("size") or 0.0))
            if qty <= 0:
                self.logger.info("TradeEngine: panic skip symbol={} qty<=0", symbol)
                continue

            side = str(position.get("side") or "").lower()
            if side not in {"long", "short"}:
                signed_size = float(position.get("signed_size") or 0.0)
                side = "short" if signed_size < 0 else "long"
            close_side = "buy" if side == "short" else "sell"

            try:
                close_id = await self.mexc_client.place_order(
                    symbol=symbol,
                    side=close_side,
                    type="market",
                    qty=qty,
                    reduce_only=True,
                )
                filled, filled_price = await self._wait_fill(symbol, close_id, 8)
                if not filled:
                    filled_price = self.mexc_client.get_market_price(symbol) or float(position.get("entry_price") or 0.0)
                if filled_price <= 0:
                    filled_price = float(position.get("entry_price") or 0.0)

                trade = None
                if symbol in trades_by_symbol and trades_by_symbol[symbol]:
                    trade = trades_by_symbol[symbol].pop(0)

                if trade is None:
                    self.logger.info("TradeEngine: panic closed exchange position symbol={} without matching open trade", symbol)
                    closed_count += 1
                    continue

                pnl = self._calculate_pnl(
                    trade.direction,
                    trade.entry_price,
                    filled_price,
                    trade.usdt_amount,
                    trade.leverage,
                )
                await self.database.update_trade(
                    trade.id,
                    status="closed",
                    close_reason=reason,
                    exit_price=filled_price,
                    pnl_usdt=pnl,
                )
                self.logger.info(
                    "TradeEngine: panic closed trade_id={} symbol={} exit_price={} pnl={}",
                    trade.id,
                    symbol,
                    filled_price,
                    pnl,
                )
                closed_count += 1
            except Exception as exc:  # noqa: BLE001
                error_count += 1
                self.logger.error("TradeEngine: panic close failed symbol={} err={}", symbol, exc)
                continue

        self.logger.info(
            "TradeEngine: panic close completed closed={} errors={} reason={}",
            closed_count,
            error_count,
            reason,
        )
        return closed_count, error_count

    async def _handle_signal(self, signal: SignalRecord) -> None:
        self.logger.info("TradeEngine: processing signal id={} symbol={}", signal.id, signal.symbol)

        effective = await self._effective_settings()
        can_enter, reason = await self._can_enter(signal, effective.cooldown_sec)
        if not can_enter:
            self.logger.info("TradeEngine: skip signal id={} reason={}", signal.id, reason)
            await self.database.mark_signal_status(signal.id, "error")
            return

        strategy_decision = self.strategy_controller.on_before_entry(
            signal,
            effective,
            self._loss_guard_until,
            self._api_guard_until,
        )
        if not strategy_decision.allowed:
            self.logger.info("TradeEngine: strategy block signal id={} reason={}", signal.id, strategy_decision.reason)
            await self.database.mark_signal_status(signal.id, "filtered")
            return

        symbol = self.normalize_symbol(signal.symbol)
        if symbol is None:
            await self.database.mark_signal_status(signal.id, "error")
            return

        if not await self._symbol_exists(symbol):
            self.logger.info("TradeEngine: skip signal id={} unknown symbol={}", signal.id, symbol)
            await self.database.mark_signal_status(signal.id, "error")
            return

        funding_rate = None
        try:
            funding_rate = await self.mexc_client.get_funding_rate(symbol)
        except Exception as exc:  # noqa: BLE001
            self.logger.info("TradeEngine: funding unavailable symbol={} err={}", symbol, exc)

        self.direction_engine.min_spread_percent = effective.min_spread_percent
        self.direction_engine.min_liquidity_usd = effective.min_liquidity_usd
        self.direction_engine.min_confidence = effective.min_confidence

        volatility_1m = None
        if signal.spread_percent is not None:
            volatility_1m = abs(float(signal.spread_percent)) / 100.0

        market_data = MarketData(
            funding_rate=funding_rate,
            volatility_1m=volatility_1m,
            base_leverage=self.config.trading.leverage,
        )
        signal_data = DirectionSignal(
            dex_price=signal.dex_price,
            mexc_price=signal.mexc_price,
            spread_percent=signal.spread_percent,
            liquidity_usd=signal.liquidity_usd,
        )
        direction = self.direction_engine.determine_direction(signal_data, market_data)
        if direction is None:
            self.logger.info(
                "TradeEngine: signal id={} filtered by direction_engine confidence={} adjusted_lev={}",
                signal.id,
                market_data.confidence_score,
                market_data.adjusted_leverage,
            )
            await self.database.mark_signal_status(signal.id, "filtered")
            return

        side = "buy" if direction == "long" else "sell"
        entry_type = effective.entry_type

        leverage_to_use = self._resolve_leverage(
            base_leverage=self.config.trading.leverage,
            effective=effective,
            signal=signal,
            volatility_1m=volatility_1m,
            direction_adjusted=market_data.adjusted_leverage,
        )

        last_price = await self._resolve_last_price(symbol, signal)
        if last_price is None:
            self.logger.info("TradeEngine: signal id={} missing valid last price", signal.id)
            await self.database.mark_signal_status(signal.id, "error")
            return

        qty, min_qty = await self._compute_qty(symbol, last_price, leverage_to_use)

        risk = await self.risk_manager.evaluate_entry(
            signal=signal,
            symbol=symbol,
            qty=qty,
            last_price=last_price,
            min_qty=min_qty,
            max_positions_override=effective.max_positions,
            min_spread_override=effective.min_spread_percent,
        )
        if not risk.allowed:
            self.logger.info("TradeEngine: skip signal id={} reason={}", signal.id, risk.reason)
            await self.database.mark_signal_status(signal.id, "filtered")
            return

        trade_id = await self.database.add_trade(
            symbol=symbol,
            direction=direction,
            usdt_amount=self.config.trading.usdt_amount,
            leverage=leverage_to_use,
            entry_price=None,
            status="opening",
            close_reason=None,
        )
        self.logger.info("TradeEngine: created trade id={} status=opening", trade_id)

        await self.mexc_client.set_leverage(symbol, leverage_to_use)

        entry_price = None
        if entry_type == "limit":
            adjust = self.strategy_controller.get_entry_price_adjustment(direction, effective)
            entry_price = last_price * adjust
        try:
            entry_order_id = await self.mexc_client.place_order(
                symbol=symbol,
                side=side,
                type=entry_type,
                qty=qty,
                price=entry_price,
                reduce_only=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._record_order_error(effective)
            self.logger.error("TradeEngine: entry placement failed trade_id={} err={}", trade_id, exc)
            await self.database.update_trade(trade_id, status="error", close_reason="error")
            await self.database.mark_signal_status(signal.id, "error")
            return

        self.logger.info("TradeEngine: placed entry order {} type={}", entry_order_id, entry_type)

        timeout = effective.entry_timeout_sec if entry_type == "limit" else 5
        filled, filled_price = await self._wait_fill(symbol, entry_order_id, timeout)
        if not filled:
            self.logger.info("TradeEngine: entry timeout for order {}", entry_order_id)
            await self.mexc_client.cancel_order(symbol, entry_order_id)
            retry_result = await self._retry_after_timeout(symbol, side, qty, effective, direction)
            if retry_result is None:
                await self.database.update_trade(trade_id, status="canceled", close_reason="timeout")
                await self.database.mark_signal_status(signal.id, "used")
                self.logger.info("TradeEngine: trade id={} canceled due timeout", trade_id)
                return
            filled_price = retry_result

        try:
            tp_id, sl_id = await self.mexc_client.place_tp_sl(
                symbol=symbol,
                direction=direction,
                qty=qty,
                entry_price=filled_price,
                tp_percent=effective.tp_percent,
                sl_percent=effective.sl_percent,
            )
            self.logger.info("TradeEngine: TP/SL placed tp_id={} sl_id={}", tp_id, sl_id)
        except Exception as exc:  # noqa: BLE001
            self._record_order_error(effective)
            self.logger.error("TradeEngine: TP/SL placement failed trade_id={} err={}", trade_id, exc)
            await self._force_close_after_tp_sl_error(trade_id, symbol, side, qty, filled_price, direction, leverage_to_use)
            await self.database.mark_signal_status(signal.id, "error")
            return

        await self.database.update_trade(trade_id, status="open", entry_price=filled_price)
        await self.database.mark_signal_status(signal.id, "used")
        if self.notifier is not None and hasattr(self.notifier, "send_message"):
            try:
                asyncio.create_task(
                    self.notifier.send_message(
                        f"📈 OPEN {direction.upper()} {symbol}\nEntry: {filled_price}\nLeverage: {leverage_to_use}\nSize: {qty}"
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.info("TradeEngine: notifier failed err={}", exc)

        self._last_entry_by_symbol[symbol] = datetime.now(UTC)
        self.logger.info("TradeEngine: trade id={} is open with entry_price={} leverage={}", trade_id, filled_price, leverage_to_use)

    async def _force_close_after_tp_sl_error(
        self,
        trade_id: int,
        symbol: str,
        side: str,
        qty: float,
        entry_price: float,
        direction: str,
        leverage: int,
    ) -> None:
        close_side = "sell" if side == "buy" else "buy"
        exit_price = self.mexc_client.get_market_price(symbol) or entry_price
        try:
            close_id = await self.mexc_client.place_order(
                symbol=symbol,
                side=close_side,
                type="market",
                qty=qty,
                reduce_only=True,
            )
            _, close_fill = await self._wait_fill(symbol, close_id, 5)
            if close_fill > 0:
                exit_price = close_fill
        except Exception as exc:  # noqa: BLE001
            self.logger.error("TradeEngine: emergency close failed trade_id={} err={}", trade_id, exc)

        pnl = self._calculate_pnl(direction, entry_price, exit_price, self.config.trading.usdt_amount, leverage)
        await self.database.update_trade(
            trade_id,
            exit_price=exit_price,
            pnl_usdt=pnl,
            status="closed",
            close_reason="error",
        )

    async def _can_enter(self, signal: SignalRecord, cooldown_sec: int) -> tuple[bool, str]:
        if not self.is_running():
            return False, "bot_not_running"

        if self.allow_new_entries is not None and not self.allow_new_entries():
            return False, "health_pause"

        if signal.symbol is None:
            return False, "missing_symbol"

        normalized = self.normalize_symbol(signal.symbol)
        if normalized is None:
            return False, "invalid_symbol"

        last_entry = self._last_entry_by_symbol.get(normalized)
        if last_entry is not None:
            cooldown_until = last_entry + timedelta(seconds=cooldown_sec)
            if datetime.now(UTC) < cooldown_until:
                return False, "symbol_cooldown"

        return True, "ok"

    async def _wait_fill(self, symbol: str, exchange_order_id: str, timeout_sec: int) -> tuple[bool, float]:
        deadline = datetime.now(UTC) + timedelta(seconds=timeout_sec)
        while datetime.now(UTC) < deadline:
            order_info = await self.mexc_client.get_order(symbol, exchange_order_id)
            if order_info.get("status") == "filled":
                filled_price = float(order_info.get("avg_price") or order_info.get("filled_price") or 0.0)
                return True, filled_price
            await asyncio.sleep(1)
        return False, 0.0

    async def _retry_after_timeout(self, symbol: str, side: str, qty: float, effective, direction: str) -> float | None:
        retry_mode = effective.retry_mode
        if retry_mode == "cancel":
            return None

        if retry_mode == "cancel_then_market":
            retry_id = await self.mexc_client.place_order(symbol=symbol, side=side, type="market", qty=qty)
            filled, filled_price = await self._wait_fill(symbol, retry_id, 5)
            return filled_price if filled else None

        if retry_mode == "cancel_then_relimit":
            current_price = self.mexc_client.get_market_price(symbol)
            if current_price is None or current_price <= 0:
                return None
            adj = self.strategy_controller.get_entry_price_adjustment(direction, effective)
            retry_price = current_price * adj
            retry_id = await self.mexc_client.place_order(
                symbol=symbol,
                side=side,
                type="limit",
                qty=qty,
                price=retry_price,
            )
            filled, filled_price = await self._wait_fill(symbol, retry_id, effective.entry_timeout_sec)
            return filled_price if filled else None

        return None

    async def _symbol_exists(self, symbol: str) -> bool:
        now = datetime.now(UTC)
        if self._contracts_last_loaded is None or (now - self._contracts_last_loaded).total_seconds() > 300:
            contracts = await self.mexc_client.get_contracts()
            self._contracts_cache = {self.normalize_symbol(c) for c in contracts if self.normalize_symbol(c)}
            self._contracts_last_loaded = now
        return symbol in self._contracts_cache

    async def _resolve_last_price(self, symbol: str, signal: SignalRecord) -> float | None:
        price = signal.mexc_price if signal.mexc_price and signal.mexc_price > 0 else None
        if price is None:
            try:
                price = await self.mexc_client.get_last_price(symbol)
            except Exception:  # noqa: BLE001
                price = None
        if price is None:
            price = self.mexc_client.get_market_price(symbol)
        if price is None or price <= 0:
            return None
        return float(price)

    async def _compute_qty(self, symbol: str, last_price: float, leverage: int) -> tuple[float, float]:
        raw_qty = (self.config.trading.usdt_amount * leverage) / last_price
        rules = await self.mexc_client.get_symbol_rules(symbol)
        step = float(rules.get("step_size") or 0)
        min_qty = float(rules.get("min_qty") or 0)

        qty = raw_qty
        if step > 0:
            qty = (qty // step) * step
            qty = round(qty, 8)
        else:
            qty = round(qty, 6)

        return max(qty, 0.0), max(min_qty, 0.0)

    def _resolve_leverage(self, base_leverage: int, effective, signal: SignalRecord, volatility_1m: float | None, direction_adjusted: int | None) -> int:
        if effective.leverage_mode == "fixed":
            lev = base_leverage
        else:
            lev = float(base_leverage)
            spread = float(signal.spread_percent or 0.0)
            liquidity = float(signal.liquidity_usd or 0.0)
            if spread > 2 * effective.min_spread_percent:
                lev *= 1.3
            if liquidity < 2 * effective.min_liquidity_usd:
                lev *= 0.7
            if volatility_1m is not None and volatility_1m > 0.03:
                lev *= 0.5
            lev = int(round(lev))

        if direction_adjusted is not None:
            lev = min(lev, int(direction_adjusted))

        lev = max(1, min(int(lev), int(effective.max_leverage)))
        return lev

    def _record_order_error(self, effective) -> None:
        now = datetime.now(UTC)
        self._order_error_times.append(now)
        while self._order_error_times and (now - self._order_error_times[0]).total_seconds() > 120:
            self._order_error_times.popleft()
        if self.strategy_controller.on_error_burst(effective, len(self._order_error_times)):
            self._api_guard_until = now + timedelta(minutes=10)
            self.logger.error("TradeEngine: panic_on_api_errors activated until={}", self._api_guard_until.isoformat())

    async def _refresh_loss_streak_guard(self, effective) -> None:
        if effective.strategy_mode != "conservative":
            return
        since = datetime.now(UTC) - timedelta(hours=24)
        closed = await self.database.list_closed_trades(since=since)
        losses = 0
        for trade in reversed(closed):
            pnl = float(trade.pnl_usdt or 0.0)
            if pnl < 0:
                losses += 1
            else:
                break
        if losses >= 3:
            until = datetime.now(UTC) + timedelta(minutes=30)
            if self._loss_guard_until is None or until > self._loss_guard_until:
                self._loss_guard_until = until
                self.logger.warning("TradeEngine: loss_streak_guard activated until={}", until.isoformat())

    async def _process_time_exits(self, effective) -> None:
        if effective.strategy_mode != "aggressive":
            return
        open_trades = await self.database.list_open_trades()
        now = datetime.now(UTC)
        for trade in open_trades:
            if trade.status != "open" or trade.entry_price is None or trade.entry_price <= 0:
                continue
            age_sec = (now - trade.created_at).total_seconds()
            price = self.mexc_client.get_market_price(trade.symbol)
            if price is None:
                try:
                    price = await self.mexc_client.get_last_price(trade.symbol)
                except Exception:
                    continue
            if price is None or price <= 0:
                continue
            pnl_pct = ((price - trade.entry_price) / trade.entry_price) * 100.0
            if trade.direction == "short":
                pnl_pct = -pnl_pct
            if not self.strategy_controller.should_time_exit(trade, age_sec, pnl_pct, effective):
                continue
            await self._close_trade_market(trade.id, price, "time_exit")

    async def _close_trade_market(self, trade_id: int, fallback_price: float, reason: str) -> None:
        trade = await self.database.get_trade_by_id(trade_id)
        if trade is None or trade.status != "open":
            return
        close_side = "sell" if trade.direction == "long" else "buy"
        qty = (trade.usdt_amount * trade.leverage) / max(trade.entry_price or fallback_price, 1e-9)
        exit_price = fallback_price
        try:
            close_id = await self.mexc_client.place_order(
                symbol=trade.symbol,
                side=close_side,
                type="market",
                qty=qty,
                reduce_only=True,
            )
            filled, fill_price = await self._wait_fill(trade.symbol, close_id, 5)
            if filled and fill_price > 0:
                exit_price = fill_price
        except Exception as exc:  # noqa: BLE001
            self.logger.error("TradeEngine: time_exit market close failed trade_id={} err={}", trade_id, exc)

        pnl = self._calculate_pnl(trade.direction, trade.entry_price, exit_price, trade.usdt_amount, trade.leverage)
        await self.database.update_trade(
            trade_id,
            exit_price=exit_price,
            pnl_usdt=pnl,
            status="closed",
            close_reason=reason,
        )

    def get_safety_status(self) -> dict[str, str | None]:
        now = datetime.now(UTC)
        loss_guard = self._loss_guard_until.isoformat() if self._loss_guard_until and now < self._loss_guard_until else None
        api_guard = self._api_guard_until.isoformat() if self._api_guard_until and now < self._api_guard_until else None
        return {"loss_streak_guard_until": loss_guard, "api_error_guard_until": api_guard}

    def normalize_symbol(self, symbol: str | None) -> str | None:
        if not symbol:
            return None
        normalized = self.mexc_client.normalize_symbol(symbol)
        return normalized if normalized else None

    def _calculate_pnl(
        self,
        direction: str,
        entry_price: float | None,
        exit_price: float,
        usdt_amount: float,
        leverage: int,
    ) -> float:
        if entry_price is None or entry_price <= 0:
            return 0.0
        change = (exit_price - entry_price) / entry_price
        if direction == "short":
            change = -change
        return usdt_amount * leverage * change
