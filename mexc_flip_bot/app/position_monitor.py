"""Background monitor that syncs exchange positions with local trades."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from app.database import Database, TradeRecord
from app.mexc_client import MexcFuturesClient


class PositionMonitor:
    def __init__(self, database: Database, mexc_client: MexcFuturesClient, logger: Any | None = None) -> None:
        self.database = database
        self.mexc_client = mexc_client
        self.logger = logger

    async def run_loop(self) -> None:
        while True:
            started = datetime.now(UTC)
            try:
                await self.sync_once()
            except Exception as exc:  # noqa: BLE001
                self._error("PositionMonitor sync error: {}", exc)
            elapsed = (datetime.now(UTC) - started).total_seconds()
            if elapsed > 10:
                self._warning("PositionMonitor API sync took {:.2f}s (>10s)", elapsed)
            await asyncio.sleep(3)

    async def sync_once(self) -> None:
        positions = await self.mexc_client.get_open_positions()
        pos_by_symbol = {
            self.mexc_client.normalize_symbol(str(p.get("symbol") or "")): p
            for p in positions
            if self.mexc_client.normalize_symbol(str(p.get("symbol") or ""))
        }

        open_trades = await self.database.list_open_trades()
        for trade in open_trades:
            await self._sync_trade(trade, pos_by_symbol)

    async def _sync_trade(self, trade: TradeRecord, pos_by_symbol: dict[str, dict[str, Any]]) -> None:
        symbol = self.mexc_client.normalize_symbol(trade.symbol)
        position = pos_by_symbol.get(symbol)
        if position is not None:
            return

        exit_price = await self._resolve_exit_price(symbol, trade)
        pnl = self._calculate_pnl(trade.direction, trade.entry_price, exit_price, trade.usdt_amount, trade.leverage)
        close_reason = self._infer_close_reason(trade, exit_price)

        await self.database.update_trade(
            trade.id,
            status="closed",
            close_reason=close_reason,
            exit_price=exit_price,
            pnl_usdt=pnl,
        )
        self._info("PositionMonitor: closed missing exchange position trade_id={} symbol={}", trade.id, symbol)

    async def _resolve_exit_price(self, symbol: str, trade: TradeRecord) -> float:
        try:
            price = await self.mexc_client.get_last_price(symbol)
            if price > 0:
                return float(price)
        except Exception:  # noqa: BLE001
            pass

        cached = self.mexc_client.get_market_price(symbol)
        if cached is not None and cached > 0:
            return float(cached)

        if trade.entry_price is not None and trade.entry_price > 0:
            return float(trade.entry_price)

        return 0.0

    def _infer_close_reason(self, trade: TradeRecord, exit_price: float) -> str:
        if trade.tp_price and exit_price > 0:
            if trade.direction == "long" and exit_price >= trade.tp_price:
                return "tp"
            if trade.direction == "short" and exit_price <= trade.tp_price:
                return "tp"
        if trade.sl_price and exit_price > 0:
            if trade.direction == "long" and exit_price <= trade.sl_price:
                return "sl"
            if trade.direction == "short" and exit_price >= trade.sl_price:
                return "sl"
        return "unknown"

    def _calculate_pnl(
        self,
        direction: str,
        entry_price: float | None,
        exit_price: float,
        usdt_amount: float,
        leverage: int,
    ) -> float:
        if entry_price is None or entry_price <= 0 or exit_price <= 0:
            return 0.0
        change = (exit_price - entry_price) / entry_price
        if direction == "short":
            change = -change
        return usdt_amount * leverage * change

    def _info(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(message, *args)

    def _warning(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "warning"):
            self.logger.warning(message, *args)
        elif self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(message, *args)

    def _error(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "error"):
            self.logger.error(message, *args)
