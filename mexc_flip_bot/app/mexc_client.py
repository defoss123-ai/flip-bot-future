"""Unified MEXC futures client wrapper (paper/live)."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from app.database import Database
from app.mexc_client_real import MexcFuturesClientReal


@dataclass(slots=True)
class PaperOrder:
    exchange_order_id: str
    symbol: str
    side: str
    type: str
    qty: float
    price: float | None
    status: str
    filled_price: float | None
    db_order_id: int
    reduce_only: bool = False
    reason: str | None = None


class _MexcFuturesClientPaper:
    VALID_SIDES = {"buy", "sell"}
    VALID_TYPES = {"market", "limit"}

    def __init__(self, database: Database, mode: str = "paper") -> None:
        self.database = database
        self.mode = mode
        self._contracts = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "DOGE_USDT"]
        self._leverage_by_symbol: dict[str, int] = {}
        self._last_price_by_symbol: dict[str, float] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._positions: dict[str, dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def test_connection(self) -> bool:
        return self.mode == "paper"

    async def get_contracts(self) -> list[str]:
        return self._contracts.copy()

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        self._leverage_by_symbol[self.normalize_symbol(symbol)] = leverage

    async def get_symbol_rules(self, symbol: str) -> dict[str, float]:
        return {"step_size": 0.0001, "min_qty": 0.0}

    async def has_open_position(self, symbol: str) -> bool:
        sym = self.normalize_symbol(symbol)
        pos = self._positions.get(sym)
        return bool(pos and abs(pos.get("size", 0.0)) > 0)

    async def get_funding_rate(self, symbol: str) -> float | None:
        return None

    async def get_open_positions(self) -> list[dict[str, float | str]]:
        result: list[dict[str, float | str]] = []
        for symbol, pos in self._positions.items():
            size = float(pos.get("size", 0.0))
            if abs(size) <= 0:
                continue
            entry_price = float(pos.get("entry_price", 0.0))
            last = self._last_price_by_symbol.get(symbol, entry_price)
            side_mult = 1 if size > 0 else -1
            unrealized = abs(size) * (last - entry_price) * side_mult
            result.append(
                {
                    "symbol": symbol,
                    "size": abs(size),
                    "signed_size": size,
                    "side": "long" if size > 0 else "short",
                    "entry_price": entry_price,
                    "unrealized_pnl": unrealized,
                }
            )
        return result

    async def get_last_price(self, symbol: str) -> float:
        price = self._last_price_by_symbol.get(self.normalize_symbol(symbol))
        if price is None:
            raise ValueError("market price unavailable")
        return price

    def get_market_price(self, symbol: str) -> float | None:
        return self._last_price_by_symbol.get(self.normalize_symbol(symbol))

    def normalize_symbol(self, symbol: str) -> str:
        value = symbol.replace("/", "_").replace("-", "_").upper().strip()
        if value.endswith("USDT") and not value.endswith("_USDT"):
            value = value[:-4] + "_USDT"
        return value

    async def place_order(
        self,
        symbol: str,
        side: str,
        type: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> str:
        async with self._lock:
            symbol = self.normalize_symbol(symbol)
            exchange_order_id = str(uuid.uuid4())
            normalized_side = side.lower()
            normalized_type = type.lower()
            current_price = self._last_price_by_symbol.get(symbol)
            status = "placed"
            filled_price: float | None = None
            reason: str | None = None

            if normalized_side not in self.VALID_SIDES or normalized_type not in self.VALID_TYPES:
                status = "error"
                reason = "invalid side/type"
            elif normalized_type == "market":
                if current_price is not None:
                    status = "filled"
                    filled_price = current_price
                else:
                    status = "error"
                    reason = "market price unavailable"
            elif price is None:
                status = "error"
                reason = "limit order requires price"
            elif current_price is not None and self._limit_should_fill(normalized_side, current_price, price):
                status = "filled"
                filled_price = price

            db_order_id = await self.database.add_order(
                symbol=symbol,
                side=normalized_side,
                type=normalized_type,
                price=price,
                qty=qty,
                status=status,
                exchange_order_id=exchange_order_id,
                reason=reason,
            )
            order = PaperOrder(
                exchange_order_id=exchange_order_id,
                symbol=symbol,
                side=normalized_side,
                type=normalized_type,
                qty=qty,
                price=price,
                status=status,
                filled_price=filled_price,
                db_order_id=db_order_id,
                reduce_only=reduce_only,
                reason=reason,
            )
            self._orders[exchange_order_id] = order

            if status == "filled" and filled_price is not None:
                self._apply_fill(symbol, normalized_side, qty, filled_price, reduce_only)

            return exchange_order_id

    async def get_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        symbol = self.normalize_symbol(symbol)
        order = self._orders.get(order_id)
        if order is None or order.symbol != symbol:
            return {"status": "rejected", "avg_price": None, "filled_qty": 0.0, "filled_price": None}
        return {
            "status": order.status,
            "avg_price": order.filled_price,
            "filled_qty": order.qty if order.status == "filled" else 0.0,
            "filled_price": order.filled_price,
        }

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        async with self._lock:
            symbol = self.normalize_symbol(symbol)
            order = self._orders.get(order_id)
            if order is None or order.symbol != symbol:
                return False
            if order.status in {"filled", "canceled", "rejected", "error"}:
                return False
            order.status = "canceled"
            order.reason = "canceled_by_user"
            await self.database.update_order_status(order.db_order_id, status="canceled", reason=order.reason)
            return True

    async def place_tp_sl(
        self,
        symbol: str,
        direction: str,
        qty: float,
        entry_price: float,
        tp_percent: float,
        sl_percent: float,
    ) -> tuple[str, str]:
        if direction.lower() == "long":
            tp_side, sl_side = "sell", "sell"
            tp_price = entry_price * (1 + tp_percent / 100)
            sl_price = entry_price * (1 - sl_percent / 100)
        else:
            tp_side, sl_side = "buy", "buy"
            tp_price = entry_price * (1 - tp_percent / 100)
            sl_price = entry_price * (1 + sl_percent / 100)

        tp_id = await self.place_order(symbol, tp_side, "limit", qty, price=tp_price, reduce_only=True)
        sl_id = await self.place_order(symbol, sl_side, "limit", qty, price=sl_price, reduce_only=True)
        return tp_id, sl_id

    async def apply_market_price(self, symbol: str, current_price: float) -> None:
        async with self._lock:
            symbol = self.normalize_symbol(symbol)
            self._last_price_by_symbol[symbol] = current_price
            for order in self._orders.values():
                if order.symbol != symbol or order.type != "limit" or order.status != "placed" or order.price is None:
                    continue
                if self._limit_should_fill(order.side, current_price, order.price):
                    order.status = "filled"
                    order.filled_price = order.price
                    await self.database.update_order_status(order.db_order_id, status="filled")
                    self._apply_fill(symbol, order.side, order.qty, order.price, order.reduce_only)

    def _apply_fill(self, symbol: str, side: str, qty: float, price: float, reduce_only: bool) -> None:
        signed_qty = qty if side == "buy" else -qty
        pos = self._positions.get(symbol, {"size": 0.0, "entry_price": 0.0})
        size = float(pos["size"])
        entry = float(pos["entry_price"])

        if reduce_only:
            if size > 0:
                size = max(0.0, size - qty)
            elif size < 0:
                size = min(0.0, size + qty)
            if abs(size) <= 0:
                self._positions.pop(symbol, None)
            else:
                self._positions[symbol] = {"size": size, "entry_price": entry}
            return

        new_size = size + signed_qty
        if abs(new_size) <= 0:
            self._positions.pop(symbol, None)
            return

        if size == 0 or (size > 0 and signed_qty > 0) or (size < 0 and signed_qty < 0):
            weighted = (abs(size) * entry) + (abs(signed_qty) * price)
            new_entry = weighted / abs(new_size)
            self._positions[symbol] = {"size": new_size, "entry_price": new_entry}
        else:
            self._positions[symbol] = {"size": new_size, "entry_price": price if size == 0 else entry}

    def _limit_should_fill(self, side: str, current_price: float, limit_price: float) -> bool:
        return (side == "buy" and current_price <= limit_price) or (side == "sell" and current_price >= limit_price)


class MexcFuturesClient:
    """Common wrapper so callers do not depend on paper/live implementation."""

    def __init__(
        self,
        database: Database,
        mode: str,
        api_key: str | None = None,
        api_secret: str | None = None,
        logger: Any | None = None,
    ) -> None:
        self.mode = mode
        if mode == "live":
            if not api_key or not api_secret:
                raise ValueError("Live mode requires api_key and api_secret")
            self._client: Any = MexcFuturesClientReal(
                api_key=api_key,
                api_secret=api_secret,
                logger=logger,
                database=database,
            )
        else:
            self._client = _MexcFuturesClientPaper(database=database, mode=mode)

    async def test_connection(self) -> bool:
        return await self._client.test_connection()

    async def get_contracts(self) -> list[str]:
        return await self._client.get_contracts()

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self._client.set_leverage(symbol, leverage)

    async def get_symbol_rules(self, symbol: str) -> dict[str, float]:
        if hasattr(self._client, "get_symbol_rules"):
            return await self._client.get_symbol_rules(symbol)
        return {"step_size": 0.0001, "min_qty": 0.0}

    async def has_open_position(self, symbol: str) -> bool:
        if hasattr(self._client, "has_open_position"):
            return await self._client.has_open_position(symbol)
        return False

    async def get_open_positions(self) -> list[dict[str, float | str]]:
        if hasattr(self._client, "get_open_positions"):
            return await self._client.get_open_positions()
        return []

    async def get_funding_rate(self, symbol: str) -> float | None:
        if hasattr(self._client, "get_funding_rate"):
            return await self._client.get_funding_rate(symbol)
        return None

    async def get_last_price(self, symbol: str) -> float:
        return await self._client.get_last_price(symbol)

    async def place_order(
        self,
        symbol: str,
        side: str,
        type: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> str:
        return await self._client.place_order(symbol, side, type, qty, price=price, reduce_only=reduce_only)

    async def get_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        return await self._client.get_order(symbol, order_id)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return await self._client.cancel_order(symbol, order_id)

    async def place_tp_sl(
        self,
        symbol: str,
        direction: str,
        qty: float,
        entry_price: float,
        tp_percent: float,
        sl_percent: float,
    ) -> tuple[str, str]:
        return await self._client.place_tp_sl(symbol, direction, qty, entry_price, tp_percent, sl_percent)

    async def apply_market_price(self, symbol: str, current_price: float) -> None:
        if hasattr(self._client, "apply_market_price"):
            await self._client.apply_market_price(symbol, current_price)

    def get_market_price(self, symbol: str) -> float | None:
        if hasattr(self._client, "get_market_price"):
            return self._client.get_market_price(symbol)
        return None

    def normalize_symbol(self, symbol: str) -> str:
        if hasattr(self._client, "normalize_symbol"):
            return self._client.normalize_symbol(symbol)
        value = symbol.replace("/", "_").replace("-", "_").upper().strip()
        if value.endswith("USDT") and not value.endswith("_USDT"):
            value = value[:-4] + "_USDT"
        return value
