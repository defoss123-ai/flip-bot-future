"""Real MEXC Futures (USDT-M) client with safe async wrappers."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.database import Database
from app.mexc_sign import build_query, signed_query


class MexcFuturesClientReal:
    """Best-effort async wrapper for MEXC Futures REST API."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        logger: Any | None = None,
        base_url: str = "https://contract.mexc.com",
        database: Database | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.logger = logger or logging.getLogger(__name__)
        self.database = database
        self._last_price_by_symbol: dict[str, float] = {}
        self._contracts_cache: dict[str, dict[str, float]] = {}

    async def test_connection(self) -> bool:
        try:
            await self._request("GET", "/api/v1/contract/ping")
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_error("test_connection", exc)
            try:
                await self._signed_request("GET", "/api/v1/private/account/assets")
                return True
            except Exception as inner:  # noqa: BLE001
                self._log_error("test_connection_fallback", inner)
                return False

    async def get_contracts(self) -> list[str]:
        try:
            data = await self._request("GET", "/api/v1/contract/detail")
            rows = data.get("data") if isinstance(data, dict) else data
            if not isinstance(rows, list):
                return []
            contracts: list[str] = []
            cache: dict[str, dict[str, float]] = {}
            for row in rows:
                symbol = self.normalize_symbol(str(row.get("symbol") or row.get("contractCode") or ""))
                if not symbol:
                    continue
                contracts.append(symbol)
                cache[symbol] = {
                    "step_size": float(row.get("volUnit") or row.get("stepSize") or 0.0001),
                    "min_qty": float(row.get("minVol") or row.get("minQty") or 0.0),
                }
            self._contracts_cache = cache
            return sorted(set(contracts))
        except Exception as exc:  # noqa: BLE001
            self._log_error("get_contracts", exc)
            return []

    async def get_symbol_rules(self, symbol: str) -> dict[str, float]:
        normalized = self.normalize_symbol(symbol)
        if normalized not in self._contracts_cache:
            await self.get_contracts()
        return self._contracts_cache.get(normalized, {"step_size": 0.0001, "min_qty": 0.0})

    async def get_open_positions(self) -> list[dict[str, float | str]]:
        try:
            data = await self._signed_request("GET", "/api/v1/private/position/open_positions")
            rows = data.get("data") if isinstance(data, dict) else []
            if not isinstance(rows, list):
                return []
            result: list[dict[str, float | str]] = []
            for row in rows:
                raw_size = float(row.get("holdVol") or row.get("positionVol") or 0)
                if raw_size <= 0:
                    continue
                symbol = self.normalize_symbol(str(row.get("symbol") or ""))
                if not symbol:
                    continue
                pos_type = str(row.get("positionType") or row.get("position_type") or "").lower()
                side_field = str(row.get("positionSide") or row.get("holdSide") or "").lower()
                is_short = pos_type in {"2", "short"} or side_field == "short"
                signed_size = -raw_size if is_short else raw_size
                entry_price = float(row.get("openAvgPrice") or row.get("avgPrice") or row.get("openAvg") or 0.0)
                unrealized = float(row.get("unrealizedPnl") or row.get("unrealizedProfit") or 0.0)
                result.append(
                    {
                        "symbol": symbol,
                        "size": raw_size,
                        "signed_size": signed_size,
                        "side": "short" if is_short else "long",
                        "entry_price": entry_price,
                        "unrealized_pnl": unrealized,
                    }
                )
            return result
        except Exception as exc:  # noqa: BLE001
            self._log_error("get_open_positions", exc)
            return []

    async def has_open_position(self, symbol: str) -> bool:
        try:
            normalized = self.normalize_symbol(symbol)
            data = await self._signed_request("GET", "/api/v1/private/position/open_positions")
            rows = data.get("data") if isinstance(data, dict) else []
            if not isinstance(rows, list):
                return False
            for row in rows:
                s = self.normalize_symbol(str(row.get("symbol") or ""))
                hold = float(row.get("holdVol") or row.get("positionVol") or 0)
                if s == normalized and hold > 0:
                    return True
            return False
        except Exception as exc:  # noqa: BLE001
            self._log_error("has_open_position", exc)
            return False

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            payload = {
                "symbol": self._exchange_symbol(symbol),
                "leverage": int(leverage),
                "openType": 2,
                "positionType": 1,
            }
            await self._signed_request("POST", "/api/v1/private/position/change_leverage", payload)
        except Exception as exc:  # noqa: BLE001
            self._log_error("set_leverage", exc)

    async def get_funding_rate(self, symbol: str) -> float | None:
        try:
            data = await self._request("GET", "/api/v1/contract/funding_rate", {"symbol": self._exchange_symbol(symbol)})
            row = data.get("data") if isinstance(data, dict) else None
            if isinstance(row, list):
                row = row[0] if row else {}
            if not isinstance(row, dict):
                return None
            value = row.get("fundingRate") or row.get("funding_rate") or row.get("fairRate")
            if value is None:
                return None
            return float(value) * 100
        except Exception as exc:  # noqa: BLE001
            self._log_error("get_funding_rate", exc)
            return None

    async def get_last_price(self, symbol: str) -> float:
        try:
            data = await self._request("GET", "/api/v1/contract/ticker", {"symbol": self._exchange_symbol(symbol)})
            row = data.get("data", {}) if isinstance(data, dict) else {}
            price = row.get("lastPrice") or row.get("last_price") or row.get("price")
            if price is None:
                raise ValueError("last price missing")
            value = float(price)
            self._last_price_by_symbol[self.normalize_symbol(symbol)] = value
            return value
        except Exception as exc:  # noqa: BLE001
            self._log_error("get_last_price", exc)
            cached = self._last_price_by_symbol.get(self.normalize_symbol(symbol))
            if cached is not None:
                return cached
            raise

    async def place_order(
        self,
        symbol: str,
        side: str,
        type: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
    ) -> str:
        try:
            order_type = type.upper()
            normalized_symbol = self._exchange_symbol(symbol)
            payload: dict[str, object] = {
                "symbol": normalized_symbol,
                "vol": float(qty),
                "side": self._to_mexc_side(side.lower(), reduce_only),
                "type": 1 if order_type == "LIMIT" else 5,
                "openType": 2,
                "externalOid": str(uuid.uuid4()),
                "reduceOnly": True if reduce_only else None,
                "price": float(price) if price is not None else None,
            }
            data = await self._signed_request("POST", "/api/v1/private/order/submit", payload)
            oid = data.get("data", {}).get("orderId") if isinstance(data, dict) else None
            if oid is None:
                oid = data.get("orderId") if isinstance(data, dict) else None
            if oid is None:
                raise ValueError("exchange order id missing")
            return str(oid)
        except Exception as exc:  # noqa: BLE001
            self._log_error("place_order", exc)
            await self._save_error_order(symbol, side, type, qty, price, str(exc))
            raise

    async def get_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        try:
            data = await self._signed_request(
                "GET",
                "/api/v1/private/order/get",
                {"symbol": self._exchange_symbol(symbol), "orderId": order_id},
            )
            row = data.get("data", {}) if isinstance(data, dict) else {}
            status_code = int(row.get("state") or row.get("status") or -1)
            avg_price = row.get("dealAvgPrice") or row.get("avgPrice") or row.get("price")
            filled_qty = row.get("dealVol") or row.get("filledQty") or 0
            return {
                "status": self._map_order_status(status_code),
                "avg_price": float(avg_price) if avg_price not in (None, "") else None,
                "filled_qty": float(filled_qty or 0),
            }
        except Exception as exc:  # noqa: BLE001
            self._log_error("get_order", exc)
            return {"status": "error", "avg_price": None, "filled_qty": 0.0}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            await self._signed_request(
                "POST",
                "/api/v1/private/order/cancel",
                {"symbol": self._exchange_symbol(symbol), "orderId": order_id},
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._log_error("cancel_order", exc)
            return False

    async def place_tp_sl(
        self,
        symbol: str,
        direction: str,
        qty: float,
        entry_price: float,
        tp_percent: float,
        sl_percent: float,
    ) -> tuple[str, str]:
        tp_price, sl_price, close_side = self._tp_sl_params(direction, entry_price, tp_percent, sl_percent)
        tp_id = await self._place_with_retry(symbol, close_side, "LIMIT", qty, tp_price, True, "tp")
        sl_id = await self._place_with_retry(symbol, close_side, "LIMIT", qty, sl_price, True, "sl")
        return tp_id, sl_id

    def get_market_price(self, symbol: str) -> float | None:
        return self._last_price_by_symbol.get(self.normalize_symbol(symbol))

    async def apply_market_price(self, symbol: str, current_price: float) -> None:
        self._last_price_by_symbol[self.normalize_symbol(symbol)] = float(current_price)

    def normalize_symbol(self, symbol: str) -> str:
        value = symbol.replace("/", "_").replace("-", "_").upper().strip()
        if value.endswith("USDT") and not value.endswith("_USDT"):
            value = value[:-4] + "_USDT"
        return value

    async def _place_with_retry(
        self,
        symbol: str,
        side: str,
        type: str,
        qty: float,
        price: float,
        reduce_only: bool,
        tag: str,
    ) -> str:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                return await self.place_order(symbol, side, type, qty, price=price, reduce_only=reduce_only)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._log_error(f"place_tp_sl_{tag}", exc)
                await self._save_error_order(symbol, side, type, qty, price, f"{tag}: {exc}")
                await asyncio.sleep(0.5)
        raise RuntimeError(f"Failed to place {tag} after retries: {last_error}")

    async def _save_error_order(
        self,
        symbol: str,
        side: str,
        type: str,
        qty: float,
        price: float | None,
        reason: str,
    ) -> None:
        if self.database is None:
            return
        try:
            await self.database.add_order(
                symbol=self.normalize_symbol(symbol),
                side=side.lower(),
                type=type.lower(),
                price=price,
                qty=qty,
                status="error",
                exchange_order_id=None,
                reason=reason[:250],
            )
        except Exception as exc:  # noqa: BLE001
            self._log_error("save_error_order", exc)

    async def safe_request(
        self,
        method: str,
        path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        signed: bool = False,
    ) -> dict[str, Any]:
        retries = 3
        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                if signed:
                    return await self._signed_request_once(method, path, params)
                return await self._request_once(method, path, params, headers)
            except RuntimeError as exc:
                message = str(exc).lower()
                is_signature = "signature" in message
                is_retryable = any(k in message for k in ["timeout", "timed out", "http 5", "httperror 5", "urlerror"])
                if is_signature:
                    self._log_error("safe_request_signature", exc)
                    raise
                if not is_retryable or attempt >= retries:
                    raise
                last_exc = exc
                self._log_warning("MEXC request retry {}/{} path={} err={}", attempt, retries, path, exc)
                await asyncio.sleep(delay)
                delay *= 2
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("safe_request failed")

    async def _request(self, method: str, path: str, params: dict[str, object] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
        return await self.safe_request(method, path, params=params, headers=headers, signed=False)

    async def _signed_request(self, method: str, path: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        return await self.safe_request(method, path, params=params, signed=True)

    async def _request_once(
        self,
        method: str,
        path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._request_sync, method, path, params, headers)

    async def _signed_request_once(self, method: str, path: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        params = params or {}
        query, signature = signed_query(params, self.api_secret)
        method = method.upper()

        if method == "GET":
            headers = {
                "ApiKey": self.api_key,
                "Request-Time": self._extract_ts(query),
                "Signature": signature,
            }
            return await self._request_once("GET", path, self._decode_query_to_payload(query), headers)

        payload = self._decode_query_to_payload(query)
        payload["signature"] = signature
        headers = {"ApiKey": self.api_key, "Content-Type": "application/json"}
        return await self._request_once("POST", path, payload, headers)

    def _request_sync(
        self,
        method: str,
        path: str,
        params: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        headers = headers or {}
        url = f"{self.base_url}{path}"
        method = method.upper()
        data = None

        if method == "GET" and params:
            url = f"{url}?{build_query(params)}"
        elif method == "POST":
            data = json.dumps(params).encode("utf-8")

        req = Request(url=url, data=data, method=method, headers=headers)

        try:
            with urlopen(req, timeout=10) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTPError {exc.code}: {raw}") from exc
        except URLError as exc:
            raise RuntimeError(f"URLError: {exc}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"timeout: {exc}") from exc

        payload = json.loads(raw or "{}")
        code = payload.get("code")
        if code not in (0, 200, None):
            raise RuntimeError(f"API error code={code} msg={payload.get('msg')}")
        return payload

    def _exchange_symbol(self, symbol: str) -> str:
        return self.normalize_symbol(symbol)

    def _to_mexc_side(self, side: str, reduce_only: bool) -> int:
        if side not in {"buy", "sell"}:
            raise ValueError("invalid side")
        if not reduce_only:
            return 1 if side == "buy" else 3
        return 4 if side == "buy" else 2

    def _map_order_status(self, status_code: int) -> str:
        mapping = {1: "new", 2: "placed", 3: "canceled", 4: "filled", 5: "rejected", 6: "error"}
        return mapping.get(status_code, "placed")

    def _tp_sl_params(self, direction: str, entry_price: float, tp_percent: float, sl_percent: float) -> tuple[float, float, str]:
        if direction.lower() == "long":
            return entry_price * (1 + tp_percent / 100), entry_price * (1 - sl_percent / 100), "sell"
        return entry_price * (1 - tp_percent / 100), entry_price * (1 + sl_percent / 100), "buy"

    def _decode_query_to_payload(self, query: str) -> dict[str, object]:
        payload: dict[str, object] = {}
        for part in query.split("&"):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            payload[k] = v
        return payload

    def _extract_ts(self, query: str) -> str:
        for part in query.split("&"):
            if part.startswith("timestamp="):
                return part.split("=", 1)[1]
        return "0"

    def _log_warning(self, message: str, *args: object) -> None:
        if hasattr(self.logger, "warning"):
            self.logger.warning(message, *args)
        elif hasattr(self.logger, "info"):
            self.logger.info(message, *args)

    def _log_error(self, scope: str, exc: Exception) -> None:
        if hasattr(self.logger, "error"):
            self.logger.error("MEXC real client error [{}]: {}", scope, exc)
