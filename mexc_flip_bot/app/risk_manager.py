"""Risk management layer for entry validation."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import AppConfig
from app.database import SignalRecord
from app.mexc_client import MexcFuturesClient


@dataclass(slots=True)
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    """Evaluates pre-entry risk checks in a single layer."""

    def __init__(
        self,
        config: AppConfig,
        mexc_client: MexcFuturesClient,
        logger,
    ) -> None:
        self.config = config
        self.mexc_client = mexc_client
        self.logger = logger

    async def evaluate_entry(
        self,
        signal: SignalRecord,
        symbol: str,
        qty: float,
        last_price: float,
        min_qty: float,
        max_positions_override: int | None = None,
        min_spread_override: float | None = None,
    ) -> RiskDecision:
        open_positions = await self.mexc_client.get_open_positions()

        if await self.mexc_client.has_open_position(symbol):
            self.logger.info("RiskManager: symbol already has open position symbol={} reason=risk_limit", symbol)
            return RiskDecision(False, "risk_limit")

        max_positions = max_positions_override if max_positions_override is not None else self.config.trading.max_positions
        if len(open_positions) >= max_positions:
            self.logger.info(
                "RiskManager: max_positions reached open={} limit={} reason=risk_limit",
                len(open_positions),
                max_positions,
            )
            return RiskDecision(False, "risk_limit")

        if qty <= 0:
            self.logger.info("RiskManager: non-positive qty symbol={} qty={} reason=risk_limit", symbol, qty)
            return RiskDecision(False, "risk_limit")

        if min_qty > 0 and qty < min_qty:
            self.logger.info(
                "RiskManager: qty below min_qty symbol={} qty={} min_qty={} reason=risk_limit",
                symbol,
                qty,
                min_qty,
            )
            return RiskDecision(False, "risk_limit")

        max_exposure = self.config.risk.max_total_exposure_usdt
        if max_exposure > 0:
            current_exposure = self._total_exposure_usdt(open_positions)
            next_exposure = current_exposure + (qty * last_price)
            if next_exposure > max_exposure:
                self.logger.info(
                    "RiskManager: exposure limit exceeded current={} next={} limit={} reason=exposure_limit",
                    round(current_exposure, 6),
                    round(next_exposure, 6),
                    max_exposure,
                )
                return RiskDecision(False, "exposure_limit")

        spread = signal.spread_percent or 0.0
        min_spread = min_spread_override if min_spread_override is not None else self.config.filters.min_spread_percent
        spread_limit = min_spread * self.config.risk.spread_anomaly_multiplier
        if spread_limit > 0 and spread > spread_limit:
            self.logger.info(
                "RiskManager: spread anomaly spread={} limit={} reason=risk_limit",
                spread,
                spread_limit,
            )
            return RiskDecision(False, "risk_limit")

        funding_rate = await self.mexc_client.get_funding_rate(symbol)
        if funding_rate is not None and funding_rate > self.config.risk.max_funding_rate_percent:
            self.logger.info(
                "RiskManager: funding filter symbol={} funding_rate={} limit={} reason=funding_filter",
                symbol,
                funding_rate,
                self.config.risk.max_funding_rate_percent,
            )
            return RiskDecision(False, "funding_filter")

        return RiskDecision(True, "ok")

    def _total_exposure_usdt(self, positions: list[dict[str, float | str]]) -> float:
        exposure = 0.0
        for position in positions:
            try:
                size = float(position.get("size") or 0.0)
                entry_price = float(position.get("entry_price") or 0.0)
            except (TypeError, ValueError):
                continue
            if entry_price > 0:
                exposure += abs(size) * entry_price
        return exposure
