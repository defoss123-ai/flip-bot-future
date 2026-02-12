"""Direction engine with confidence and risk-aware filters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DirectionSignal:
    dex_price: float | None
    mexc_price: float | None
    spread_percent: float | None
    liquidity_usd: float | None


@dataclass(slots=True)
class MarketData:
    funding_rate: float | None = None
    volatility_1m: float | None = None
    base_leverage: int = 1
    adjusted_leverage: int | None = None
    confidence_score: float = 0.0


@dataclass(slots=True)
class DirectionDecision:
    direction: str | None
    confidence_score: float
    adjusted_leverage: int | None
    reason: str


class DirectionEngine:
    """Evaluates direction with confidence and market-aware filters."""

    def __init__(
        self,
        min_spread_percent: float,
        min_liquidity_usd: float,
        funding_threshold: float = 0.005,
        volatility_limit: float = 0.03,
        min_confidence: int = 50,
    ) -> None:
        self.min_spread_percent = max(min_spread_percent, 0.0)
        self.min_liquidity_usd = max(min_liquidity_usd, 0.0)
        self.funding_threshold = abs(funding_threshold)
        self.volatility_limit = max(volatility_limit, 0.0)
        self.min_confidence = max(0, min(100, int(min_confidence)))

    def determine_direction(self, signal: DirectionSignal, market_data: MarketData) -> str | None:
        decision = self.evaluate(signal, market_data)
        market_data.confidence_score = decision.confidence_score
        market_data.adjusted_leverage = decision.adjusted_leverage
        return decision.direction

    def evaluate(self, signal: DirectionSignal, market_data: MarketData) -> DirectionDecision:
        if signal.dex_price is None or signal.mexc_price is None or signal.mexc_price <= 0:
            return DirectionDecision(None, 0.0, None, "missing_price")

        spread = float(signal.spread_percent or 0.0)
        if spread < self.min_spread_percent:
            return DirectionDecision(None, 0.0, None, "spread_too_low")

        deviation = abs(signal.dex_price - signal.mexc_price) / signal.mexc_price
        deviation_percent = deviation * 100.0
        if deviation_percent < 0.3 * self.min_spread_percent:
            return DirectionDecision(None, 0.0, None, "deviation_too_low")

        direction = "long" if signal.dex_price > signal.mexc_price else "short"

        funding = market_data.funding_rate
        if funding is not None:
            if funding > self.funding_threshold and direction == "long":
                return DirectionDecision(None, 0.0, None, "funding_blocks_long")
            if funding < -self.funding_threshold and direction == "short":
                return DirectionDecision(None, 0.0, None, "funding_blocks_short")

        confidence = self._confidence_score(spread, deviation_percent, float(signal.liquidity_usd or 0.0))
        if confidence < float(self.min_confidence):
            return DirectionDecision(None, confidence, None, "confidence_too_low")

        adjusted_lev = self._adjust_leverage(market_data.base_leverage, market_data.volatility_1m)
        return DirectionDecision(direction, confidence, adjusted_lev, "ok")

    def _adjust_leverage(self, base_leverage: int, volatility_1m: float | None) -> int | None:
        if volatility_1m is None:
            return None
        if volatility_1m <= self.volatility_limit:
            return None
        reduced = max(1, int(base_leverage * 0.5))
        return reduced if reduced < base_leverage else None

    def _confidence_score(self, spread_percent: float, deviation_percent: float, liquidity_usd: float) -> float:
        spread_base = self.min_spread_percent if self.min_spread_percent > 0 else 1.0
        spread_score = min(1.0, spread_percent / (spread_base * 2.0))

        liq_base = self.min_liquidity_usd if self.min_liquidity_usd > 0 else max(liquidity_usd, 1.0)
        liquidity_score = min(1.0, liquidity_usd / (liq_base * 1.5 if liq_base > 0 else 1.0))

        deviation_floor = max(0.3 * self.min_spread_percent, 0.0001)
        deviation_score = min(1.0, deviation_percent / (deviation_floor * 3.0))

        score = (spread_score * 0.4 + liquidity_score * 0.3 + deviation_score * 0.3) * 100.0
        return round(max(0.0, min(100.0, score)), 2)
