"""Strategy behavior controller for conservative/aggressive modes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from app.config_resolver import EffectiveSettings
from app.database import SignalRecord, TradeRecord


@dataclass(slots=True)
class BeforeEntryDecision:
    allowed: bool
    reason: str


class StrategyController:
    def on_before_entry(
        self,
        signal: SignalRecord,
        effective: EffectiveSettings,
        loss_guard_until: datetime | None,
        api_guard_until: datetime | None,
    ) -> BeforeEntryDecision:
        now = datetime.now(UTC)
        if loss_guard_until is not None and now < loss_guard_until:
            return BeforeEntryDecision(False, "loss_streak_guard")
        if api_guard_until is not None and now < api_guard_until:
            return BeforeEntryDecision(False, "panic_on_api_errors")

        spread = float(signal.spread_percent or 0.0)
        liquidity = float(signal.liquidity_usd or 0.0)
        align_age = int(signal.align_age_sec or 0)
        if spread < effective.min_spread_percent:
            return BeforeEntryDecision(False, "strategy_spread_filter")
        if liquidity < effective.min_liquidity_usd:
            return BeforeEntryDecision(False, "strategy_liquidity_filter")
        if align_age > effective.max_align_age_sec:
            return BeforeEntryDecision(False, "strategy_align_filter")
        return BeforeEntryDecision(True, "ok")

    def get_entry_price_adjustment(self, direction: str, effective: EffectiveSettings) -> float:
        if effective.strategy_mode != "conservative":
            return 1.0
        return 0.9995 if direction == "long" else 1.0005

    def on_after_open(self, _trade: TradeRecord, _effective: EffectiveSettings) -> None:
        return None

    def should_time_exit(
        self,
        trade: TradeRecord,
        age_sec: float,
        current_pnl_percent: float,
        effective: EffectiveSettings,
    ) -> bool:
        if effective.strategy_mode != "aggressive":
            return False
        if trade.status != "open":
            return False
        return age_sec >= 45 and current_pnl_percent < 0.05

    def on_error_burst(self, effective: EffectiveSettings, error_count_in_2m: int) -> bool:
        return effective.strategy_mode == "aggressive" and error_count_in_2m >= 5
