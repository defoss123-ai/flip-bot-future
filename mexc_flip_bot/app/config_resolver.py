"""Strategy-aware config resolver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import AppConfig, StrategyProfile


@dataclass(slots=True)
class EffectiveSettings:
    strategy_mode: str
    min_confidence: int
    tp_percent: float
    sl_percent: float
    cooldown_sec: int
    max_positions: int
    entry_type: str
    entry_timeout_sec: int
    retry_mode: str
    leverage_mode: str
    max_leverage: int
    min_spread_percent: float
    min_liquidity_usd: float
    max_align_age_sec: int


class ConfigResolver:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def _effective_from_config(config: AppConfig) -> EffectiveSettings:
        profile: StrategyProfile
        mode = config.strategy.mode
        if mode == "aggressive":
            profile = config.strategy.aggressive
        else:
            mode = "conservative"
            profile = config.strategy.conservative

        return EffectiveSettings(
            strategy_mode=mode,
            min_confidence=profile.min_confidence,
            tp_percent=profile.tp_percent,
            sl_percent=profile.sl_percent,
            cooldown_sec=profile.cooldown_sec,
            max_positions=profile.max_positions,
            entry_type=profile.entry_type,
            entry_timeout_sec=profile.entry_timeout_sec,
            retry_mode=profile.retry_mode,
            leverage_mode=profile.leverage_mode or config.trading.leverage_mode,
            max_leverage=profile.max_leverage if profile.max_leverage > 0 else config.trading.max_leverage,
            min_spread_percent=profile.filters.min_spread_percent,
            min_liquidity_usd=profile.filters.min_liquidity_usd,
            max_align_age_sec=profile.filters.max_align_age_sec,
        )

    @staticmethod
    def _merge_runtime(config: AppConfig, runtime: dict[str, Any]) -> AppConfig:
        payload = config.model_dump()
        payload["mode"] = runtime.get("mode", payload["mode"])
        for nested in ["trading", "orders", "risk", "filters", "strategy"]:
            payload[nested].update(runtime.get(nested) or {})
        return AppConfig.model_validate(payload)

    def get_effective_settings(self, runtime_settings: dict[str, Any] | None = None) -> EffectiveSettings:
        if runtime_settings is None:
            return self._effective_from_config(self.config)
        merged = self._merge_runtime(self.config, runtime_settings)
        return self._effective_from_config(merged)
