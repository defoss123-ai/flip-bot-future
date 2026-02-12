"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class MexcConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key: str
    api_secret: str


class TradingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usdt_amount: float = Field(gt=0)
    leverage: int = Field(ge=1)
    max_positions: int = Field(ge=1)
    cooldown_sec: int = Field(ge=0)
    default_entry_type: str = Field(pattern=r"^(market|limit)$")
    leverage_mode: str = Field(default="fixed", pattern=r"^(fixed|adaptive)$")
    max_leverage: int = Field(default=25, ge=1)


class OrdersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_timeout_sec: int = Field(ge=1)
    retry_mode: str = Field(pattern=r"^(cancel|cancel_then_market|cancel_then_relimit)$")


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tp_percent: float = Field(gt=0)
    sl_percent: float = Field(gt=0)
    max_total_exposure_usdt: float = Field(gt=0)
    max_funding_rate_percent: float = Field(ge=0)
    spread_anomaly_multiplier: float = Field(gt=0)


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_spread_percent: float = Field(ge=0)
    min_liquidity_usd: float = Field(ge=0)
    max_align_age_sec: int = Field(ge=0)


class StrategyProfileFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_spread_percent: float = Field(ge=0)
    min_liquidity_usd: float = Field(ge=0)
    max_align_age_sec: int = Field(ge=0)


class StrategyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: int = Field(ge=0, le=100)
    tp_percent: float = Field(gt=0)
    sl_percent: float = Field(gt=0)
    cooldown_sec: int = Field(ge=0)
    max_positions: int = Field(ge=1)
    entry_type: str = Field(pattern=r"^(market|limit)$")
    entry_timeout_sec: int = Field(ge=1)
    retry_mode: str = Field(pattern=r"^(cancel|cancel_then_market|cancel_then_relimit)$")
    leverage_mode: str = Field(pattern=r"^(fixed|adaptive)$")
    max_leverage: int = Field(ge=1)
    filters: StrategyProfileFilters


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = Field(default="conservative", pattern=r"^(conservative|aggressive)$")
    conservative: StrategyProfile = Field(
        default_factory=lambda: StrategyProfile(
            min_confidence=65,
            tp_percent=0.35,
            sl_percent=0.25,
            cooldown_sec=90,
            max_positions=2,
            entry_type="limit",
            entry_timeout_sec=60,
            retry_mode="cancel_then_relimit",
            leverage_mode="adaptive",
            max_leverage=12,
            filters=StrategyProfileFilters(
                min_spread_percent=0.8,
                min_liquidity_usd=60000,
                max_align_age_sec=15,
            ),
        )
    )
    aggressive: StrategyProfile = Field(
        default_factory=lambda: StrategyProfile(
            min_confidence=50,
            tp_percent=0.20,
            sl_percent=0.35,
            cooldown_sec=15,
            max_positions=5,
            entry_type="market",
            entry_timeout_sec=20,
            retry_mode="cancel_then_market",
            leverage_mode="adaptive",
            max_leverage=25,
            filters=StrategyProfileFilters(
                min_spread_percent=0.5,
                min_liquidity_usd=25000,
                max_align_age_sec=30,
            ),
        )
    )


class TelegramConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_telethon: bool
    api_id: int
    api_hash: str
    session_name: str
    source_chat: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = Field(pattern=r"^(paper|live)$")
    mexc: MexcConfig
    trading: TradingConfig
    orders: OrdersConfig
    risk: RiskConfig
    filters: FiltersConfig
    telegram: TelegramConfig
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)


def load_config(path: str | Path = "config.yml") -> AppConfig:
    """Load configuration from YAML file and validate schema."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_path}' not found. Copy config.yml.example to config.yml first."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw_data = yaml.safe_load(fh) or {}

    try:
        return AppConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ValueError(f"Invalid config '{config_path}': {exc}") from exc
