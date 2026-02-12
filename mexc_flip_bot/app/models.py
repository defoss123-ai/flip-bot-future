"""Domain models for future trading logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TradeSignal:
    symbol: str
    side: str
    entry_price: float | None = None
