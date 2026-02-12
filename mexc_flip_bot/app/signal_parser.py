"""Parser for incoming Telegram signal messages."""

from __future__ import annotations

import hashlib
import re


_NUMBER_RE = r"([-+]?\d+(?:[.,]\d+)?)"


class SignalParser:
    """Parse raw signal text and evaluate basic filters."""

    def __init__(self, min_spread_percent: float, min_liquidity_usd: float, max_align_age_sec: int) -> None:
        self.min_spread_percent = min_spread_percent
        self.min_liquidity_usd = min_liquidity_usd
        self.max_align_age_sec = max_align_age_sec

    def normalize_text(self, raw_text: str) -> str:
        text = raw_text.replace("\r", "")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines).lower()

    def compute_hash(self, raw_text: str) -> str:
        normalized = self.normalize_text(raw_text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def parse(self, raw_text: str) -> dict[str, float | int | str | None]:
        text = raw_text or ""

        return {
            "dex_price": self._extract_price(text, r"price\s*dex[^\n]*\$\s*" + _NUMBER_RE),
            "mexc_price": self._extract_price(text, r"price\s*mexc[^\n]*\$\s*" + _NUMBER_RE),
            "spread_percent": self._extract_price(text, r"spread\s*:\s*" + _NUMBER_RE + r"\s*%"),
            "liquidity_usd": self._extract_liquidity(text),
            "align_age_sec": self._extract_align_age(text),
            "symbol": self._extract_symbol(text),
            "direction": None,
        }

    def passes_filters(self, signal: dict[str, float | int | str | None]) -> bool:
        spread = signal.get("spread_percent")
        liquidity = signal.get("liquidity_usd")
        align_age = signal.get("align_age_sec")

        if spread is None or float(spread) < self.min_spread_percent:
            return False
        if liquidity is None or float(liquidity) < self.min_liquidity_usd:
            return False
        if align_age is None or int(align_age) > self.max_align_age_sec:
            return False
        return True

    def _extract_price(self, text: str, pattern: str) -> float | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        value = match.group(1).replace(",", ".")
        try:
            return float(value)
        except ValueError:
            return None

    def _extract_liquidity(self, text: str) -> float | None:
        match = re.search(r"liquidity\s*:\s*\$?\s*([0-9.,]+)\s*([kmb])?", text, flags=re.IGNORECASE)
        if not match:
            return None

        base_value = match.group(1).replace(",", "")
        try:
            value = float(base_value)
        except ValueError:
            return None

        multiplier = (match.group(2) or "").lower()
        if multiplier == "k":
            value *= 1_000
        elif multiplier == "m":
            value *= 1_000_000
        elif multiplier == "b":
            value *= 1_000_000_000
        return value

    def _extract_align_age(self, text: str) -> int | None:
        match = re.search(r"avg\s*align\s*time\s*/\s*" + _NUMBER_RE + r"\s*(sec|s|min|m)?", text, flags=re.IGNORECASE)
        if not match:
            return None

        value_raw = match.group(1).replace(",", ".")
        try:
            value = float(value_raw)
        except ValueError:
            return None

        unit = (match.group(2) or "sec").lower()
        if unit in {"min", "m"}:
            value *= 60
        return int(value)

    def _extract_symbol(self, text: str) -> str | None:
        hash_match = re.search(r"#([A-Z0-9]{2,12})", text)
        if hash_match:
            return hash_match.group(1)

        pair_match = re.search(r"\b([A-Z0-9]{2,12})\s*/\s*USDT\b", text)
        if pair_match:
            return pair_match.group(1)
        return None
