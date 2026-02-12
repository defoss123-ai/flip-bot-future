"""Minimal smoke-test for MEXC client connectivity."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from app.config import load_config
from app.database import Database
from app.mexc_client import MexcFuturesClient


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config.yml")

    runtime_path = root / "data" / "config_runtime.yml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    mode = str((runtime or {}).get("mode") or config.mode)
    mexc_runtime = (runtime or {}).get("mexc") or {}

    db = Database(root / "data" / "bot.db")
    await db.init_db()

    client = MexcFuturesClient(
        database=db,
        mode=mode,
        api_key=str(mexc_runtime.get("api_key") or config.mexc.api_key),
        api_secret=str(mexc_runtime.get("api_secret") or config.mexc.api_secret),
    )

    ok = await client.test_connection()
    print("OK" if ok else "FAIL")

    contracts = await client.get_contracts()
    print("contracts:", contracts[:10])

    symbol = "BTC_USDT"
    try:
        last_price = await client.get_last_price(symbol)
    except Exception:  # noqa: BLE001
        symbol = "BTCUSDT"
        last_price = await client.get_last_price(symbol)
    print(f"last_price {symbol}: {last_price}")


if __name__ == "__main__":
    asyncio.run(main())
