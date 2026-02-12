"""System health monitoring for API/Telegram/DB subsystems."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.database import Database
from app.mexc_client import MexcFuturesClient


@dataclass(slots=True)
class HealthStatus:
    api: str = "OK"
    telegram: str = "OK"
    db: str = "OK"
    api_consecutive_errors: int = 0


class HealthMonitor:
    def __init__(
        self,
        database: Database,
        mexc_client: MexcFuturesClient,
        logger: Any | None = None,
    ) -> None:
        self.database = database
        self.mexc_client = mexc_client
        self.logger = logger
        self.status = HealthStatus()
        self._telegram_ok = True
        self._pause_entries = False

    async def run_loop(self) -> None:
        while True:
            await self.check_once()
            await asyncio.sleep(5)

    async def check_once(self) -> None:
        await self._check_db()
        await self._check_api()
        self.status.telegram = "OK" if self._telegram_ok else "ERROR"

    async def _check_db(self) -> None:
        try:
            await self.database.healthcheck()
            self.status.db = "OK"
        except Exception as exc:  # noqa: BLE001
            self.status.db = "ERROR"
            self._log_error("HealthMonitor DB check failed: {}", exc)

    async def _check_api(self) -> None:
        try:
            ok = await self.mexc_client.test_connection()
            if ok:
                self.status.api = "OK"
                self.status.api_consecutive_errors = 0
                self._pause_entries = False
                return
            raise RuntimeError("API test_connection returned False")
        except Exception as exc:  # noqa: BLE001
            self.status.api = "ERROR"
            self.status.api_consecutive_errors += 1
            self._log_error("HealthMonitor API check failed: {}", exc)
            if self.status.api_consecutive_errors >= 3:
                self._pause_entries = True
                self._log_warning("HealthMonitor: API failed 3 times, pausing new entries")

    def set_telegram_ok(self, ok: bool) -> None:
        self._telegram_ok = ok
        self.status.telegram = "OK" if ok else "ERROR"

    def allow_new_entries(self) -> bool:
        return not self._pause_entries

    def snapshot(self) -> dict[str, str | int]:
        return {
            "api": self.status.api,
            "telegram": self.status.telegram,
            "db": self.status.db,
            "api_consecutive_errors": self.status.api_consecutive_errors,
            "entries": "PAUSED" if self._pause_entries else "RUNNING",
        }

    def _log_warning(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "warning"):
            self.logger.warning(message, *args)
        elif self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(message, *args)

    def _log_error(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "error"):
            self.logger.error(message, *args)
