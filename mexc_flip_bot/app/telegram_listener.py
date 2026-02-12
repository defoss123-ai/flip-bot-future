"""Telegram listener built on top of Telethon with reconnect support."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class TelegramListener:
    """Listens to new messages from a configured Telegram chat and pushes them to a queue."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str,
        source_chat: str,
        message_queue: asyncio.Queue[str],
        logger: Any | None = None,
    ) -> None:
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.source_chat = source_chat
        self.message_queue = message_queue
        self.logger = logger

        self._client = None
        self._running = False
        self._handler: Callable[[object], Awaitable[None]] | None = None

    async def start(self) -> None:
        """Start Telegram client, auto-reconnecting on disconnect/errors."""
        if self._running:
            return

        try:
            from telethon import TelegramClient, events
            from telethon.errors import FloodWaitError
        except ImportError as exc:
            raise RuntimeError(
                "Telethon is not installed. Add 'telethon' to requirements and install dependencies."
            ) from exc

        self._running = True
        retry_delay = 1

        while self._running:
            client = TelegramClient(self.session_name, self.api_id, self.api_hash)

            async def _event_handler(event: object) -> None:
                text = getattr(event, "raw_text", "") or ""
                await self.on_message(text)

            self._handler = _event_handler
            client.add_event_handler(self._handler, events.NewMessage(chats=self.source_chat))
            self._client = client

            try:
                await client.start()
                retry_delay = 1
                self._log_info("Telegram listener connected chat={}", self.source_chat)
                await client.run_until_disconnected()
                if self._running:
                    self._log_warning("Telegram disconnected, reconnecting in {}s", retry_delay)
                    await asyncio.sleep(retry_delay)
            except FloodWaitError as exc:
                wait_sec = int(getattr(exc, "seconds", 1) or 1)
                self._log_warning("Telegram FloodWaitError: wait {}s", wait_sec)
                await asyncio.sleep(wait_sec)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log_error("Telegram listener error: {}", exc)
                await asyncio.sleep(retry_delay)
            finally:
                await client.disconnect()

            retry_delay = min(retry_delay * 2, 30)

    async def on_message(self, text: str) -> None:
        """Put incoming message text into shared queue."""
        await self.message_queue.put(text)

    async def stop(self) -> None:
        """Gracefully stop Telegram client."""
        self._running = False
        if self._client is not None:
            await self._client.disconnect()

    def _log_info(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(message, *args)

    def _log_warning(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "warning"):
            self.logger.warning(message, *args)
        elif self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(message, *args)

    def _log_error(self, message: str, *args: object) -> None:
        if self.logger is not None and hasattr(self.logger, "error"):
            self.logger.error(message, *args)
