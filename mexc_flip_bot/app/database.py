"""Async SQLite storage powered by SQLAlchemy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, desc, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


class SignalORM(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    raw_text: Mapped[str] = mapped_column(String, nullable=False)
    symbol: Mapped[str | None] = mapped_column(String, nullable=True)
    dex_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    mexc_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    align_age_sec: Mapped[int | None] = mapped_column(Integer, nullable=True)
    direction: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="new", nullable=False)
    hash: Mapped[str] = mapped_column(String, unique=True, nullable=False)


class OrderORM(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    qty: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String, default="new", nullable=False)
    exchange_order_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)


class TradeORM(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    symbol: Mapped[str] = mapped_column(String, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    usdt_amount: Mapped[float] = mapped_column(Float, nullable=False)
    leverage: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    sl_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl_usdt: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, default="opening", nullable=False)
    close_reason: Mapped[str | None] = mapped_column(String, nullable=True)


class ProfileORM(Base):
    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    data_json: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


@dataclass(slots=True)
class TradeStats:
    total_trades: int
    winrate: float
    pnl_day: float
    pnl_week: float
    pnl_month: float


@dataclass(slots=True)
class OrderRecord:
    id: int
    created_at: datetime
    symbol: str
    side: str
    type: str
    price: float | None
    qty: float
    status: str
    exchange_order_id: str | None
    reason: str | None


@dataclass(slots=True)
class SignalRecord:
    id: int
    created_at: datetime
    raw_text: str
    symbol: str | None
    dex_price: float | None
    mexc_price: float | None
    spread_percent: float | None
    liquidity_usd: float | None
    align_age_sec: int | None
    direction: str | None
    status: str
    hash: str


@dataclass(slots=True)
class TradeRecord:
    id: int
    created_at: datetime
    symbol: str
    direction: str
    usdt_amount: float
    leverage: int
    entry_price: float | None
    exit_price: float | None
    tp_price: float | None
    sl_price: float | None
    pnl_usdt: float | None
    status: str
    close_reason: str | None


@dataclass(slots=True)
class ProfileRecord:
    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    data_json: str
    is_active: bool


class Database:
    """Persistence layer for bot entities and statistics."""

    def __init__(self, db_path: str | Path = "data/bot.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine: AsyncEngine = create_async_engine(f"sqlite+aiosqlite:///{path}", echo=False)
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init_db(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def healthcheck(self) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(select(func.count(SignalORM.id)).limit(1))
            _ = result.scalar_one_or_none()
        return True


    async def signal_hash_exists(self, signal_hash: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(select(SignalORM.id).where(SignalORM.hash == signal_hash).limit(1))
            return result.scalar_one_or_none() is not None

    async def add_signal(self, **signal_data: Any) -> int | None:
        signal = SignalORM(**signal_data)
        async with self._session_factory() as session:
            session.add(signal)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return None
            await session.refresh(signal)
        return signal.id

    async def mark_signal_status(self, signal_id: int, status: str) -> bool:
        async with self._session_factory() as session:
            signal = await session.get(SignalORM, signal_id)
            if signal is None:
                return False
            signal.status = status
            await session.commit()
        return True


    async def get_next_signal(self, status: str = "new") -> SignalRecord | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SignalORM).where(SignalORM.status == status).order_by(SignalORM.created_at.asc()).limit(1)
            )
            row = result.scalars().first()
            if row is None:
                return None
            return SignalRecord(
                id=row.id,
                created_at=row.created_at,
                raw_text=row.raw_text,
                symbol=row.symbol,
                dex_price=row.dex_price,
                mexc_price=row.mexc_price,
                spread_percent=row.spread_percent,
                liquidity_usd=row.liquidity_usd,
                align_age_sec=row.align_age_sec,
                direction=row.direction,
                status=row.status,
                hash=row.hash,
            )

    async def list_recent_signals(self, limit: int = 10) -> list[SignalRecord]:
        async with self._session_factory() as session:
            result = await session.execute(select(SignalORM).order_by(desc(SignalORM.created_at)).limit(limit))
            rows = result.scalars().all()
        return [
            SignalRecord(
                id=row.id,
                created_at=row.created_at,
                raw_text=row.raw_text,
                symbol=row.symbol,
                dex_price=row.dex_price,
                mexc_price=row.mexc_price,
                spread_percent=row.spread_percent,
                liquidity_usd=row.liquidity_usd,
                align_age_sec=row.align_age_sec,
                direction=row.direction,
                status=row.status,
                hash=row.hash,
            )
            for row in rows
        ]

    async def get_trade_by_id(self, trade_id: int) -> TradeRecord | None:
        async with self._session_factory() as session:
            row = await session.get(TradeORM, trade_id)
            if row is None:
                return None
            return TradeRecord(
                id=row.id,
                created_at=row.created_at,
                symbol=row.symbol,
                direction=row.direction,
                usdt_amount=row.usdt_amount,
                leverage=row.leverage,
                entry_price=row.entry_price,
                exit_price=row.exit_price,
                tp_price=row.tp_price,
                sl_price=row.sl_price,
                pnl_usdt=row.pnl_usdt,
                status=row.status,
                close_reason=row.close_reason,
            )

    async def get_latest_signal_price(self, symbol: str) -> float | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SignalORM.mexc_price)
                .where(SignalORM.symbol == symbol, SignalORM.mexc_price.is_not(None))
                .order_by(desc(SignalORM.created_at))
                .limit(1)
            )
            value = result.scalar_one_or_none()
            return float(value) if value is not None else None

    async def count_active_trades(self) -> int:
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count(TradeORM.id)).where(TradeORM.status.in_(["opening", "open", "closing"]))
            )
            return int(result.scalar_one() or 0)

    async def add_order(self, **order_data: Any) -> int:
        order = OrderORM(**order_data)
        async with self._session_factory() as session:
            session.add(order)
            await session.commit()
            await session.refresh(order)
        return order.id

    async def update_order_status(
        self,
        order_id: int,
        status: str,
        exchange_order_id: str | None = None,
        reason: str | None = None,
    ) -> bool:
        async with self._session_factory() as session:
            order = await session.get(OrderORM, order_id)
            if order is None:
                return False
            order.status = status
            if exchange_order_id is not None:
                order.exchange_order_id = exchange_order_id
            if reason is not None:
                order.reason = reason
            await session.commit()
        return True

    async def add_trade(self, **trade_data: Any) -> int:
        trade = TradeORM(**trade_data)
        async with self._session_factory() as session:
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
        return trade.id

    async def update_trade(self, trade_id: int, **update_data: Any) -> bool:
        async with self._session_factory() as session:
            trade = await session.get(TradeORM, trade_id)
            if trade is None:
                return False
            for field_name, value in update_data.items():
                if hasattr(trade, field_name):
                    setattr(trade, field_name, value)
            await session.commit()
        return True


    async def list_open_trades(self) -> list[TradeRecord]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(TradeORM).where(TradeORM.status.in_(["opening", "open", "closing"]))
                .order_by(TradeORM.created_at.asc())
            )
            rows = result.scalars().all()
        return [
            TradeRecord(
                id=row.id,
                created_at=row.created_at,
                symbol=row.symbol,
                direction=row.direction,
                usdt_amount=row.usdt_amount,
                leverage=row.leverage,
                entry_price=row.entry_price,
                exit_price=row.exit_price,
                tp_price=row.tp_price,
                sl_price=row.sl_price,
                pnl_usdt=row.pnl_usdt,
                status=row.status,
                close_reason=row.close_reason,
            )
            for row in rows
        ]

    async def list_recent_trades(self, limit: int = 200) -> list[TradeRecord]:
        async with self._session_factory() as session:
            result = await session.execute(select(TradeORM).order_by(desc(TradeORM.created_at)).limit(limit))
            rows = result.scalars().all()
        return [
            TradeRecord(
                id=row.id,
                created_at=row.created_at,
                symbol=row.symbol,
                direction=row.direction,
                usdt_amount=row.usdt_amount,
                leverage=row.leverage,
                entry_price=row.entry_price,
                exit_price=row.exit_price,
                tp_price=row.tp_price,
                sl_price=row.sl_price,
                pnl_usdt=row.pnl_usdt,
                status=row.status,
                close_reason=row.close_reason,
            )
            for row in rows
        ]



    async def list_closed_trades(self, since: datetime | None = None) -> list[TradeRecord]:
        async with self._session_factory() as session:
            query = select(TradeORM).where(TradeORM.status == "closed").order_by(TradeORM.created_at.asc())
            if since is not None:
                query = query.where(TradeORM.created_at >= since)
            result = await session.execute(query)
            rows = result.scalars().all()
        return [
            TradeRecord(
                id=row.id,
                created_at=row.created_at,
                symbol=row.symbol,
                direction=row.direction,
                usdt_amount=row.usdt_amount,
                leverage=row.leverage,
                entry_price=row.entry_price,
                exit_price=row.exit_price,
                tp_price=row.tp_price,
                sl_price=row.sl_price,
                pnl_usdt=row.pnl_usdt,
                status=row.status,
                close_reason=row.close_reason,
            )
            for row in rows
        ]

    async def list_orders_by_status(self, status: str = "placed", limit: int = 500) -> list[OrderRecord]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(OrderORM)
                .where(OrderORM.status == status)
                .order_by(desc(OrderORM.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
        return [
            OrderRecord(
                id=row.id,
                created_at=row.created_at,
                symbol=row.symbol,
                side=row.side,
                type=row.type,
                price=row.price,
                qty=row.qty,
                status=row.status,
                exchange_order_id=row.exchange_order_id,
                reason=row.reason,
            )
            for row in rows
        ]

    async def get_open_trade_by_symbol(self, symbol: str) -> TradeRecord | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(TradeORM)
                .where(TradeORM.symbol == symbol, TradeORM.status.in_(["opening", "open", "closing"]))
                .order_by(TradeORM.created_at.asc())
                .limit(1)
            )
            row = result.scalars().first()
            if row is None:
                return None
            return TradeRecord(
                id=row.id,
                created_at=row.created_at,
                symbol=row.symbol,
                direction=row.direction,
                usdt_amount=row.usdt_amount,
                leverage=row.leverage,
                entry_price=row.entry_price,
                exit_price=row.exit_price,
                tp_price=row.tp_price,
                sl_price=row.sl_price,
                pnl_usdt=row.pnl_usdt,
                status=row.status,
                close_reason=row.close_reason,
            )

    async def list_recent_orders(self, limit: int = 200) -> list[OrderRecord]:
        async with self._session_factory() as session:
            result = await session.execute(select(OrderORM).order_by(desc(OrderORM.created_at)).limit(limit))
            rows = result.scalars().all()
        return [
            OrderRecord(
                id=row.id,
                created_at=row.created_at,
                symbol=row.symbol,
                side=row.side,
                type=row.type,
                price=row.price,
                qty=row.qty,
                status=row.status,
                exchange_order_id=row.exchange_order_id,
                reason=row.reason,
            )
            for row in rows
        ]

    async def create_profile(self, name: str, data_json: str) -> int:
        now = datetime.now(UTC)
        profile = ProfileORM(name=name, data_json=data_json, created_at=now, updated_at=now, is_active=False)
        async with self._session_factory() as session:
            session.add(profile)
            await session.commit()
            await session.refresh(profile)
        return profile.id

    async def list_profiles(self) -> list[ProfileRecord]:
        async with self._session_factory() as session:
            result = await session.execute(select(ProfileORM).order_by(ProfileORM.id.asc()))
            rows = result.scalars().all()
        return [
            ProfileRecord(
                id=row.id,
                name=row.name,
                created_at=row.created_at,
                updated_at=row.updated_at,
                data_json=row.data_json,
                is_active=bool(row.is_active),
            )
            for row in rows
        ]

    async def get_profile(self, profile_id: int) -> ProfileRecord | None:
        async with self._session_factory() as session:
            row = await session.get(ProfileORM, profile_id)
            if row is None:
                return None
            return ProfileRecord(
                id=row.id,
                name=row.name,
                created_at=row.created_at,
                updated_at=row.updated_at,
                data_json=row.data_json,
                is_active=bool(row.is_active),
            )

    async def update_profile(self, profile_id: int, data_json: str) -> bool:
        async with self._session_factory() as session:
            row = await session.get(ProfileORM, profile_id)
            if row is None:
                return False
            row.data_json = data_json
            row.updated_at = datetime.now(UTC)
            await session.commit()
            return True

    async def delete_profile(self, profile_id: int) -> bool:
        async with self._session_factory() as session:
            row = await session.get(ProfileORM, profile_id)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def set_active_profile(self, profile_id: int) -> bool:
        async with self._session_factory() as session:
            profile = await session.get(ProfileORM, profile_id)
            if profile is None:
                return False
            await session.execute(update(ProfileORM).values(is_active=False))
            profile.is_active = True
            profile.updated_at = datetime.now(UTC)
            await session.commit()
            return True

    async def get_active_profile(self) -> ProfileRecord | None:
        async with self._session_factory() as session:
            result = await session.execute(select(ProfileORM).where(ProfileORM.is_active.is_(True)).limit(1))
            row = result.scalars().first()
            if row is None:
                return None
            return ProfileRecord(
                id=row.id,
                name=row.name,
                created_at=row.created_at,
                updated_at=row.updated_at,
                data_json=row.data_json,
                is_active=bool(row.is_active),
            )

    async def stats(self) -> TradeStats:
        now = datetime.now(UTC)
        day_start = now - timedelta(days=1)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)

        async with self._session_factory() as session:
            total_trades_result = await session.execute(select(func.count(TradeORM.id)))
            total_trades = int(total_trades_result.scalar_one() or 0)

            win_result = await session.execute(
                select(func.count(TradeORM.id)).where(TradeORM.pnl_usdt.is_not(None), TradeORM.pnl_usdt > 0)
            )
            wins = int(win_result.scalar_one() or 0)

            pnl_day = await self._sum_pnl_since(session, day_start)
            pnl_week = await self._sum_pnl_since(session, week_start)
            pnl_month = await self._sum_pnl_since(session, month_start)

        winrate = (wins / total_trades * 100.0) if total_trades else 0.0
        return TradeStats(
            total_trades=total_trades,
            winrate=round(winrate, 2),
            pnl_day=round(pnl_day, 6),
            pnl_week=round(pnl_week, 6),
            pnl_month=round(pnl_month, 6),
        )

    async def _sum_pnl_since(self, session: AsyncSession, since: datetime) -> float:
        result = await session.execute(
            select(func.coalesce(func.sum(TradeORM.pnl_usdt), 0.0)).where(
                TradeORM.created_at >= since,
                TradeORM.pnl_usdt.is_not(None),
            )
        )
        return float(result.scalar_one() or 0.0)
