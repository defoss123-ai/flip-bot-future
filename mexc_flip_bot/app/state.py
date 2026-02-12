"""Runtime state helpers for bot lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class BotState:
    started_at: datetime = field(default_factory=datetime.utcnow)
    running: bool = False
    active_profile_id: int = 0
    recovered: bool = False
    revert_at: datetime | None = None
    revert_to: str | None = None


def _state_path(root_dir: Path | None = None) -> Path:
    base = root_dir or Path(__file__).resolve().parents[1]
    return base / "data" / "state.yml"


def load_state(root_dir: Path | None = None) -> BotState:
    path = _state_path(root_dir)
    if not path.exists():
        return BotState(running=False, active_profile_id=0, recovered=False)

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    started_at = raw.get("started_at")
    revert_at_raw = raw.get("revert_at")
    try:
        dt = datetime.fromisoformat(started_at) if started_at else datetime.utcnow()
    except ValueError:
        dt = datetime.utcnow()

    revert_at = None
    if revert_at_raw:
        try:
            revert_at = datetime.fromisoformat(str(revert_at_raw))
        except ValueError:
            revert_at = None

    return BotState(
        started_at=dt,
        running=bool(raw.get("running", False)),
        active_profile_id=int(raw.get("active_profile_id", 0) or 0),
        recovered=bool(raw.get("recovered", False)),
        revert_at=revert_at,
        revert_to=raw.get("revert_to"),
    )


def save_state(
    running: bool | None = None,
    root_dir: Path | None = None,
    recovered: bool | None = None,
    active_profile_id: int | None = None,
    revert_at: datetime | None | str = None,
    revert_to: str | None = None,
) -> BotState:
    path = _state_path(root_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    current = load_state(root_dir)

    new_revert_at = current.revert_at
    if revert_at is not None:
        if isinstance(revert_at, str):
            try:
                new_revert_at = datetime.fromisoformat(revert_at)
            except ValueError:
                new_revert_at = None
        else:
            new_revert_at = revert_at

    new_revert_to = current.revert_to if revert_to is None else revert_to

    state = BotState(
        started_at=current.started_at,
        running=current.running if running is None else running,
        active_profile_id=current.active_profile_id if active_profile_id is None else int(active_profile_id),
        recovered=current.recovered if recovered is None else recovered,
        revert_at=new_revert_at,
        revert_to=new_revert_to,
    )
    payload: dict[str, Any] = {
        "started_at": state.started_at.isoformat(),
        "running": state.running,
        "active_profile_id": state.active_profile_id,
        "recovered": state.recovered,
        "revert_at": state.revert_at.isoformat() if state.revert_at else None,
        "revert_to": state.revert_to,
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return state
