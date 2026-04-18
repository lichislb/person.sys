"""SQLite event store for candidate events."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


class EventStore:
    """Minimal SQLite persistence for candidate events."""

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path or "data/events.db")

    def init_db(self) -> None:
        """Create database and events table if not exists."""
        self._ensure_parent_dir()
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    zone_name TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    "current_time" REAL NOT NULL,
                    duration REAL NOT NULL,
                    confidence_local REAL NOT NULL,
                    review_status TEXT,
                    confidence_vlm REAL,
                    risk_level TEXT,
                    explanation TEXT,
                    final_decision TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_optional_columns(conn)
            conn.commit()
        finally:
            conn.close()

    def insert_event(self, event: dict[str, Any]) -> None:
        """Insert one event with simple deduplication by event_id."""
        if not isinstance(event, dict):
            return

        event_id = str(event.get("event_id") or self._make_event_id(event))
        if not event_id:
            return
        if self.event_exists(event_id):
            return

        event_type = str(event.get("event_type", "unknown"))
        zone_name = str(event.get("zone_name", "unknown"))
        try:
            track_id = int(event.get("track_id"))
            start_time = float(event.get("start_time"))
            current_time = float(event.get("current_time"))
            duration = float(event.get("duration"))
            confidence_local = float(event.get("confidence_local", 0.0))
        except (TypeError, ValueError):
            return
        review_status = event.get("review_status")
        confidence_vlm = event.get("confidence_vlm")
        risk_level = event.get("risk_level")
        explanation = event.get("explanation")
        final_decision = event.get("final_decision")

        created_at = datetime.now(timezone.utc).isoformat()

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    event_id, event_type, track_id, zone_name,
                    start_time, "current_time", duration, confidence_local,
                    review_status, confidence_vlm, risk_level, explanation, final_decision,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    track_id,
                    zone_name,
                    start_time,
                    current_time,
                    duration,
                    confidence_local,
                    None if review_status is None else str(review_status),
                    self._safe_float(confidence_vlm),
                    None if risk_level is None else str(risk_level),
                    None if explanation is None else str(explanation),
                    None if final_decision is None else str(final_decision),
                    created_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """List events ordered by creation time descending."""
        try:
            lim = max(1, int(limit))
        except (TypeError, ValueError):
            lim = 100

        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT
                    event_id, event_type, track_id, zone_name,
                    start_time, "current_time", duration, confidence_local,
                    review_status, confidence_vlm, risk_level, explanation, final_decision,
                    created_at
                FROM events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_event_dict(r) for r in rows]

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Get one event by event_id."""
        eid = str(event_id or "").strip()
        if not eid:
            return None
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT
                    event_id, event_type, track_id, zone_name,
                    start_time, "current_time", duration, confidence_local,
                    review_status, confidence_vlm, risk_level, explanation, final_decision,
                    created_at
                FROM events
                WHERE event_id = ?
                """,
                (eid,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return self._row_to_event_dict(row)

    def event_exists(self, event_id: str) -> bool:
        """Check event existence by event_id."""
        eid = str(event_id or "").strip()
        if not eid:
            return False
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM events WHERE event_id = ? LIMIT 1",
                (eid,),
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _ensure_optional_columns(conn: sqlite3.Connection) -> None:
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(events)").fetchall()
        }
        optional_defs = {
            "review_status": "TEXT",
            "confidence_vlm": "REAL",
            "risk_level": "TEXT",
            "explanation": "TEXT",
            "final_decision": "TEXT",
        }
        for col, col_type in optional_defs.items():
            if col not in existing_cols:
                conn.execute(f"ALTER TABLE events ADD COLUMN {col} {col_type}")

    def _make_event_id(self, event: dict[str, Any]) -> str:
        """Generate stable and readable event_id."""
        event_type = str(event.get("event_type", "unknown")).lower()
        track_id = str(event.get("track_id", "x"))
        zone_name = str(event.get("zone_name", "zone")).lower()
        try:
            start_time = f"{float(event.get('start_time')):.3f}"
        except (TypeError, ValueError):
            start_time = "0.000"

        base = f"{event_type}|{track_id}|{zone_name}|{start_time}"
        digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        return f"{event_type}_{track_id}_{zone_name}_{start_time}_{digest}"

    @staticmethod
    def _row_to_event_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "track_id": int(row["track_id"]),
            "zone_name": row["zone_name"],
            "start_time": float(row["start_time"]),
            "current_time": float(row["current_time"]),
            "duration": float(row["duration"]),
            "confidence_local": float(row["confidence_local"]),
            "review_status": row["review_status"],
            "confidence_vlm": EventStore._safe_float(row["confidence_vlm"]),
            "risk_level": row["risk_level"] if row["risk_level"] is not None else "unknown",
            "explanation": row["explanation"] if row["explanation"] is not None else "",
            "final_decision": row["final_decision"]
            if row["final_decision"] is not None
            else "local_only",
            "created_at": row["created_at"],
        }

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
