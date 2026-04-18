"""Track-level state manager for event rule engines.

This module only maintains cross-frame states. It does not contain event
business logic itself.
"""

from __future__ import annotations

from typing import Any


class TrackStateManager:
    """Manage per-track and per-zone runtime states."""

    def __init__(self) -> None:
        self._tracks: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        """Clear all runtime states."""
        self._tracks.clear()

    def update_track(
        self,
        track_id: int,
        timestamp: float,
        bbox: list[float] | list[int] | None,
        center: list[float] | list[int] | None,
    ) -> None:
        """Update general track status."""
        try:
            tid = int(track_id)
            ts = float(timestamp)
        except (TypeError, ValueError):
            return

        track_state = self._ensure_track_state(tid)
        track_state["last_timestamp"] = ts
        track_state["last_bbox"] = self._normalize_bbox(bbox)
        track_state["last_center"] = self._normalize_center(center)

    def get_track_state(self, track_id: int) -> dict[str, Any] | None:
        """Get full state dict for one track."""
        try:
            tid = int(track_id)
        except (TypeError, ValueError):
            return None
        return self._tracks.get(tid)

    def get_zone_state(self, track_id: int, zone_name: str) -> dict[str, Any] | None:
        """Get zone state of one track."""
        track_state = self.get_track_state(track_id)
        if track_state is None:
            return None
        zones = track_state.get("zones", {})
        return zones.get(str(zone_name))

    def mark_zone_enter(self, track_id: int, zone_name: str, timestamp: float) -> None:
        """Mark track enters a zone."""
        try:
            tid = int(track_id)
            ts = float(timestamp)
        except (TypeError, ValueError):
            return

        zone_state = self._ensure_zone_state(tid, str(zone_name))
        zone_state["inside"] = True
        if zone_state.get("enter_time") is None:
            zone_state["enter_time"] = ts
        zone_state["last_seen_time"] = ts

    def mark_zone_exit(self, track_id: int, zone_name: str) -> None:
        """Mark track exits a zone and clear enter status."""
        zone_state = self._ensure_zone_state(self._safe_int(track_id), str(zone_name))
        if zone_state is None:
            return

        zone_state["inside"] = False
        zone_state["enter_time"] = None
        zone_state["last_seen_time"] = None
        # Reset event flags so re-entry can trigger new events.
        zone_state["triggered_intrusion"] = False
        zone_state["triggered_dwell"] = False

    def is_event_triggered(self, track_id: int, zone_name: str, event_type: str) -> bool:
        """Check if an event has already been triggered."""
        zone_state = self.get_zone_state(track_id, zone_name)
        if zone_state is None:
            return False
        key = self._event_key(event_type)
        if key is None:
            return False
        return bool(zone_state.get(key, False))

    def set_event_triggered(self, track_id: int, zone_name: str, event_type: str) -> None:
        """Set event-triggered flag for a track-zone pair."""
        tid = self._safe_int(track_id)
        if tid is None:
            return
        key = self._event_key(event_type)
        if key is None:
            return
        zone_state = self._ensure_zone_state(tid, str(zone_name))
        if zone_state is None:
            return
        zone_state[key] = True

    def cleanup_stale_tracks(self, current_time: float, max_missing_sec: float = 2.0) -> None:
        """Remove tracks not updated within max_missing_sec."""
        try:
            now = float(current_time)
            max_gap = float(max_missing_sec)
        except (TypeError, ValueError):
            return

        if max_gap < 0:
            max_gap = 0.0

        stale_ids: list[int] = []
        for tid, state in self._tracks.items():
            last_ts = state.get("last_timestamp")
            if last_ts is None:
                stale_ids.append(tid)
                continue
            try:
                gap = now - float(last_ts)
            except (TypeError, ValueError):
                stale_ids.append(tid)
                continue
            if gap > max_gap:
                stale_ids.append(tid)

        for tid in stale_ids:
            self._tracks.pop(tid, None)

    def _ensure_track_state(self, track_id: int) -> dict[str, Any]:
        """Create default track state if missing."""
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "last_bbox": None,
                "last_center": None,
                "last_timestamp": None,
                "zones": {},
            }
        return self._tracks[track_id]

    def _ensure_zone_state(self, track_id: int | None, zone_name: str) -> dict[str, Any] | None:
        """Create default zone state if missing."""
        if track_id is None:
            return None
        track_state = self._ensure_track_state(track_id)
        zones = track_state["zones"]
        if zone_name not in zones:
            zones[zone_name] = {
                "inside": False,
                "enter_time": None,
                "last_seen_time": None,
                "triggered_intrusion": False,
                "triggered_dwell": False,
            }
        return zones[zone_name]

    @staticmethod
    def _event_key(event_type: str) -> str | None:
        event_name = str(event_type).lower()
        if event_name == "intrusion":
            return "triggered_intrusion"
        if event_name == "dwell":
            return "triggered_dwell"
        return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_bbox(bbox: Any) -> list[float] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _normalize_center(center: Any) -> list[float] | None:
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            return None
        try:
            cx, cy = float(center[0]), float(center[1])
        except (TypeError, ValueError):
            return None
        return [cx, cy]
