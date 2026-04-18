"""Dwell rule engine for service-like zones."""

from __future__ import annotations

from typing import Any

from src.vision.geometry import point_in_polygon


class DwellRuleEngine:
    """Detect dwell candidate events in configured zones."""

    VALID_ZONE_TYPES = {"service", "dwell"}

    def __init__(self, zones: dict[str, dict[str, Any]], min_duration_sec: float = 8.0) -> None:
        self.zones = zones if isinstance(zones, dict) else {}
        self.min_duration_sec = float(max(0.0, min_duration_sec))

    def check_track(
        self, track: dict[str, Any], timestamp: float, state_manager: Any
    ) -> list[dict[str, Any]]:
        """Check one track for dwell events."""
        events: list[dict[str, Any]] = []
        if not isinstance(track, dict):
            return events

        try:
            track_id = int(track.get("track_id"))
            current_time = float(timestamp)
        except (TypeError, ValueError):
            return events

        bbox = track.get("bbox")
        center = track.get("center")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            return events

        try:
            cx, cy = float(center[0]), float(center[1])
        except (TypeError, ValueError):
            return events

        state_manager.update_track(track_id, current_time, bbox, [cx, cy])

        for zone_name, zone_cfg in self.zones.items():
            if not isinstance(zone_cfg, dict):
                continue
            zone_type = str(zone_cfg.get("type", "")).lower()
            if zone_type not in self.VALID_ZONE_TYPES:
                continue

            polygon = zone_cfg.get("polygon")
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
                continue

            in_zone = point_in_polygon((cx, cy), polygon)
            zone_state = state_manager.get_zone_state(track_id, zone_name) or {}
            enter_time_raw = zone_state.get("enter_time")
            enter_time: float | None = None
            if enter_time_raw is not None:
                try:
                    enter_time = float(enter_time_raw)
                except (TypeError, ValueError):
                    enter_time = None

            if in_zone:
                if enter_time is None:
                    state_manager.mark_zone_enter(track_id, zone_name, current_time)
                    enter_time = current_time

                duration = max(0.0, current_time - enter_time)
                already_triggered = bool(
                    state_manager.is_event_triggered(track_id, zone_name, "dwell")
                )
                if duration >= self.min_duration_sec and not already_triggered:
                    events.append(
                        {
                            "event_type": "dwell",
                            "track_id": track_id,
                            "zone_name": zone_name,
                            "start_time": float(enter_time),
                            "current_time": current_time,
                            "duration": duration,
                            "confidence_local": self._confidence_from_duration(duration),
                        }
                    )
                    state_manager.set_event_triggered(track_id, zone_name, "dwell")
            else:
                if enter_time is not None:
                    state_manager.mark_zone_exit(track_id, zone_name)

        return events

    def _confidence_from_duration(self, duration_sec: float) -> float:
        """Map dwell duration to confidence score."""
        if self.min_duration_sec <= 0.0:
            return 0.9
        ratio = duration_sec / self.min_duration_sec
        # 0.55 at threshold, linearly increase to 0.95.
        conf = 0.55 + max(0.0, min(1.0, ratio - 1.0)) * 0.40
        return float(max(0.0, min(0.95, conf)))
