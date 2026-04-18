"""Visualization helpers for retail security MVP."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_zones(frame: np.ndarray, zones: dict[str, dict[str, Any]] | None) -> np.ndarray:
    """Draw polygon zones and labels."""
    canvas = frame.copy()
    if zones is None:
        return canvas

    for zone_name, zone_cfg in zones.items():
        if not isinstance(zone_cfg, dict):
            continue
        polygon = zone_cfg.get("polygon")
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            continue

        zone_type = str(zone_cfg.get("type", "unknown")).lower()
        color = (0, 100, 255) if zone_type == "restricted" else (255, 180, 0)

        try:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        except Exception:
            continue

        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        anchor = tuple(pts[0][0].tolist())
        label = f"{zone_name} ({zone_type})"
        cv2.putText(
            canvas,
            label,
            (int(anchor[0]), max(20, int(anchor[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_tracks(frame: np.ndarray, tracks: list[dict[str, Any]] | None) -> np.ndarray:
    """Draw track bboxes and IDs."""
    canvas = frame.copy()
    if not tracks:
        return canvas

    for track in tracks:
        if not isinstance(track, dict):
            continue

        bbox = track.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        except (TypeError, ValueError):
            continue

        track_id = track.get("track_id", -1)
        class_name = track.get("class_name", "")
        label = f"id={track_id}"
        if class_name:
            label = f"{label} {class_name}"

        color = (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_events(frame: np.ndarray, events: list[dict[str, Any]] | None) -> np.ndarray:
    """Draw current frame events as text list."""
    canvas = frame.copy()
    if not events:
        return canvas

    y = 48
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", "event"))
        track_id = event.get("track_id", "?")
        zone_name = str(event.get("zone_name", "zone"))
        final_decision = str(event.get("final_decision", "local_only"))
        risk_level = str(event.get("risk_level", "unknown"))
        try:
            duration = float(event.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0

        text = (
            f"{idx + 1}. {event_type} id={track_id} zone={zone_name} "
            f"dur={duration:.2f}s decision={final_decision} risk={risk_level}"
        )
        color = (0, 0, 255) if event_type == "intrusion" else (0, 165, 255)
        cv2.putText(
            canvas,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 22

    return canvas


def draw_status_bar(frame: np.ndarray, stats: dict[str, Any] | None = None) -> np.ndarray:
    """Draw top status bar with concise pipeline state."""
    canvas = frame.copy()
    stats = stats or {}

    frame_id = stats.get("frame_id", "-")
    num_tracks = stats.get("num_tracks", 0)
    num_events = stats.get("num_events", 0)
    num_dets = stats.get("num_dets", 0)
    timestamp = stats.get("timestamp", 0.0)

    text = (
        f"frame={frame_id} ts={float(timestamp):.2f}s "
        f"dets={num_dets} tracks={num_tracks} events={num_events}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
