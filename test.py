"""Second-round end-to-end MVP demo test.

Pipeline:
1) Video reading
2) Person detection
3) Multi-object tracking
4) Restricted-zone intrusion checking
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from src.event.roi_rules import IntrusionRuleEngine
from src.video.frame_sampler import FrameSampler
from src.video.stream_reader import StreamReader
from src.vision.detector import PersonDetector
from src.vision.geometry import point_in_polygon
from src.vision.tracker import ObjectTracker


class DemoStateManager:
    """Minimal in-memory state manager for local demo testing."""

    def __init__(self) -> None:
        self._tracks: dict[int, dict[str, Any]] = {}
        self._zone_states: dict[tuple[int, str], dict[str, Any]] = {}
        self._triggered_events: set[tuple[int, str, str]] = set()

    def update_track(self, track_id: int, timestamp: float, bbox: Any, center: Any) -> None:
        self._tracks[track_id] = {
            "timestamp": float(timestamp),
            "bbox": bbox,
            "center": center,
        }

    def get_zone_state(self, track_id: int, zone_name: str) -> dict[str, Any] | None:
        return self._zone_states.get((track_id, zone_name))

    def mark_zone_enter(self, track_id: int, zone_name: str, timestamp: float) -> None:
        self._zone_states[(track_id, zone_name)] = {"enter_time": float(timestamp)}

    def mark_zone_exit(self, track_id: int, zone_name: str) -> None:
        self._zone_states.pop((track_id, zone_name), None)
        self._triggered_events = {
            key
            for key in self._triggered_events
            if not (key[0] == track_id and key[1] == zone_name)
        }

    def is_event_triggered(self, track_id: int, zone_name: str, event_type: str) -> bool:
        return (track_id, zone_name, event_type) in self._triggered_events

    def set_event_triggered(self, track_id: int, zone_name: str, event_type: str) -> None:
        self._triggered_events.add((track_id, zone_name, event_type))


def _draw_zones(canvas: np.ndarray, zones: dict[str, dict[str, Any]]) -> None:
    """Draw configured zone polygons and labels."""
    for zone_name, zone_cfg in zones.items():
        polygon = zone_cfg.get("polygon")
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            continue

        color = (0, 100, 255) if zone_cfg.get("type") == "restricted" else (255, 180, 0)
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

        anchor = tuple(np.array(polygon[0], dtype=np.int32).tolist())
        cv2.putText(
            canvas,
            f"{zone_name} ({zone_cfg.get('type', 'unknown')})",
            (anchor[0], max(20, anchor[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_tracks(
    canvas: np.ndarray, tracks: list[dict[str, Any]], zones: dict[str, dict[str, Any]]
) -> None:
    """Draw track bbox, center, and id."""
    for track in tracks:
        bbox = track.get("bbox", [0, 0, 0, 0])
        center = track.get("center", [0.0, 0.0])
        track_id = track.get("track_id", -1)

        try:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox]
            cx, cy = [int(float(v)) for v in center]
        except (TypeError, ValueError):
            continue

        in_restricted = False
        for zone_cfg in zones.values():
            if zone_cfg.get("type") != "restricted":
                continue
            polygon = zone_cfg.get("polygon")
            if point_in_polygon((cx, cy), polygon):
                in_restricted = True
                break

        color = (0, 0, 255) if in_restricted else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.circle(canvas, (cx, cy), 3, color, -1)
        cv2.putText(
            canvas,
            f"id={track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_status_bar(
    canvas: np.ndarray,
    frame_id: int,
    timestamp: float,
    det_count: int,
    track_count: int,
    event_count: int,
) -> None:
    """Draw top status text."""
    text = (
        f"frame={frame_id} ts={timestamp:.2f}s "
        f"dets={det_count} tracks={track_count} events={event_count}"
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


def main() -> None:
    reader = StreamReader(source="your_video.mp4", loop=False, reconnect=False)
    sampler = FrameSampler(frame_skip=2)  # 每2帧处理1帧
    detector = PersonDetector(model_name="yolov8n.pt", conf_threshold=0.35, device="cuda")
    tracker = ObjectTracker(max_lost=30, min_box_area=100.0, iou_threshold=0.3)

    # 示例敏感区（请按实际画面分辨率调整）。
    zones = {
        "staff_only_zone": {
            "polygon": [[100, 100], [400, 100], [400, 300], [100, 300]],
            "type": "restricted",
        },
        "checkout_zone": {
            "polygon": [[450, 100], [800, 100], [800, 350], [450, 350]],
            "type": "service",
        },
    }
    rule_engine = IntrusionRuleEngine(zones=zones, min_duration_sec=1.5)
    state_manager = DemoStateManager()
    window_name = "Retail Security Demo (Press q to quit)"
    save_output = True
    output_path = "output_demo.mp4"
    video_writer: cv2.VideoWriter | None = None
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    enable_window = has_display

    assert reader.open(), "视频打开失败，请检查 source 路径"
    detector.load_model()

    if enable_window:
        print("Demo started: read + detect + track + intrusion + visualize(window+video)")
    else:
        print("Demo started: read + detect + track + intrusion + visualize(video-only, headless)")
    try:
        while True:
            ok, frame_obj = reader.read()
            if not ok:
                break

            frame_id = int(frame_obj["frame_id"])
            timestamp = float(frame_obj["timestamp"])
            if not sampler.should_process(frame_id):
                continue

            image = frame_obj["image"]
            person_dets = detector.predict(image)
            tracks = tracker.update(person_dets, image=image)

            frame_events: list[dict[str, Any]] = []
            for track in tracks:
                events = rule_engine.check_track(
                    track=track,
                    timestamp=timestamp,
                    state_manager=state_manager,
                )
                if events:
                    frame_events.extend(events)

            vis = image.copy()
            _draw_zones(vis, zones)
            _draw_tracks(vis, tracks, zones)
            _draw_status_bar(vis, frame_id, timestamp, len(person_dets), len(tracks), len(frame_events))

            if frame_events:
                first_event = frame_events[0]
                hint = (
                    f"ALERT intrusion: id={first_event['track_id']} "
                    f"zone={first_event['zone_name']} dur={first_event['duration']:.2f}s"
                )
                cv2.putText(
                    vis,
                    hint,
                    (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if save_output:
                if video_writer is None:
                    h, w = vis.shape[:2]
                    fps = 15.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                video_writer.write(vis)

            if enable_window:
                cv2.imshow(window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("User stopped demo by pressing 'q'.")
                    break

            print(
                f"frame={frame_id} ts={timestamp:.3f} "
                f"dets={len(person_dets)} tracks={len(tracks)} events={len(frame_events)}"
            )
            for event in frame_events:
                print("  intrusion_event:", event)
    finally:
        reader.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved visualization video to: {output_path}")
        if enable_window:
            cv2.destroyAllWindows()
        print("Demo finished.")


if __name__ == "__main__":
    main()
