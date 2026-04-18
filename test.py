"""Second-round end-to-end MVP demo test.

Pipeline:
1) Video reading
2) Person detection
3) Multi-object tracking
4) Restricted-zone intrusion checking
"""

from __future__ import annotations

from typing import Any

from src.event.roi_rules import IntrusionRuleEngine
from src.video.frame_sampler import FrameSampler
from src.video.stream_reader import StreamReader
from src.vision.detector import PersonDetector
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

    assert reader.open(), "视频打开失败，请检查 source 路径"
    detector.load_model()

    print("Demo started: read + detect + track + intrusion")
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

            print(
                f"frame={frame_id} ts={timestamp:.3f} "
                f"dets={len(person_dets)} tracks={len(tracks)} events={len(frame_events)}"
            )
            for event in frame_events:
                print("  intrusion_event:", event)
    finally:
        reader.release()
        print("Demo finished.")


if __name__ == "__main__":
    main()
