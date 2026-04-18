"""Local MVP pipeline entry for retail anomaly detection."""

from __future__ import annotations

import os
from typing import Any

import cv2

from src.event.candidate_generator import CandidateEventGenerator
from src.event.dwell_rules import DwellRuleEngine
from src.event.roi_rules import IntrusionRuleEngine
from src.event.state_manager import TrackStateManager
from src.service.event_store import EventStore
from src.utils.vis import draw_events, draw_status_bar, draw_tracks, draw_zones
from src.video.frame_sampler import FrameSampler
from src.video.stream_reader import StreamReader
from src.vision.detector import PersonDetector
from src.vision.tracker import ObjectTracker


def build_components(config: dict[str, Any]) -> dict[str, Any]:
    """Build all pipeline components from config."""
    video_source = str(config.get("video_source", "your_video.mp4"))
    frame_skip = int(config.get("frame_skip", 2))
    detector_model = str(config.get("detector_model", "yolov8n.pt"))
    detector_conf = float(config.get("detector_conf", 0.35))
    detector_device = str(config.get("detector_device", "cuda"))
    tracker_max_lost = int(config.get("tracker_max_lost", 30))
    tracker_min_box_area = float(config.get("tracker_min_box_area", 100.0))
    tracker_iou_threshold = float(config.get("tracker_iou_threshold", 0.3))
    restricted_min_duration = float(config.get("restricted_min_duration", 1.5))
    dwell_min_duration = float(config.get("dwell_min_duration", 8.0))
    state_max_missing_sec = float(config.get("state_max_missing_sec", 2.0))
    zones = config.get("zones", {})
    db_path = str(config.get("db_path", "data/events.db"))

    reader = StreamReader(source=video_source, loop=False, reconnect=False)
    sampler = FrameSampler(frame_skip=frame_skip)
    detector = PersonDetector(
        model_name=detector_model,
        conf_threshold=detector_conf,
        device=detector_device,
    )
    tracker = ObjectTracker(
        max_lost=tracker_max_lost,
        min_box_area=tracker_min_box_area,
        iou_threshold=tracker_iou_threshold,
    )
    state_manager = TrackStateManager()
    intrusion_engine = IntrusionRuleEngine(zones=zones, min_duration_sec=restricted_min_duration)
    dwell_engine = DwellRuleEngine(zones=zones, min_duration_sec=dwell_min_duration)
    candidate_generator = CandidateEventGenerator(
        intrusion_engine=intrusion_engine,
        dwell_engine=dwell_engine,
        state_manager=state_manager,
        max_missing_sec=state_max_missing_sec,
    )
    event_store = EventStore(db_path=db_path)

    return {
        "reader": reader,
        "sampler": sampler,
        "detector": detector,
        "tracker": tracker,
        "state_manager": state_manager,
        "intrusion_engine": intrusion_engine,
        "dwell_engine": dwell_engine,
        "candidate_generator": candidate_generator,
        "event_store": event_store,
        "zones": zones,
    }


def run_pipeline(config: dict[str, Any]) -> None:
    """Run local MVP pipeline end-to-end."""
    components = build_components(config)
    reader: StreamReader = components["reader"]
    sampler: FrameSampler = components["sampler"]
    detector: PersonDetector = components["detector"]
    tracker: ObjectTracker = components["tracker"]
    candidate_generator: CandidateEventGenerator = components["candidate_generator"]
    event_store: EventStore = components["event_store"]
    zones: dict[str, dict[str, Any]] = components["zones"]

    display = bool(config.get("display", True))
    output_video = bool(config.get("output_video", False))
    output_video_path = str(config.get("output_video_path", "output_demo.mp4"))
    fps = float(config.get("output_fps", 15.0))
    window_name = "Retail Security MVP (Press q to quit)"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    enable_window = display and has_display
    video_writer: cv2.VideoWriter | None = None

    event_store.init_db()
    assert reader.open(), "视频打开失败，请检查 video_source"
    detector.load_model()

    if enable_window:
        print("Pipeline started (window enabled).")
    else:
        print("Pipeline started (headless or display disabled).")

    try:
        while True:
            ok, frame_obj = reader.read()
            if not ok or frame_obj is None:
                break

            frame_id = int(frame_obj["frame_id"])
            timestamp = float(frame_obj["timestamp"])
            frame = frame_obj["image"]

            # Keep original frame on skipped steps for smooth display/output.
            person_dets: list[dict[str, Any]] = []
            tracks: list[dict[str, Any]] = []
            events: list[dict[str, Any]] = []

            if sampler.should_process(frame_id):
                person_dets = detector.predict(frame)
                tracks = tracker.update(person_dets, image=frame)
                events = candidate_generator.generate(tracks=tracks, timestamp=timestamp)
                candidate_generator.cleanup(current_time=timestamp)

                for event in events:
                    event_store.insert_event(event)

            vis_frame = draw_zones(frame, zones)
            vis_frame = draw_tracks(vis_frame, tracks)
            vis_frame = draw_events(vis_frame, events)
            vis_frame = draw_status_bar(
                vis_frame,
                stats={
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "num_dets": len(person_dets),
                    "num_tracks": len(tracks),
                    "num_events": len(events),
                },
            )

            if output_video:
                if video_writer is None:
                    h, w = vis_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                video_writer.write(vis_frame)

            if enable_window:
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Stopped by user (q).")
                    break

            if events:
                print(f"frame={frame_id} ts={timestamp:.3f} events={len(events)}")
                for event in events:
                    print("  event:", event)
    finally:
        reader.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved output video: {output_video_path}")
        if enable_window:
            cv2.destroyAllWindows()

    stored_count = len(event_store.list_events(limit=100000))
    print(f"Pipeline finished. total_events_in_db={stored_count}")


if __name__ == "__main__":
    sample_config: dict[str, Any] = {
        "video_source": "your_video.mp4",
        "frame_skip": 2,
        "display": True,
        "db_path": "data/events.db",
        "detector_model": "yolov8n.pt",
        "detector_conf": 0.35,
        "detector_device": "cuda",
        "restricted_min_duration": 1.5,
        "dwell_min_duration": 8.0,
        "zones": {
            "staff_only_zone": {
                "polygon": [[100, 100], [400, 100], [400, 300], [100, 300]],
                "type": "restricted",
            },
            "checkout_zone": {
                "polygon": [[450, 100], [800, 100], [800, 350], [450, 350]],
                "type": "service",
            },
        },
        # Optional:
        "output_video": False,
        "output_video_path": "output_demo.mp4",
        "output_fps": 15.0,
    }
    run_pipeline(sample_config)
