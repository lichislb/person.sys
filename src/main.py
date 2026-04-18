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
from src.vlm.client import VLMApiClient
from src.vlm.fallback import fuse_event_with_review
from src.vlm.parser import ResponseParser
from src.vlm.prompt_builder import PromptBuilder
from src.vlm.reviewer import VLMReviewer
from src.video.frame_sampler import FrameSampler
from src.video.stream_reader import StreamReader
from src.vision.detector import PersonDetector
from src.vision.tracker import ObjectTracker


def _load_env_file(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from .env into process env (non-overwrite)."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Keep pipeline resilient even if .env has formatting issues.
        return


def build_components(config: dict[str, Any]) -> dict[str, Any]:
    """Build all pipeline components from config."""
    _load_env_file(".env")
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
    enable_vlm_review = bool(config.get("enable_vlm_review", False))
    vlm_api_config = config.get("vlm_api_config", {}) if enable_vlm_review else {}
    vlm_base_url = str(vlm_api_config.get("base_url") or os.getenv("VLM_BASE_URL", ""))
    vlm_api_key = str(vlm_api_config.get("api_key") or os.getenv("VLM_API_KEY", ""))
    vlm_model_name = str(vlm_api_config.get("model_name") or os.getenv("VLM_MODEL_NAME", ""))
    vlm_timeout_sec = int(vlm_api_config.get("timeout_sec", os.getenv("VLM_TIMEOUT_SEC", 20)))
    vlm_max_retry = int(vlm_api_config.get("max_retry", os.getenv("VLM_MAX_RETRY", 2)))

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
    reviewer: VLMReviewer | None = None
    if enable_vlm_review:
        try:
            client = VLMApiClient(
                base_url=vlm_base_url,
                api_key=vlm_api_key,
                model_name=vlm_model_name,
                timeout_sec=vlm_timeout_sec,
                max_retry=vlm_max_retry,
            )
            reviewer = VLMReviewer(
                client=client,
                prompt_builder=PromptBuilder(),
                parser=ResponseParser(),
            )
        except Exception:
            reviewer = None

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
        "reviewer": reviewer,
        "enable_vlm_review": enable_vlm_review,
    }


def _save_snapshot(
    frame: Any,
    snapshot_dir: str,
    frame_id: int,
    track_id: Any,
    event_type: Any,
) -> str:
    """Save current frame as a snapshot for VLM review."""
    os.makedirs(snapshot_dir, exist_ok=True)
    filename = f"f{frame_id}_t{track_id}_{event_type}.jpg"
    path = os.path.join(snapshot_dir, filename)
    ok = cv2.imwrite(path, frame)
    if not ok:
        raise RuntimeError(f"failed to write snapshot: {path}")
    return path


def _create_video_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a browser-friendly writer with codec fallback."""
    codec_candidates = ["avc1", "mp4v"]
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    # last fallback keeps behavior explicit
    raise RuntimeError(f"failed to create VideoWriter for {path}")


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run local MVP pipeline end-to-end."""
    components = build_components(config)
    reader: StreamReader = components["reader"]
    sampler: FrameSampler = components["sampler"]
    detector: PersonDetector = components["detector"]
    tracker: ObjectTracker = components["tracker"]
    candidate_generator: CandidateEventGenerator = components["candidate_generator"]
    event_store: EventStore = components["event_store"]
    zones: dict[str, dict[str, Any]] = components["zones"]
    reviewer: VLMReviewer | None = components.get("reviewer")
    enable_vlm_review: bool = bool(components.get("enable_vlm_review", False))

    display = bool(config.get("display", True))
    output_video = bool(config.get("output_video", False))
    output_video_path = str(config.get("output_video_path", "output_demo.mp4"))
    fps = float(config.get("output_fps", 15.0))
    window_name = "Retail Security MVP (Press q to quit)"
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    enable_window = display and has_display
    video_writer: cv2.VideoWriter | None = None
    snapshot_dir = str(config.get("snapshot_dir", "data/snapshots"))
    extra_context = config.get("vlm_extra_context", {})
    store_events = bool(config.get("store_events", False))
    alert_hold_sec = float(config.get("alert_hold_sec", 3.0))
    active_alerts: dict[str, dict[str, Any]] = {}

    num_candidate_events = 0
    num_final_events = 0
    num_vlm_success = 0
    num_vlm_failed = 0
    num_vlm_skipped = 0
    last_frame_id: int | None = None
    last_num_tracks: int | None = None

    if store_events:
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
            last_frame_id = frame_id

            # Keep original frame on skipped steps for smooth display/output.
            person_dets: list[dict[str, Any]] = []
            tracks: list[dict[str, Any]] = []
            final_events: list[dict[str, Any]] = []

            if sampler.should_process(frame_id):
                person_dets = detector.predict(frame)
                tracks = tracker.update(person_dets, image=frame)
                last_num_tracks = len(tracks)
                candidate_events = candidate_generator.generate(tracks=tracks, timestamp=timestamp)
                candidate_generator.cleanup(current_time=timestamp)
                num_candidate_events += len(candidate_events)

                for event in candidate_events:
                    review_result: dict[str, Any] | None = None

                    if enable_vlm_review and reviewer is not None:
                        try:
                            snapshot_path = _save_snapshot(
                                frame=frame,
                                snapshot_dir=snapshot_dir,
                                frame_id=frame_id,
                                track_id=event.get("track_id"),
                                event_type=event.get("event_type"),
                            )
                            review_result = reviewer.review(
                                candidate_event=event,
                                image_paths=[snapshot_path],
                                extra_context=extra_context,
                            )
                            if review_result.get("review_status") == "success":
                                num_vlm_success += 1
                            else:
                                num_vlm_failed += 1
                        except Exception as exc:
                            review_result = {
                                "review_status": "failed",
                                "is_abnormal": None,
                                "abnormal_type": None,
                                "risk_level": "unknown",
                                "confidence_vlm": None,
                                "explanation": f"VLM review exception: {exc}",
                                "raw_response": None,
                            }
                            num_vlm_failed += 1
                    elif enable_vlm_review and reviewer is None:
                        review_result = {
                            "review_status": "failed",
                            "is_abnormal": None,
                            "abnormal_type": None,
                            "risk_level": "unknown",
                            "confidence_vlm": None,
                            "explanation": "VLM reviewer not initialized.",
                            "raw_response": None,
                        }
                        num_vlm_failed += 1
                    else:
                        num_vlm_skipped += 1

                    final_event = fuse_event_with_review(
                        candidate_event=event,
                        review_result=review_result,
                    )
                    final_events.append(final_event)
                    # Keep only actionable alerts in video overlay.
                    if str(final_event.get("final_decision", "")).lower() != "normal":
                        alert_key = (
                            f"{final_event.get('event_type')}_"
                            f"{final_event.get('track_id')}_"
                            f"{final_event.get('zone_name')}"
                        )
                        active_alerts[alert_key] = {
                            **final_event,
                            "_expire_at": float(timestamp) + max(0.1, alert_hold_sec),
                        }
                    if store_events:
                        event_store.insert_event(final_event)
                    num_final_events += 1

            # Persist alert rendering for a few seconds, not just trigger frame.
            alive_alerts: list[dict[str, Any]] = []
            for key in list(active_alerts.keys()):
                info = active_alerts[key]
                if float(info.get("_expire_at", 0.0)) >= float(timestamp):
                    alive_alerts.append(info)
                else:
                    active_alerts.pop(key, None)

            vis_frame = draw_zones(frame, zones)
            vis_frame = draw_tracks(vis_frame, tracks)
            vis_frame = draw_events(vis_frame, alive_alerts)
            vis_frame = draw_status_bar(
                vis_frame,
                stats={
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "num_dets": len(person_dets),
                    "num_tracks": len(tracks),
                    "num_events": len(alive_alerts),
                },
            )

            if output_video:
                if video_writer is None:
                    h, w = vis_frame.shape[:2]
                    video_writer = _create_video_writer(output_video_path, fps, w, h)
                video_writer.write(vis_frame)

            if enable_window:
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Stopped by user (q).")
                    break

            if alive_alerts:
                print(f"frame={frame_id} ts={timestamp:.3f} alerts={len(alive_alerts)}")
                for event in alive_alerts:
                    print("  event:", event)
    finally:
        reader.release()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved output video: {output_video_path}")
        if enable_window:
            cv2.destroyAllWindows()

    stored_count = 0
    if store_events:
        stored_count = len(event_store.list_events(limit=100000))
        print(f"Pipeline finished. total_events_in_db={stored_count}")
    else:
        print("Pipeline finished. event storage disabled.")
    return {
        "num_candidate_events": num_candidate_events,
        "num_final_events": num_final_events,
        "num_vlm_success": num_vlm_success,
        "num_vlm_failed": num_vlm_failed,
        "num_vlm_skipped": num_vlm_skipped,
        "current_frame": last_frame_id if last_frame_id is not None else "N/A",
        "num_active_tracks": last_num_tracks if last_num_tracks is not None else "N/A",
        "total_events_in_db": stored_count,
    }


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
