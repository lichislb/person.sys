"""Round-4 MVP verification script.

Usage examples:
1) Quick environment and DB checks only:
   python tests/verify_mvp_round4.py --mode quick

2) Full pipeline smoke run on local video:
   python tests/verify_mvp_round4.py --mode full --video your_video.mp4
"""

from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
from pathlib import Path
from typing import Any

from src.vision.geometry import point_in_polygon


def check_dependencies() -> dict[str, bool]:
    """Check required third-party dependencies."""
    deps = ["cv2", "numpy", "ultralytics"]
    result: dict[str, bool] = {}
    for name in deps:
        try:
            importlib.import_module(name)
            result[name] = True
        except Exception:
            result[name] = False
    return result


def check_event_store_schema(db_path: str) -> tuple[bool, list[str]]:
    """Initialize DB and verify events table columns."""
    from src.service.event_store import EventStore

    store = EventStore(db_path=db_path)
    store.init_db()

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("PRAGMA table_info(events)").fetchall()
    cols = [r[1] for r in rows]

    required = {
        "event_id",
        "event_type",
        "track_id",
        "zone_name",
        "start_time",
        "current_time",
        "duration",
        "confidence_local",
        "created_at",
    }
    ok = required.issubset(set(cols))
    return ok, cols


def check_event_store_crud(db_path: str, seed: bool = True) -> tuple[bool, dict[str, Any]]:
    """Verify insert/list/get/exists behavior."""
    from src.service.event_store import EventStore

    store = EventStore(db_path=db_path)
    store.init_db()

    if not seed:
        return True, {"seed": False, "message": "crud seed skipped by flag"}

    event = {
        "event_type": "intrusion",
        "track_id": 1,
        "zone_name": "staff_only_zone",
        "start_time": 10.0,
        "current_time": 11.2,
        "duration": 1.2,
        "confidence_local": 0.8,
    }

    store.insert_event(event)
    store.insert_event(dict(event))  # dedup check
    events = store.list_events(limit=10)
    if not events:
        return False, {"reason": "no events after insert"}

    eid = events[0]["event_id"]
    exists = store.event_exists(eid)
    got = store.get_event(eid)
    dedup_ok = len(events) == 1
    all_ok = bool(exists and got is not None and dedup_ok)

    return all_ok, {
        "event_id": eid,
        "event_count": len(events),
        "dedup_ok": dedup_ok,
        "exists_ok": exists,
        "get_ok": got is not None,
    }


def run_pipeline_smoke(
    video_path: str,
    db_path: str,
    max_frames: int = 120,
    verbose_zones: bool = False,
) -> tuple[bool, str]:
    """Run a bounded smoke pipeline using existing modules."""
    from src.main import build_components

    config = {
        "video_source": video_path,
        "frame_skip": 2,
        "display": False,
        "db_path": db_path,
        "detector_model": "yolov8n.pt",
        "detector_conf": 0.35,
        "detector_device": "cpu",
        "restricted_min_duration": 1.0,
        "dwell_min_duration": 3.0,
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
    }
    zones = config["zones"]

    components = build_components(config)
    reader = components["reader"]
    sampler = components["sampler"]
    detector = components["detector"]
    tracker = components["tracker"]
    generator = components["candidate_generator"]
    store = components["event_store"]

    store.init_db()
    if not reader.open():
        return False, f"failed to open video: {video_path}"

    try:
        detector.load_model()
    except Exception as exc:
        reader.release()
        return False, f"detector.load_model() failed: {exc}"

    processed = 0
    inserted = 0
    try:
        while processed < max_frames:
            ok, frame_obj = reader.read()
            if not ok or frame_obj is None:
                break
            frame_id = int(frame_obj["frame_id"])
            ts = float(frame_obj["timestamp"])
            if not sampler.should_process(frame_id):
                continue

            image = frame_obj["image"]
            dets = detector.predict(image)
            tracks = tracker.update(dets, image=image)

            if verbose_zones:
                zone_counts: dict[str, int] = {z: 0 for z in zones.keys()}
                for track in tracks:
                    center = track.get("center")
                    if not isinstance(center, (list, tuple)) or len(center) != 2:
                        continue
                    try:
                        cx, cy = float(center[0]), float(center[1])
                    except (TypeError, ValueError):
                        continue
                    for zone_name, zone_cfg in zones.items():
                        polygon = zone_cfg.get("polygon")
                        if point_in_polygon((cx, cy), polygon):
                            zone_counts[zone_name] += 1
                print(
                    f"[zone-debug] frame={frame_id} ts={ts:.2f} "
                    f"tracks={len(tracks)} in_zone={zone_counts}"
                )

            events = generator.generate(tracks=tracks, timestamp=ts)
            generator.cleanup(current_time=ts)
            for e in events:
                store.insert_event(e)
                inserted += 1
            processed += 1
    finally:
        reader.release()

    total_saved = len(store.list_events(limit=100000))
    return True, (
        f"processed_frames={processed}, runtime_inserted={inserted}, "
        f"db_events={total_saved}, db_path={db_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify round-4 MVP pipeline.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--video", default="your_video.mp4")
    parser.add_argument("--db", default="data/verify_events.db")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--skip-crud-seed",
        action="store_true",
        help="Do not write CRUD test seed event into DB.",
    )
    parser.add_argument(
        "--verbose-zones",
        action="store_true",
        help="Print per-frame track and in-zone counters during full run.",
    )
    args = parser.parse_args()

    print("== Dependency Check ==")
    dep = check_dependencies()
    for k, v in dep.items():
        print(f"{k}: {'OK' if v else 'MISSING'}")

    print("\n== EventStore Schema Check ==")
    schema_ok, cols = check_event_store_schema(args.db)
    print(f"schema_ok={schema_ok}, columns={cols}")

    print("\n== EventStore CRUD Check ==")
    crud_ok, detail = check_event_store_crud(args.db, seed=not args.skip_crud_seed)
    print(f"crud_ok={crud_ok}, detail={detail}")

    if args.mode == "quick":
        print("\nQuick mode completed.")
        return

    if not dep.get("ultralytics", False):
        print("\nFull mode skipped: ultralytics is missing.")
        return
    if not Path(args.video).exists():
        print(f"\nFull mode skipped: video not found: {args.video}")
        return

    print("\n== Full Pipeline Smoke Check ==")
    ok, msg = run_pipeline_smoke(
        args.video,
        args.db,
        max_frames=max(1, args.max_frames),
        verbose_zones=args.verbose_zones,
    )
    print(f"pipeline_ok={ok}, detail={msg}")


if __name__ == "__main__":
    main()
