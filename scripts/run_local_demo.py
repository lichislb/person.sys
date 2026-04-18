"""Simple local demo runner for the MVP pipeline."""

from __future__ import annotations

from src.main import run_pipeline


def main() -> None:
    """Run local pipeline with a minimal static config."""
    config = {
        "video_source": "your_video.mp4",
        "frame_skip": 2,
        "display": False,
        "db_path": "data/local_demo_events.db",
        "detector_model": "yolov8n.pt",
        "detector_conf": 0.35,
        "detector_device": "cpu",
        "restricted_min_duration": 1.0,
        "dwell_min_duration": 3.0,
        "zones": {
            "staff_only_zone": {
                "polygon": [[80, 680], [620, 680], [620, 1040], [80, 1040]],
                "type": "restricted",
            },
            "checkout_zone": {
                "polygon": [[680, 700], [1840, 700], [1840, 1040], [680, 1040]],
                "type": "service",
            },
        },
        "output_video": True,
        "output_video_path": "data/demos/local_demo_output.mp4",
        "output_fps": 15.0,
    }
    run_pipeline(config)


if __name__ == "__main__":
    main()
