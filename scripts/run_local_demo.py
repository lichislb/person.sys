"""Simple local demo runner for the MVP pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from src.main import run_pipeline


def _load_env_file(env_path: str = ".env") -> None:
    if not Path(env_path).exists():
        return
    try:
        for raw_line in Path(env_path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


def main() -> None:
    """Run local pipeline with a minimal static config."""
    _load_env_file(".env")
    api_key = os.getenv("VLM_API_KEY", "")
    base_url = os.getenv("VLM_BASE_URL", "")
    model_name = os.getenv("VLM_MODEL_NAME", "")
    enable_vlm = bool(api_key and base_url and model_name)

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
        "enable_vlm_review": enable_vlm,
        "snapshot_dir": "data/snapshots",
        "vlm_api_config": {
            "base_url": base_url,
            "api_key": api_key,
            "model_name": model_name,
            "timeout_sec": int(os.getenv("VLM_TIMEOUT_SEC", "20")),
            "max_retry": int(os.getenv("VLM_MAX_RETRY", "2")),
        },
    }
    print(f"enable_vlm_review={enable_vlm}")
    run_pipeline(config)


if __name__ == "__main__":
    main()
