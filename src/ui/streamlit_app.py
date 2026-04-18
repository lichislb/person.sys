"""Streamlit UI for local MVP demo."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.main import run_pipeline
from src.service.event_store import EventStore


DEFAULT_ZONES = {
    "staff_only_zone": {
        "polygon": [[80, 680], [620, 680], [620, 1040], [80, 1040]],
        "type": "restricted",
    },
    "checkout_zone": {
        "polygon": [[680, 700], [1840, 700], [1840, 1040], [680, 1040]],
        "type": "service",
    },
}


def _save_uploaded_video(uploaded_file) -> str:
    """Save uploaded video to local demo directory and return path."""
    save_dir = Path("data/demos")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    target = save_dir / filename
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def _load_events(db_path: str, limit: int) -> list[dict[str, Any]]:
    """Load events from SQLite using EventStore."""
    store = EventStore(db_path=db_path)
    store.init_db()
    return store.list_events(limit=limit)


def _load_env_file(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from .env into process env (non-overwrite)."""
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
    """Render Streamlit MVP page."""
    st.set_page_config(page_title="Retail Security MVP", layout="wide")
    st.title("Retail Security MVP - Local Demo")
    st.caption(
        "本页面用于本地视频的异常行为检测演示：检测 -> 跟踪 -> intrusion/dwell -> SQLite 存储。"
    )

    _load_env_file(".env")
    env_base_url = os.getenv("VLM_BASE_URL", "")
    env_api_key = os.getenv("VLM_API_KEY", "")
    env_model_name = os.getenv("VLM_MODEL_NAME", "")

    init_base_url = env_base_url
    init_api_key = env_api_key
    init_model_name = env_model_name
    init_timeout = int(os.getenv("VLM_TIMEOUT_SEC", "20"))
    init_retry = int(os.getenv("VLM_MAX_RETRY", "2"))

    st.sidebar.header("运行参数")
    uploaded_video = st.sidebar.file_uploader(
        "上传视频 (mp4/avi/mov)", type=["mp4", "avi", "mov"]
    )
    frame_skip = st.sidebar.number_input("frame_skip", min_value=1, max_value=10, value=2, step=1)
    detector_conf = st.sidebar.slider("detector_conf", 0.1, 0.9, 0.35, 0.05)
    restricted_min_duration = st.sidebar.slider(
        "restricted_min_duration (s)", 0.5, 10.0, 1.5, 0.1
    )
    dwell_min_duration = st.sidebar.slider("dwell_min_duration (s)", 0.5, 30.0, 8.0, 0.5)
    display_processed = st.sidebar.checkbox("显示 OpenCV 窗口", value=False)
    output_video = st.sidebar.checkbox("保存处理后视频", value=True)
    detector_device = st.sidebar.selectbox("detector_device", ["cpu", "cuda"], index=0)
    db_path = st.sidebar.text_input("SQLite 路径", value="data/streamlit_events.db")
    enable_vlm_review = st.sidebar.checkbox("启用 VLM 复核", value=False)
    st.sidebar.caption("VLM 失败会自动降级为本地规则结果（local_only）。")

    vlm_base_url = st.sidebar.text_input("VLM base_url", value=init_base_url)
    vlm_api_key = st.sidebar.text_input("VLM api_key", value=init_api_key, type="password")
    vlm_model_name = st.sidebar.text_input("VLM model_name", value=init_model_name)
    vlm_timeout_sec = st.sidebar.number_input("VLM timeout_sec", 1, 120, init_timeout, 1)
    vlm_max_retry = st.sidebar.number_input("VLM max_retry", 0, 10, init_retry, 1)
    st.sidebar.caption(
        "VLM配置优先级: .env > 页面输入（页面输入会覆盖）"
    )

    st.sidebar.markdown("### Zones JSON")
    zones_text = st.sidebar.text_area(
        "可编辑区域配置",
        value=json.dumps(DEFAULT_ZONES, ensure_ascii=False, indent=2),
        height=220,
    )
    event_limit = st.sidebar.number_input("历史事件展示条数", 10, 5000, 200, 10)

    st.subheader("输入视频")
    if uploaded_video is None:
        st.warning("请先上传一个视频文件，再点击“开始运行”。")
    else:
        st.info(
            f"已上传: {uploaded_video.name} | 大小: {uploaded_video.size / 1024 / 1024:.2f} MB"
        )

    st.subheader("参数摘要")
    param_summary = {
        "frame_skip": int(frame_skip),
        "detector_conf": float(detector_conf),
        "restricted_min_duration": float(restricted_min_duration),
        "dwell_min_duration": float(dwell_min_duration),
        "display": bool(display_processed),
        "output_video": bool(output_video),
        "db_path": db_path,
        "detector_device": detector_device,
        "enable_vlm_review": bool(enable_vlm_review),
    }
    st.json(param_summary)

    run_clicked = st.button("开始运行", type="primary", use_container_width=True)

    if run_clicked:
        if uploaded_video is None:
            st.error("未上传视频，无法运行。")
            return

        try:
            zones = json.loads(zones_text)
            if not isinstance(zones, dict):
                raise ValueError("zones 必须是 JSON 对象")
        except Exception as exc:
            st.error(f"zones 配置解析失败: {exc}")
            return

        if enable_vlm_review and (
            not vlm_base_url.strip() or not vlm_api_key.strip() or not vlm_model_name.strip()
        ):
            st.error("已启用 VLM 复核，但 base_url/api_key/model_name 未完整填写。")
            return

        try:
            video_path = _save_uploaded_video(uploaded_video)
            output_video_path = str(Path("data/demos") / "streamlit_output.mp4")
            cfg = {
                "video_source": video_path,
                "frame_skip": int(frame_skip),
                "display": bool(display_processed),
                "db_path": db_path,
                "detector_model": "yolov8n.pt",
                "detector_conf": float(detector_conf),
                "detector_device": detector_device,
                "restricted_min_duration": float(restricted_min_duration),
                "dwell_min_duration": float(dwell_min_duration),
                "zones": zones,
                "output_video": bool(output_video),
                "output_video_path": output_video_path,
                "output_fps": 15.0,
                "enable_vlm_review": bool(enable_vlm_review),
                "snapshot_dir": "data/snapshots",
                "vlm_api_config": {
                    "base_url": vlm_base_url.strip(),
                    "api_key": vlm_api_key.strip(),
                    "model_name": vlm_model_name.strip(),
                    "timeout_sec": int(vlm_timeout_sec),
                    "max_retry": int(vlm_max_retry),
                },
            }
            with st.spinner("本地主流程运行中，请稍候..."):
                summary = run_pipeline(cfg)
            st.success("处理完成。")
            if isinstance(summary, dict):
                st.write("运行摘要:", summary)
            if output_video:
                st.write(f"处理后视频已保存到: `{output_video_path}`")
        except Exception as exc:
            st.error(f"视频处理失败: {exc}")
            return

    st.subheader("历史事件")
    try:
        events = _load_events(db_path=db_path, limit=int(event_limit))
    except Exception as exc:
        st.error(f"读取数据库失败: {exc}")
        return

    if not events:
        st.info("当前数据库中暂无事件记录。")
    else:
        cols = [
            "event_id",
            "event_type",
            "track_id",
            "zone_name",
            "duration",
            "confidence_local",
            "review_status",
            "confidence_vlm",
            "risk_level",
            "explanation",
            "final_decision",
            "created_at",
        ]
        normalized_events: list[dict[str, Any]] = []
        for e in events:
            item = dict(e)
            item.setdefault("review_status", "-")
            item.setdefault("confidence_vlm", None)
            item.setdefault("risk_level", "unknown")
            item.setdefault("explanation", "")
            item.setdefault("final_decision", "local_only")
            normalized_events.append(item)
        st.dataframe(normalized_events, use_container_width=True, column_order=cols)
        intrusion_cnt = sum(1 for e in events if e.get("event_type") == "intrusion")
        dwell_cnt = sum(1 for e in events if e.get("event_type") == "dwell")
        st.write(
            f"总事件数: {len(events)} | intrusion: {intrusion_cnt} | dwell: {dwell_cnt}"
        )


if __name__ == "__main__":
    main()
