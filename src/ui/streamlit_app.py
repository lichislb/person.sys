"""Enhanced Streamlit dashboard for retail anomaly analysis."""

from __future__ import annotations

import json
import os
from collections import Counter
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


def _load_env_file(env_path: str = ".env") -> None:
    """Load KEY=VALUE pairs from .env into process env (non-overwrite)."""
    p = Path(env_path)
    if not p.exists():
        return
    try:
        for raw_line in p.read_text(encoding="utf-8").splitlines():
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


def _normalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure event dicts have consistent display fields."""
    normalized: list[dict[str, Any]] = []
    for e in events:
        item = dict(e)
        item.setdefault("event_id", "")
        item.setdefault("event_type", "unknown")
        item.setdefault("track_id", -1)
        item.setdefault("zone_name", "unknown")
        item.setdefault("duration", 0.0)
        item.setdefault("confidence_local", 0.0)
        item.setdefault("review_status", "skipped")
        item.setdefault("confidence_vlm", None)
        item.setdefault("risk_level", "unknown")
        item.setdefault("explanation", "")
        item.setdefault("final_decision", "local_only")
        item.setdefault("created_at", "")
        normalized.append(item)
    return normalized


def _style_for_event(event: dict[str, Any]) -> tuple[str, str]:
    """Return (border_color, badge_text) by decision/risk."""
    decision = str(event.get("final_decision", "local_only")).lower()
    risk = str(event.get("risk_level", "unknown")).lower()
    event_type = str(event.get("event_type", "unknown")).lower()

    if decision == "abnormal" and risk == "high":
        return "#d62728", "High Risk"
    if event_type == "intrusion":
        return "#e74c3c", "Intrusion"
    if event_type == "dwell":
        return "#f39c12", "Dwell"
    if decision == "local_only":
        return "#f1c40f", "Fallback/Local Only"
    return "#3498db", "Info"


def _render_event_card(event: dict[str, Any]) -> None:
    """Render a compact event card with Local/VLM/Final layers."""
    color, badge = _style_for_event(event)
    created_at = str(event.get("created_at", ""))[:19]
    confidence_vlm = event.get("confidence_vlm")
    conf_vlm_text = "N/A" if confidence_vlm is None else f"{float(confidence_vlm):.2f}"

    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            background: #11182710;
            padding: 10px 12px;
            border-radius: 8px;
            margin-bottom: 10px;">
          <div style="font-weight:700;">{event.get("event_type","unknown").upper()} | Track {event.get("track_id","?")} | {badge}</div>
          <div style="font-size:12px;color:#666;">{created_at}</div>
          <div style="margin-top:6px;font-size:13px;">
            <b>Local</b>: zone={event.get("zone_name","unknown")}, duration={float(event.get("duration",0.0)):.2f}s, conf_local={float(event.get("confidence_local",0.0)):.2f}<br/>
            <b>VLM</b>: status={event.get("review_status","-")}, conf_vlm={conf_vlm_text}, risk={event.get("risk_level","unknown")}<br/>
            <b>Final</b>: decision={event.get("final_decision","local_only")}<br/>
            <b>Explanation</b>: {event.get("explanation","")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _counter_data(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(e.get(key, "unknown")) for e in events))


def _render_counter_chart(title: str, data: dict[str, int]) -> None:
    st.markdown(f"**{title}**")
    if not data:
        st.info("暂无数据")
        return
    # Streamlit can render dict-like count series directly.
    st.bar_chart(data)


def _apply_event_filters(
    events: list[dict[str, Any]],
    event_type_filter: list[str],
    risk_filter: list[str],
    decision_filter: list[str],
) -> list[dict[str, Any]]:
    """Apply multi-filter on normalized events."""
    out = events
    if event_type_filter:
        out = [e for e in out if str(e.get("event_type")) in event_type_filter]
    if risk_filter:
        out = [e for e in out if str(e.get("risk_level")) in risk_filter]
    if decision_filter:
        out = [e for e in out if str(e.get("final_decision")) in decision_filter]
    return out


def main() -> None:
    """Render enhanced analysis dashboard."""
    _load_env_file(".env")
    st.set_page_config(page_title="RetailGuard-Lite", layout="wide")

    # ===== Sidebar Config =====
    st.sidebar.header("配置中心")

    with st.sidebar.expander("视频输入", expanded=True):
        uploaded_video = st.file_uploader("上传视频 (mp4/avi/mov)", type=["mp4", "avi", "mov"])
        video_path_input = st.text_input("或本地视频路径", value="")
        clear_old_events = st.checkbox("运行前清空旧事件（删除当前 DB）", value=False)
        db_path = st.text_input("SQLite 路径", value="data/streamlit_events.db")

    with st.sidebar.expander("本地检测参数", expanded=True):
        frame_skip = st.number_input("frame_skip", min_value=1, max_value=10, value=2, step=1)
        detector_conf = st.slider("detector_conf", 0.1, 0.9, 0.35, 0.05)
        restricted_min_duration = st.slider("restricted_min_duration (s)", 0.5, 10.0, 1.5, 0.1)
        dwell_min_duration = st.slider("dwell_min_duration (s)", 0.5, 30.0, 8.0, 0.5)
        detector_device = st.selectbox("detector_device", ["cpu", "cuda"], index=0)

    with st.sidebar.expander("VLM 配置", expanded=True):
        enable_vlm_review = st.checkbox("启用 VLM 复核", value=False)
        vlm_base_url = st.text_input("VLM base_url", value=os.getenv("VLM_BASE_URL", ""))
        vlm_api_key = st.text_input("VLM api_key", value=os.getenv("VLM_API_KEY", ""), type="password")
        vlm_model_name = st.text_input("VLM model_name", value=os.getenv("VLM_MODEL_NAME", ""))
        vlm_timeout_sec = st.number_input(
            "VLM timeout_sec", 1, 120, int(os.getenv("VLM_TIMEOUT_SEC", "20")), 1
        )
        vlm_max_retry = st.number_input(
            "VLM max_retry", 0, 10, int(os.getenv("VLM_MAX_RETRY", "2")), 1
        )
        if not enable_vlm_review:
            st.caption("当前模式：Local Detection Only")
        else:
            st.caption("当前模式：Local + VLM Review Enabled")

    with st.sidebar.expander("调试选项", expanded=False):
        display_processed = st.checkbox("显示 OpenCV 窗口", value=False)
        output_video = st.checkbox("保存处理后视频", value=True)
        show_verbose_log = st.checkbox("显示详细日志", value=False)
        snapshot_dir = st.text_input("关键帧目录", value="data/snapshots")

    with st.sidebar.expander("区域配置 (JSON)", expanded=False):
        zones_text = st.text_area(
            "zones",
            value=json.dumps(DEFAULT_ZONES, ensure_ascii=False, indent=2),
            height=220,
        )

    # ===== Header =====
    mode_text = (
        "Local + VLM Review Enabled" if enable_vlm_review else "Local Detection Only"
    )
    st.markdown("## RetailGuard-Lite")
    st.markdown("### 零售监控异常行为识别与 VLM 复核系统")
    st.caption(f"本地轻量检测 + 云端视觉复核 + 异常事件管理 | 当前模式：{mode_text}")

    # ===== State =====
    if "last_summary" not in st.session_state:
        st.session_state["last_summary"] = {}
    if "last_output_video_path" not in st.session_state:
        st.session_state["last_output_video_path"] = ""
    if "last_video_name" not in st.session_state:
        st.session_state["last_video_name"] = "N/A"

    # ===== Run Button =====
    run_clicked = st.button("开始运行分析", type="primary", use_container_width=True)

    if run_clicked:
        try:
            zones = json.loads(zones_text)
            if not isinstance(zones, dict):
                raise ValueError("zones 必须是 JSON 对象")
        except Exception as exc:
            st.error(f"zones 配置解析失败: {exc}")
            return

        input_video_path = ""
        input_video_name = "N/A"
        if uploaded_video is not None:
            input_video_path = _save_uploaded_video(uploaded_video)
            input_video_name = uploaded_video.name
        elif video_path_input.strip():
            input_video_path = video_path_input.strip()
            input_video_name = Path(input_video_path).name
        else:
            st.error("请上传视频或填写视频路径。")
            return

        if enable_vlm_review and (
            not vlm_base_url.strip() or not vlm_api_key.strip() or not vlm_model_name.strip()
        ):
            st.error("已启用 VLM，但 base_url/api_key/model_name 未完整填写。")
            return

        if clear_old_events:
            try:
                db_file = Path(db_path)
                if db_file.exists():
                    db_file.unlink()
            except Exception as exc:
                st.warning(f"清空旧事件失败，将继续运行：{exc}")

        output_video_path = str(Path("data/demos") / "streamlit_output.mp4")
        cfg = {
            "video_source": input_video_path,
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
            "snapshot_dir": snapshot_dir,
            "vlm_api_config": {
                "base_url": vlm_base_url.strip(),
                "api_key": vlm_api_key.strip(),
                "model_name": vlm_model_name.strip(),
                "timeout_sec": int(vlm_timeout_sec),
                "max_retry": int(vlm_max_retry),
            },
            "vlm_extra_context": {"ui_verbose": bool(show_verbose_log)},
        }

        try:
            with st.spinner("系统处理中，请稍候..."):
                summary = run_pipeline(cfg)
            st.session_state["last_summary"] = summary if isinstance(summary, dict) else {}
            st.session_state["last_output_video_path"] = output_video_path if output_video else ""
            st.session_state["last_video_name"] = input_video_name
            st.success("处理完成。")
            if show_verbose_log:
                st.json(st.session_state["last_summary"])
        except Exception as exc:
            st.error(f"视频处理失败: {exc}")
            return

    # ===== Load Events =====
    try:
        raw_events = _load_events(db_path=db_path, limit=5000)
        events = _normalize_events(raw_events)
    except Exception as exc:
        st.error(f"读取数据库失败: {exc}")
        return

    summary = st.session_state.get("last_summary", {})
    video_name = st.session_state.get("last_video_name", "N/A")
    total_events = len(events)
    current_frame = summary.get("current_frame", "N/A")  # may be unavailable
    active_tracks = summary.get("num_active_tracks", "N/A")  # may be unavailable
    candidate_num = summary.get("num_candidate_events", 0)
    vlm_status = "Disabled"
    if enable_vlm_review:
        success_n = int(summary.get("num_vlm_success", 0))
        failed_n = int(summary.get("num_vlm_failed", 0))
        if success_n > 0:
            vlm_status = f"Review Success ({success_n})"
        elif failed_n > 0:
            vlm_status = f"Fallback Active ({failed_n})"
        else:
            vlm_status = "Enabled"

    # ===== Top Overview Cards =====
    st.markdown("### 全局概览")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("当前视频", str(video_name))
    c2.metric("当前帧号", str(current_frame))
    c3.metric("活跃 Track 数", str(active_tracks))
    c4.metric("候选事件数", str(candidate_num))
    c5.metric("VLM 状态", vlm_status)
    c6.metric("历史事件总数", str(total_events))

    # ===== Main Video + Realtime Panel =====
    st.markdown("### 主视频展示与实时事件面板")
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown("#### 处理后视频")
        output_video_path = st.session_state.get("last_output_video_path", "")
        if output_video_path and Path(output_video_path).exists():
            st.video(output_video_path)
            st.caption("已叠加 zone / track / event / decision / risk 信息")
        elif uploaded_video is not None:
            st.video(uploaded_video)
            st.caption("当前显示上传原视频（尚未生成处理后视频）")
        else:
            st.info("暂无可展示视频，请先运行分析。")

    with right:
        st.markdown("#### 实时事件面板（最近优先）")
        if not events:
            st.info("暂无事件。")
        else:
            recent_n = st.number_input("展示最近N条", min_value=1, max_value=50, value=8, step=1)
            recent_events = events[: int(recent_n)]
            for ev in recent_events:
                _render_event_card(ev)

    # ===== History Table =====
    st.markdown("### 历史事件表")
    if not events:
        st.info("当前数据库中暂无事件记录。")
    else:
        all_types = sorted({str(e.get("event_type", "unknown")) for e in events})
        all_risk = sorted({str(e.get("risk_level", "unknown")) for e in events})
        all_decisions = sorted({str(e.get("final_decision", "local_only")) for e in events})

        f1, f2, f3 = st.columns(3)
        type_filter = f1.multiselect("按 event_type 筛选", all_types, default=[])
        risk_filter = f2.multiselect("按 risk_level 筛选", all_risk, default=[])
        decision_filter = f3.multiselect("按 final_decision 筛选", all_decisions, default=[])

        filtered_events = _apply_event_filters(events, type_filter, risk_filter, decision_filter)
        if not filtered_events:
            st.warning("筛选后无数据。")
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
                "final_decision",
                "created_at",
            ]
            st.dataframe(filtered_events, use_container_width=True, column_order=cols)

    # ===== Charts =====
    st.markdown("### 统计图表")
    if not events:
        st.info("暂无数据可用于统计图。")
    else:
        chart1, chart2 = st.columns(2)
        with chart1:
            _render_counter_chart("事件类型分布", _counter_data(events, "event_type"))
            _render_counter_chart("最终决策分布", _counter_data(events, "final_decision"))
        with chart2:
            _render_counter_chart("风险等级分布", _counter_data(events, "risk_level"))
            _render_counter_chart("复核状态分布", _counter_data(events, "review_status"))


if __name__ == "__main__":
    main()
