# Retail Security Anomaly Detection MVP

本项目是一个“**本地轻量视觉感知 + 云端 VLM 复核**”的零售监控异常行为识别系统。

当前版本聚焦 MVP 落地，已经具备从视频输入到事件输出、持久化存储、可视化展示、VLM 复核与降级的完整链路。

## 1. 项目简介

系统分为两段：

1. **本地链路（实时、低成本）**  
   视频读取 -> 抽帧 -> 人体检测 -> 多目标跟踪 -> intrusion/dwell 候选事件生成
2. **云端复核（高层语义）**  
   对候选事件关键帧调用视觉 API 做复核，输出 `review_status/risk_level/final_decision`

当 VLM 调用失败时，系统自动降级为本地规则结果，不中断主流程。

## 2. 核心特性

- 本地视频处理：`StreamReader + FrameSampler`
- 人体检测：YOLO（`PersonDetector`）
- 多目标跟踪：轻量 IOU tracker（`ObjectTracker`）
- 规则引擎：`intrusion + dwell`
- 状态管理：`TrackStateManager`
- 候选事件统一调度：`CandidateEventGenerator`
- SQLite 持久化：`EventStore`
- OpenCV 可视化叠加：zone/track/event/status
- Streamlit 展示页：上传视频、运行处理、查看历史事件
- VLM 子模块：prompt 构建、API 调用、响应解析、复核编排
- 失败降级：`final_decision=local_only`

## 3. 系统架构概览

可按分层理解：

- 视频输入层：`src/video/*`
- 本地感知层：`src/vision/*`
- 规则与状态层：`src/event/*`
- 复核层：`src/vlm/*`
- 存储与可视化层：`src/service/*` + `src/utils/*`
- 编排入口：`src/main.py`
- 展示层：`src/ui/streamlit_app.py`

## 4. 项目目录结构

```text
person_item/
├─ src/
│  ├─ main.py
│  ├─ video/
│  │  ├─ stream_reader.py
│  │  └─ frame_sampler.py
│  ├─ vision/
│  │  ├─ detector.py
│  │  ├─ tracker.py
│  │  └─ geometry.py
│  ├─ event/
│  │  ├─ roi_rules.py
│  │  ├─ dwell_rules.py
│  │  ├─ state_manager.py
│  │  └─ candidate_generator.py
│  ├─ service/
│  │  └─ event_store.py
│  ├─ utils/
│  │  └─ vis.py
│  ├─ vlm/
│  │  ├─ client.py
│  │  ├─ prompt_builder.py
│  │  ├─ parser.py
│  │  ├─ reviewer.py
│  │  └─ fallback.py
│  └─ ui/
│     └─ streamlit_app.py
├─ scripts/
│  ├─ init_db.py
│  └─ run_local_demo.py
├─ docs/
│  └─ system_design.md
├─ tests/
│  ├─ test_event_round3.py
│  ├─ test_event_store.py
│  ├─ test_vlm_module.py
│  ├─ test_vlm_parser.py
│  ├─ test_fallback.py
│  └─ test_pipeline_integration.py
├─ .env.example
└─ your_video.mp4
```

## 5. 环境依赖与安装

建议 Python 3.10+。

核心依赖（按当前代码）：

- `opencv-python`
- `numpy`
- `ultralytics`
- `requests`
- `streamlit`
- `pytest`

示例安装：

```bash
pip install opencv-python numpy ultralytics requests streamlit pytest
```

## 6. 配置 VLM（.env）

复制模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
VLM_BASE_URL=https://your-api-host/v1
VLM_API_KEY=YOUR_API_KEY
VLM_MODEL_NAME=your-vision-model
VLM_TIMEOUT_SEC=20
VLM_MAX_RETRY=2
```

> 若不启用 VLM，可不配置以上字段，系统会走本地链路。

## 7. 初始化数据库

```bash
python scripts/init_db.py
```

## 8. 运行本地 demo（命令行）

```bash
python scripts/run_local_demo.py
```

输出包括：

- 控制台事件日志
- 可选处理后视频文件
- SQLite 事件记录

## 9. 启动 Streamlit 页面

```bash
python -m streamlit run src/ui/streamlit_app.py
```

页面支持：

- 上传视频
- 调整关键参数
- 开关 VLM 复核
- 展示历史事件表

## 10. 启用/关闭 VLM 复核

- 关闭：`enable_vlm_review=False`，仅本地规则输出
- 开启：`enable_vlm_review=True`，候选事件会增加 VLM 复核步骤
- 失败降级：VLM 调用失败时自动输出 `final_decision=local_only`

## 11. 主要事件类型

- `intrusion`：进入敏感区域
- `dwell`：在服务/关注区域长时间停留

## 12. 输出字段说明（关键）

最终事件（简化）：

- `event_type`：`intrusion` / `dwell`
- `confidence_local`：本地规则置信度
- `review_status`：`success` / `failed` / `skipped`
- `confidence_vlm`：VLM 置信度（可空）
- `risk_level`：`low` / `medium` / `high` / `unknown`
- `explanation`：复核说明
- `final_decision`：`abnormal` / `normal` / `local_only`

## 13. 当前限制与后续规划

当前限制：

- 关键帧提取为“当前帧快照”，未做 clip 级复核
- 事件融合逻辑偏规则型，尚未做复杂多信号融合
- 未接入生产级任务队列与服务化部署

后续规划：

- 多关键帧复核与缓存去重
- 事件详情页与片段导出
- 更细粒度融合策略（local + VLM）
- 更完善的回归测试与监控

## 14. 测试方式

运行工程回归测试：

```bash
python3 -m unittest tests/test_event_round3.py tests/test_event_store.py tests/test_vlm_module.py -v
```

运行 pytest（第八轮新增）：

```bash
pytest -q tests/test_vlm_parser.py tests/test_fallback.py tests/test_pipeline_integration.py
```

## 15. License

MIT (placeholder)  
如需开源发布，请在仓库根目录补充正式 `LICENSE` 文件。
