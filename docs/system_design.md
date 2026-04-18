# 系统设计说明（Engineering Design）

## 1. 系统目标

构建一个可落地的零售异常识别工程样例，满足：

- 本地轻量链路可实时运行
- 关键候选事件可调用云端 VLM 复核
- VLM 不可用时可自动降级，业务不中断
- 结果可存储、可展示、可追溯

## 2. 业务场景与异常事件定义

当前 MVP 聚焦两类事件：

1. `intrusion`：人员进入 restricted 区域
2. `dwell`：人员在 service/dwell 区域停留超阈值

本地规则负责“候选异常”召回，VLM 负责语义层复核。

## 3. 总体架构设计

### 3.1 视频输入层

- `src/video/stream_reader.py`：统一视频读取接口
- `src/video/frame_sampler.py`：抽帧控制

### 3.2 本地感知层

- `src/vision/detector.py`：人体检测（YOLO）
- `src/vision/tracker.py`：IOU 多目标跟踪
- `src/vision/geometry.py`：几何判定工具

### 3.3 事件规则层

- `src/event/roi_rules.py`：intrusion 规则
- `src/event/dwell_rules.py`：dwell 规则
- `src/event/state_manager.py`：跨帧状态记忆
- `src/event/candidate_generator.py`：统一候选事件生成

### 3.4 VLM 复核层

- `src/vlm/prompt_builder.py`：构建复核 prompt
- `src/vlm/client.py`：多模态 API 调用
- `src/vlm/parser.py`：响应解析与归一化
- `src/vlm/reviewer.py`：编排入口
- `src/vlm/fallback.py`：降级与最终决策融合

### 3.5 存储与展示层

- `src/service/event_store.py`：SQLite 存储
- `src/utils/vis.py`：OpenCV 绘制
- `src/ui/streamlit_app.py`：演示页面
- `src/main.py`：主流程编排

## 4. 关键模块职责

### stream_reader

读取视频源，输出标准 `frame_obj`：

```python
{
  "frame_id": int,
  "timestamp": float,
  "image": np.ndarray,
  "source_id": str
}
```

### detector

检测人体，输出人框列表：

```python
[
  {"bbox":[x1,y1,x2,y2], "score":float, "class_id":0, "class_name":"person"}
]
```

### tracker

将检测结果跨帧关联，输出 track：

```python
[
  {"track_id":int, "bbox":[...], "score":float, "class_name":"person", "center":[cx,cy]}
]
```

### roi_rules / dwell_rules

基于 track + zone + duration 生成候选事件。

### state_manager

维护 track 与 zone 的跨帧状态（enter_time、inside、trigger 标记等）。

### candidate_generator

统一调度 intrusion/dwell 引擎并汇总候选事件。

### event_store

持久化最终事件，支持查重、查询与历史展示。

### vlm 子模块

对候选事件关键帧执行“提示词构建 -> API调用 -> 响应解析”的闭环。

### fallback

融合本地事件与 VLM 复核结果，生成 `final_decision`，确保失败降级。

### main / streamlit

- `main.py`：端到端运行入口（检测、复核、融合、存储、可视化）
- `streamlit_app.py`：交互式演示入口

## 5. 核心数据流

1. 读取视频帧  
2. 抽帧判定  
3. 检测 + 跟踪  
4. 候选事件生成  
5. （可选）关键帧保存并调用 VLM 复核  
6. fallback 融合出最终事件  
7. 写入 SQLite  
8. 页面展示与历史查询

## 6. 关键数据结构

### 6.1 candidate_event（本地候选）

```python
{
  "event_type": "intrusion" | "dwell",
  "track_id": int,
  "zone_name": str,
  "start_time": float,
  "current_time": float,
  "duration": float,
  "confidence_local": float
}
```

### 6.2 review_result（VLM复核）

```python
{
  "review_status": "success" | "failed",
  "is_abnormal": bool | None,
  "abnormal_type": "intrusion" | "dwell" | "unknown" | None,
  "risk_level": "low" | "medium" | "high" | "unknown",
  "confidence_vlm": float | None,
  "explanation": str,
  "raw_response": dict | str | None
}
```

### 6.3 final_event（融合后）

```python
{
  "event_type": str,
  "track_id": int,
  "zone_name": str,
  "start_time": float,
  "current_time": float,
  "duration": float,
  "confidence_local": float,
  "review_status": "success" | "failed" | "skipped",
  "confidence_vlm": float | None,
  "risk_level": "low" | "medium" | "high" | "unknown",
  "explanation": str,
  "final_decision": "abnormal" | "normal" | "local_only"
}
```

## 7. VLM 接入策略说明

### 为什么不做本地大模型部署

- 部署成本高
- 资源要求高（显存/运维）
- 对 MVP 阶段迭代效率不友好

### 为什么采用“候选事件复核”

- 本地链路负责高召回、低延迟
- 云端复核只针对少量候选事件，成本更可控
- 兼顾性能与语义准确性

### 为什么需要 fallback 降级

- 云 API 可能超时、失败、限流
- 业务不能因外部依赖失效而停止
- 通过 `final_decision=local_only` 保证工程可用性

## 8. 当前工程边界与限制

- 尚未实现 clip 级关键片段导出与多帧聚合复核
- VLM 融合规则为第一版（可解释但较简单）
- 页面仍以演示为主，未做权限与多用户隔离
- 尚未服务化（未接 FastAPI/队列/任务调度）

## 9. 后续可扩展方向

- 多关键帧复核与缓存去重
- 更细粒度融合策略（基于历史与上下文）
- 事件详情页（截图、解释、轨迹回看）
- 端到端集成测试增强（mock VLM + mock I/O）
- 线上观测（失败率、延迟、召回/精度指标）
