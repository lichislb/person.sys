"""Minimal integration-style tests for local events + VLM fusion flow.

These tests intentionally avoid real video/YOLO/network/database.
They validate the key orchestration behavior around:
- candidate event outputs
- optional VLM review
- fallback fusion decisions
"""

from __future__ import annotations

from typing import Any

from src.vlm.fallback import fuse_event_with_review


def _run_event_fusion_flow(
    candidate_events: list[dict[str, Any]],
    enable_vlm_review: bool,
    reviewer: Any | None,
) -> list[dict[str, Any]]:
    """Small helper that mirrors the key decision block in main pipeline."""
    final_events: list[dict[str, Any]] = []
    for event in candidate_events:
        review_result: dict[str, Any] | None = None
        if enable_vlm_review and reviewer is not None:
            try:
                review_result = reviewer.review(
                    candidate_event=event,
                    image_paths=["fake_snapshot.jpg"],
                    extra_context=None,
                )
            except Exception as exc:
                review_result = {
                    "review_status": "failed",
                    "is_abnormal": None,
                    "abnormal_type": None,
                    "risk_level": "unknown",
                    "confidence_vlm": None,
                    "explanation": f"review error: {exc}",
                    "raw_response": None,
                }
        elif enable_vlm_review and reviewer is None:
            review_result = {
                "review_status": "failed",
                "is_abnormal": None,
                "abnormal_type": None,
                "risk_level": "unknown",
                "confidence_vlm": None,
                "explanation": "reviewer not initialized",
                "raw_response": None,
            }
        else:
            review_result = None

        final_events.append(fuse_event_with_review(event, review_result))
    return final_events


class _ReviewerSuccess:
    def review(
        self,
        candidate_event: dict[str, Any],
        image_paths: list[str] | str,
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = candidate_event, image_paths, extra_context
        return {
            "review_status": "success",
            "is_abnormal": True,
            "abnormal_type": "intrusion",
            "risk_level": "high",
            "confidence_vlm": 0.9,
            "explanation": "confirmed abnormal",
            "raw_response": {"ok": True},
        }


class _ReviewerFail:
    def review(
        self,
        candidate_event: dict[str, Any],
        image_paths: list[str] | str,
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        _ = candidate_event, image_paths, extra_context
        raise RuntimeError("api timeout")


def _candidate_events() -> list[dict[str, Any]]:
    return [
        {
            "event_type": "intrusion",
            "track_id": 1,
            "zone_name": "staff_only_zone",
            "start_time": 1.0,
            "current_time": 2.0,
            "duration": 1.0,
            "confidence_local": 0.8,
        }
    ]


def test_pipeline_flow_without_vlm_outputs_local_only() -> None:
    final_events = _run_event_fusion_flow(
        candidate_events=_candidate_events(),
        enable_vlm_review=False,
        reviewer=None,
    )
    assert len(final_events) == 1
    assert final_events[0]["review_status"] == "skipped"
    assert final_events[0]["final_decision"] == "local_only"


def test_pipeline_flow_with_vlm_success_outputs_review_fields() -> None:
    final_events = _run_event_fusion_flow(
        candidate_events=_candidate_events(),
        enable_vlm_review=True,
        reviewer=_ReviewerSuccess(),
    )
    assert len(final_events) == 1
    assert final_events[0]["review_status"] == "success"
    assert final_events[0]["risk_level"] == "high"
    assert final_events[0]["final_decision"] == "abnormal"


def test_pipeline_flow_with_vlm_failure_falls_back_local_only() -> None:
    final_events = _run_event_fusion_flow(
        candidate_events=_candidate_events(),
        enable_vlm_review=True,
        reviewer=_ReviewerFail(),
    )
    assert len(final_events) == 1
    assert final_events[0]["review_status"] == "failed"
    assert final_events[0]["final_decision"] == "local_only"
    assert "error" in final_events[0]["explanation"] or "failed" in final_events[0]["review_status"]
