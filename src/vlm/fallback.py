"""Fallback and fusion logic for local event + VLM review."""

from __future__ import annotations

from typing import Any


def fuse_event_with_review(
    candidate_event: dict[str, Any], review_result: dict[str, Any] | None
) -> dict[str, Any]:
    """Fuse local candidate event with VLM review result.

    Rules:
    - review_result is None -> local_only
    - review_status == failed -> local_only
    - review_status == success:
        - is_abnormal is True -> abnormal
        - is_abnormal is False -> normal
        - otherwise -> local_only
    """
    event = {
        "event_type": candidate_event.get("event_type"),
        "track_id": candidate_event.get("track_id"),
        "zone_name": candidate_event.get("zone_name"),
        "start_time": candidate_event.get("start_time"),
        "current_time": candidate_event.get("current_time"),
        "duration": candidate_event.get("duration"),
        "confidence_local": candidate_event.get("confidence_local"),
        "review_status": "skipped",
        "confidence_vlm": None,
        "risk_level": "unknown",
        "explanation": "",
        "final_decision": "local_only",
    }

    if review_result is None:
        event["review_status"] = "skipped"
        event["explanation"] = "VLM review not provided; fallback to local decision."
        return event

    review_status = str(review_result.get("review_status", "failed")).lower()
    event["review_status"] = review_status if review_status in {"success", "failed"} else "failed"
    event["confidence_vlm"] = review_result.get("confidence_vlm")
    event["risk_level"] = str(review_result.get("risk_level", "unknown"))
    event["explanation"] = str(review_result.get("explanation", ""))

    if event["review_status"] != "success":
        event["final_decision"] = "local_only"
        if not event["explanation"]:
            event["explanation"] = "VLM review failed; fallback to local decision."
        return event

    is_abnormal = review_result.get("is_abnormal")
    if is_abnormal is True:
        event["final_decision"] = "abnormal"
    elif is_abnormal is False:
        event["final_decision"] = "normal"
    else:
        event["final_decision"] = "local_only"

    return event
