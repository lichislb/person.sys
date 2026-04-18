"""Pytest tests for src.vlm.fallback."""

from __future__ import annotations

from src.vlm.fallback import fuse_event_with_review


def _candidate_event() -> dict:
    return {
        "event_type": "intrusion",
        "track_id": 12,
        "zone_name": "staff_only_zone",
        "start_time": 12.4,
        "current_time": 15.1,
        "duration": 2.7,
        "confidence_local": 0.78,
    }


def test_fallback_when_review_none() -> None:
    out = fuse_event_with_review(_candidate_event(), None)
    assert out["final_decision"] == "local_only"
    assert out["review_status"] == "skipped"
    assert isinstance(out["explanation"], str)
    assert out["risk_level"] == "unknown"


def test_fallback_when_review_failed() -> None:
    review = {
        "review_status": "failed",
        "is_abnormal": None,
        "risk_level": "unknown",
        "confidence_vlm": None,
        "explanation": "timeout",
    }
    out = fuse_event_with_review(_candidate_event(), review)
    assert out["final_decision"] == "local_only"
    assert out["review_status"] == "failed"
    assert out["explanation"] == "timeout"
    assert out["risk_level"] == "unknown"


def test_fallback_success_abnormal_true() -> None:
    review = {
        "review_status": "success",
        "is_abnormal": True,
        "risk_level": "high",
        "confidence_vlm": 0.95,
        "explanation": "confirmed intrusion",
    }
    out = fuse_event_with_review(_candidate_event(), review)
    assert out["final_decision"] == "abnormal"
    assert out["review_status"] == "success"
    assert out["risk_level"] == "high"
    assert "confirmed" in out["explanation"]


def test_fallback_success_abnormal_false() -> None:
    review = {
        "review_status": "success",
        "is_abnormal": False,
        "risk_level": "low",
        "confidence_vlm": 0.2,
        "explanation": "normal behavior",
    }
    out = fuse_event_with_review(_candidate_event(), review)
    assert out["final_decision"] == "normal"
    assert out["review_status"] == "success"
    assert out["risk_level"] == "low"
    assert "normal" in out["explanation"]
