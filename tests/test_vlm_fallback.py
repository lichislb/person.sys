"""Tests for VLM fallback fusion logic."""

from __future__ import annotations

import unittest

from src.vlm.fallback import fuse_event_with_review


class TestVLMFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.event = {
            "event_type": "intrusion",
            "track_id": 1,
            "zone_name": "staff_only_zone",
            "start_time": 1.0,
            "current_time": 2.0,
            "duration": 1.0,
            "confidence_local": 0.7,
        }

    def test_none_review_result_local_only(self) -> None:
        out = fuse_event_with_review(self.event, None)
        self.assertEqual(out["review_status"], "skipped")
        self.assertEqual(out["final_decision"], "local_only")

    def test_failed_review_local_only(self) -> None:
        out = fuse_event_with_review(
            self.event,
            {
                "review_status": "failed",
                "is_abnormal": None,
                "risk_level": "unknown",
                "confidence_vlm": None,
                "explanation": "api timeout",
            },
        )
        self.assertEqual(out["review_status"], "failed")
        self.assertEqual(out["final_decision"], "local_only")

    def test_success_true_abnormal(self) -> None:
        out = fuse_event_with_review(
            self.event,
            {
                "review_status": "success",
                "is_abnormal": True,
                "risk_level": "high",
                "confidence_vlm": 0.92,
                "explanation": "confirmed",
            },
        )
        self.assertEqual(out["review_status"], "success")
        self.assertEqual(out["final_decision"], "abnormal")
        self.assertEqual(out["risk_level"], "high")

    def test_success_false_normal(self) -> None:
        out = fuse_event_with_review(
            self.event,
            {
                "review_status": "success",
                "is_abnormal": False,
                "risk_level": "low",
                "confidence_vlm": 0.2,
                "explanation": "normal",
            },
        )
        self.assertEqual(out["review_status"], "success")
        self.assertEqual(out["final_decision"], "normal")


if __name__ == "__main__":
    unittest.main()
