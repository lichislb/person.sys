"""Validation tests for round-6 VLM submodule.

Run:
    python3 -m unittest tests/test_vlm_module.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.vlm.client import VLMApiClient
from src.vlm.parser import ResponseParser
from src.vlm.prompt_builder import PromptBuilder
from src.vlm.reviewer import VLMReviewer


class _FakeClientSuccess:
    def review_images(
        self, image_paths: list[str], prompt: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        _ = image_paths, prompt, metadata
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "is_abnormal": True,
                                "abnormal_type": "intrusion",
                                "risk_level": "high",
                                "confidence_vlm": 0.92,
                                "explanation": "Person entered restricted zone.",
                            }
                        )
                    }
                }
            ]
        }


class _FakeClientFail:
    def review_images(
        self, image_paths: list[str], prompt: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        _ = image_paths, prompt, metadata
        raise RuntimeError("simulated API failure")


class TestPromptBuilder(unittest.TestCase):
    def test_prompt_contains_core_fields(self) -> None:
        builder = PromptBuilder()
        event = {
            "event_type": "dwell",
            "track_id": 7,
            "zone_name": "checkout_zone",
            "start_time": 10.0,
            "current_time": 18.0,
            "duration": 8.0,
            "confidence_local": 0.7,
        }
        prompt = builder.build_review_prompt(event, extra_context={"camera_id": "cam_01"})
        self.assertIn("retail-security visual reviewer", prompt)
        self.assertIn("event_type: dwell", prompt)
        self.assertIn("zone_name: checkout_zone", prompt)
        self.assertIn("duration: 8.0", prompt)
        self.assertIn("Return ONLY one JSON object", prompt)


class TestResponseParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ResponseParser()

    def test_parse_openai_style_dict(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "content": '{"is_abnormal": true, "abnormal_type":"intrusion", "risk_level":"high", "confidence_vlm":0.91, "explanation":"entered restricted zone"}'
                    }
                }
            ]
        }
        out = self.parser.parse_review_response(raw)
        self.assertEqual(out["review_status"], "success")
        self.assertTrue(out["is_abnormal"])
        self.assertEqual(out["abnormal_type"], "intrusion")
        self.assertEqual(out["risk_level"], "high")
        self.assertAlmostEqual(out["confidence_vlm"], 0.91, places=3)

    def test_parse_wrapped_json_text(self) -> None:
        raw = "Model analysis:\n```json\n{\"is_abnormal\": false, \"abnormal_type\": \"unknown\", \"risk_level\": \"low\", \"confidence_vlm\": 0.2, \"explanation\": \"normal behavior\"}\n```"
        out = self.parser.parse_review_response(raw)
        self.assertEqual(out["review_status"], "success")
        self.assertFalse(out["is_abnormal"])
        self.assertEqual(out["risk_level"], "low")

    def test_parse_invalid_text_fallback(self) -> None:
        raw = "This appears suspicious with high risk intrusion behavior."
        out = self.parser.parse_review_response(raw)
        self.assertEqual(out["review_status"], "failed")
        self.assertTrue(out["is_abnormal"])
        self.assertEqual(out["abnormal_type"], "intrusion")
        self.assertEqual(out["risk_level"], "high")

    def test_parse_none(self) -> None:
        out = self.parser.parse_review_response(None)
        self.assertEqual(out["review_status"], "failed")
        self.assertIsNone(out["is_abnormal"])
        self.assertEqual(out["risk_level"], "unknown")

    def test_normalization_mid_and_confidence_bounds(self) -> None:
        raw = '{"is_abnormal":"true","abnormal_type":"abc","risk_level":"moderate","confidence_vlm":1.5,"explanation":"x"}'
        out = self.parser.parse_review_response(raw)
        self.assertEqual(out["review_status"], "success")
        self.assertTrue(out["is_abnormal"])
        self.assertEqual(out["abnormal_type"], "unknown")
        self.assertEqual(out["risk_level"], "medium")
        self.assertEqual(out["confidence_vlm"], 1.0)


class TestReviewer(unittest.TestCase):
    def setUp(self) -> None:
        self.event = {
            "event_type": "intrusion",
            "track_id": 12,
            "zone_name": "staff_only_zone",
            "start_time": 12.4,
            "current_time": 15.1,
            "duration": 2.7,
            "confidence_local": 0.78,
        }
        self.builder = PromptBuilder()
        self.parser = ResponseParser()

    def test_review_success_single_image_path(self) -> None:
        reviewer = VLMReviewer(
            client=_FakeClientSuccess(), prompt_builder=self.builder, parser=self.parser
        )
        out = reviewer.review(self.event, "frame_001.jpg")
        self.assertEqual(out["review_status"], "success")
        self.assertTrue(out["is_abnormal"])
        self.assertIn("prompt", out)
        self.assertIn("raw_response", out)

    def test_review_failure_from_client(self) -> None:
        reviewer = VLMReviewer(
            client=_FakeClientFail(), prompt_builder=self.builder, parser=self.parser
        )
        out = reviewer.review(self.event, ["frame_001.jpg", "frame_002.jpg"])
        self.assertEqual(out["review_status"], "failed")
        self.assertIn("simulated API failure", out["explanation"])

    def test_review_failure_invalid_image_paths(self) -> None:
        reviewer = VLMReviewer(
            client=_FakeClientSuccess(), prompt_builder=self.builder, parser=self.parser
        )
        out = reviewer.review(self.event, [])
        self.assertEqual(out["review_status"], "failed")
        self.assertIn("no valid image path", out["explanation"])


class TestClientLocalBehavior(unittest.TestCase):
    def test_encode_image_and_missing_path(self) -> None:
        client = VLMApiClient(
            base_url="https://example.com/v1",
            api_key="x",
            model_name="m",
            timeout_sec=10,
            max_retry=1,
        )

        with self.assertRaises(FileNotFoundError):
            client._encode_image("not_exists.jpg")

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.jpg"
            p.write_bytes(b"fakejpeg")
            encoded = client._encode_image(str(p))
            self.assertIn("data:image", encoded["data_url"])
            self.assertTrue(encoded["data_url"].startswith("data:"))

    def test_build_payload_structure(self) -> None:
        client = VLMApiClient(
            base_url="https://example.com/v1",
            api_key="x",
            model_name="m",
            timeout_sec=10,
            max_retry=1,
        )
        payload = client._build_payload(
            encoded_images=[
                {"path": "a.jpg", "mime_type": "image/jpeg", "data_url": "data:image/jpeg;base64,AAA"}
            ],
            prompt="hello",
            metadata={"event_type": "intrusion"},
        )
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "text")
        self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")
        self.assertEqual(payload["metadata"]["event_type"], "intrusion")


if __name__ == "__main__":
    unittest.main()
