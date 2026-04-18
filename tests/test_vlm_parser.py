"""Pytest tests for src.vlm.parser.ResponseParser."""

from __future__ import annotations

from src.vlm.parser import ResponseParser


def test_parse_dict_response_openai_style() -> None:
    parser = ResponseParser()
    raw = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"is_abnormal": true, "abnormal_type": "intrusion", '
                        '"risk_level": "high", "confidence_vlm": 0.91, '
                        '"explanation": "entered restricted area"}'
                    )
                }
            }
        ]
    }
    out = parser.parse_review_response(raw)
    assert out["review_status"] == "success"
    assert out["abnormal_type"] == "intrusion"
    assert out["risk_level"] == "high"
    assert "entered restricted area" in out["explanation"]


def test_parse_plain_json_string() -> None:
    parser = ResponseParser()
    raw = (
        '{"is_abnormal": false, "abnormal_type": "unknown", '
        '"risk_level": "low", "confidence_vlm": 0.2, "explanation": "normal"}'
    )
    out = parser.parse_review_response(raw)
    assert out["review_status"] == "success"
    assert out["abnormal_type"] == "unknown"
    assert out["risk_level"] == "low"
    assert out["explanation"] == "normal"


def test_parse_wrapped_text_with_json() -> None:
    parser = ResponseParser()
    raw = """
    Analysis:
    ```json
    {
      "is_abnormal": true,
      "abnormal_type": "dwell",
      "risk_level": "medium",
      "confidence_vlm": 0.77,
      "explanation": "customer stayed too long"
    }
    ```
    """
    out = parser.parse_review_response(raw)
    assert out["review_status"] == "success"
    assert out["abnormal_type"] == "dwell"
    assert out["risk_level"] == "medium"
    assert "stayed too long" in out["explanation"]


def test_parse_invalid_or_empty_returns_failed() -> None:
    parser = ResponseParser()

    out_none = parser.parse_review_response(None)
    assert out_none["review_status"] == "failed"
    assert out_none["risk_level"] == "unknown"

    out_bad = parser.parse_review_response("non-json noisy text without clear labels")
    assert out_bad["review_status"] in {"failed", "success"}
    # parser should always return unified fields even when parse is weak.
    assert "abnormal_type" in out_bad
    assert "risk_level" in out_bad
    assert "explanation" in out_bad
