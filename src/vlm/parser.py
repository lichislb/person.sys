"""Response parser for VLM review outputs."""

from __future__ import annotations

import json
import re
from typing import Any


class ResponseParser:
    """Parse raw VLM API responses into a unified review schema."""

    def parse_review_response(self, raw_response: dict | str | None) -> dict[str, Any]:
        """Parse and normalize VLM response. Never raises."""
        result = self._failed_result(raw_response)
        if raw_response is None:
            return result

        try:
            text = self._extract_text_from_response(raw_response)
            payload = None

            if isinstance(raw_response, dict) and self._looks_like_review_json(raw_response):
                payload = raw_response
            if payload is None:
                payload = self._extract_json_block(text)

            if payload is None:
                return self._fallback_from_text(result, text)

            result["review_status"] = "success"
            result["is_abnormal"] = self._normalize_bool(payload.get("is_abnormal"))
            result["abnormal_type"] = self._normalize_abnormal_type(payload.get("abnormal_type"))
            result["risk_level"] = self._normalize_risk_level(payload.get("risk_level"))
            result["confidence_vlm"] = self._normalize_confidence(payload.get("confidence_vlm"))
            result["explanation"] = str(payload.get("explanation", "")).strip()

            if result["explanation"] == "":
                result["explanation"] = text.strip()
            return result
        except Exception:
            return result

    def _extract_text_from_response(self, raw_response: dict | str) -> str:
        """Extract text from common OpenAI-style response formats."""
        if isinstance(raw_response, str):
            return raw_response

        if not isinstance(raw_response, dict):
            return str(raw_response)

        # OpenAI-like: choices[0].message.content
        choices = raw_response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: list[str] = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        texts.append(str(c.get("text", "")))
                return "\n".join([t for t in texts if t])

        if "text" in raw_response:
            return str(raw_response.get("text", ""))

        return json.dumps(raw_response, ensure_ascii=False)

    def _extract_json_block(self, text: str) -> dict[str, Any] | None:
        """Try parse direct JSON or first {...} block."""
        if not text:
            return None

        # direct JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # fenced code block
        code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_match:
            block = code_match.group(1)
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # first JSON-like braces
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            block = brace_match.group(0)
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    @staticmethod
    def _looks_like_review_json(obj: dict[str, Any]) -> bool:
        keys = set(obj.keys())
        return bool(
            {"is_abnormal", "abnormal_type", "risk_level", "confidence_vlm", "explanation"}
            & keys
        )

    def _fallback_from_text(self, result: dict[str, Any], text: str) -> dict[str, Any]:
        """Fallback keyword parsing when no valid JSON exists."""
        t = (text or "").lower()
        result["review_status"] = "failed"
        result["explanation"] = text.strip() if text else ""

        if "intrusion" in t:
            result["abnormal_type"] = "intrusion"
        elif "dwell" in t:
            result["abnormal_type"] = "dwell"
        else:
            result["abnormal_type"] = "unknown"

        if any(k in t for k in ["abnormal", "violation", "suspicious", "alert"]):
            result["is_abnormal"] = True
        elif any(k in t for k in ["normal", "no issue", "not abnormal"]):
            result["is_abnormal"] = False

        if "high" in t:
            result["risk_level"] = "high"
        elif "medium" in t or "moderate" in t:
            result["risk_level"] = "medium"
        elif "low" in t:
            result["risk_level"] = "low"
        else:
            result["risk_level"] = "unknown"
        return result

    @staticmethod
    def _normalize_bool(v: Any) -> bool | None:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.lower().strip()
            if s in {"true", "yes", "1"}:
                return True
            if s in {"false", "no", "0"}:
                return False
        return None

    @staticmethod
    def _normalize_risk_level(v: Any) -> str:
        s = str(v if v is not None else "").lower().strip()
        if s in {"low", "medium", "high"}:
            return s
        if s in {"mid", "moderate"}:
            return "medium"
        return "unknown"

    @staticmethod
    def _normalize_abnormal_type(v: Any) -> str | None:
        s = str(v if v is not None else "").lower().strip()
        if s in {"intrusion", "dwell"}:
            return s
        if s:
            return "unknown"
        return None

    @staticmethod
    def _normalize_confidence(v: Any) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f < 0:
            return 0.0
        if f > 1:
            return 1.0
        return f

    @staticmethod
    def _failed_result(raw_response: dict | str | None) -> dict[str, Any]:
        return {
            "review_status": "failed",
            "is_abnormal": None,
            "abnormal_type": None,
            "risk_level": "unknown",
            "confidence_vlm": None,
            "explanation": "",
            "raw_response": raw_response,
        }
