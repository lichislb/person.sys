"""Prompt builder for VLM review."""

from __future__ import annotations

import json
from typing import Any


class PromptBuilder:
    """Build prompts for retail anomaly candidate review."""

    def build_review_prompt(
        self,
        candidate_event: dict[str, Any],
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        """Construct a compact JSON-oriented review prompt."""
        event_type = str(candidate_event.get("event_type", "unknown"))
        zone_name = str(candidate_event.get("zone_name", "unknown"))
        track_id = candidate_event.get("track_id", "unknown")
        duration = candidate_event.get("duration", "unknown")
        local_conf = candidate_event.get("confidence_local", "unknown")
        start_time = candidate_event.get("start_time", "unknown")
        current_time = candidate_event.get("current_time", "unknown")

        event_hint = self._event_hint(event_type)
        context_text = ""
        if extra_context:
            context_text = json.dumps(extra_context, ensure_ascii=False)

        return (
            "You are a retail-security visual reviewer.\n"
            "Task: verify whether this candidate alert is truly abnormal.\n\n"
            f"Candidate event:\n"
            f"- event_type: {event_type}\n"
            f"- track_id: {track_id}\n"
            f"- zone_name: {zone_name}\n"
            f"- start_time: {start_time}\n"
            f"- current_time: {current_time}\n"
            f"- duration: {duration}\n"
            f"- confidence_local: {local_conf}\n"
            f"- event_hint: {event_hint}\n\n"
            f"Extra context (optional): {context_text if context_text else 'N/A'}\n\n"
            "Return ONLY one JSON object with fields:\n"
            "{\n"
            '  "is_abnormal": true/false,\n'
            '  "abnormal_type": "intrusion"|"dwell"|"unknown",\n'
            '  "risk_level": "low"|"medium"|"high"|"unknown",\n'
            '  "confidence_vlm": 0.0~1.0,\n'
            '  "explanation": "short reason"\n'
            "}\n"
            "No markdown, no extra text."
        )

    @staticmethod
    def _event_hint(event_type: str) -> str:
        t = event_type.lower().strip()
        if t == "intrusion":
            return "Check if person entered a restricted/sensitive area."
        if t == "dwell":
            return "Check if person stayed unusually long in a service/dwell zone."
        return "Check if behavior is abnormal in context."
