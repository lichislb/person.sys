"""Unified reviewer entry for VLM submodule."""

from __future__ import annotations

from typing import Any

from src.vlm.client import VLMApiClient
from src.vlm.parser import ResponseParser
from src.vlm.prompt_builder import PromptBuilder


class VLMReviewer:
    """Orchestrate prompt building, API calling, and response parsing."""

    def __init__(
        self,
        client: VLMApiClient,
        prompt_builder: PromptBuilder,
        parser: ResponseParser,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder
        self.parser = parser

    def review(
        self,
        candidate_event: dict[str, Any],
        image_paths: str | list[str],
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a single review request and return unified structured output."""
        images = self._normalize_image_paths(image_paths)
        if not images:
            return self._failed_output("no valid image path provided", raw_response=None)

        prompt = self.prompt_builder.build_review_prompt(
            candidate_event=candidate_event,
            extra_context=extra_context,
        )
        metadata = self._build_metadata(candidate_event, extra_context)

        try:
            raw_response = self.client.review_images(
                image_paths=images,
                prompt=prompt,
                metadata=metadata,
            )
        except Exception as exc:
            return self._failed_output(str(exc), raw_response=None, prompt=prompt)

        parsed = self.parser.parse_review_response(raw_response)
        parsed["raw_response"] = raw_response
        parsed["prompt"] = prompt
        return parsed

    @staticmethod
    def _normalize_image_paths(image_paths: str | list[str]) -> list[str]:
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not isinstance(image_paths, list):
            return []
        return [str(p).strip() for p in image_paths if str(p).strip()]

    @staticmethod
    def _build_metadata(
        candidate_event: dict[str, Any], extra_context: dict[str, Any] | None
    ) -> dict[str, Any]:
        metadata = {
            "event_type": candidate_event.get("event_type"),
            "track_id": candidate_event.get("track_id"),
            "zone_name": candidate_event.get("zone_name"),
            "duration": candidate_event.get("duration"),
            "confidence_local": candidate_event.get("confidence_local"),
        }
        if extra_context:
            metadata["extra_context"] = extra_context
        return metadata

    @staticmethod
    def _failed_output(
        reason: str,
        raw_response: dict | str | None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "review_status": "failed",
            "is_abnormal": None,
            "abnormal_type": None,
            "risk_level": "unknown",
            "confidence_vlm": None,
            "explanation": reason,
            "raw_response": raw_response,
        }
        if prompt is not None:
            out["prompt"] = prompt
        return out
