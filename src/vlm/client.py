"""Generic visual-language API client."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

import requests


class VLMApiClient:
    """Client for single-shot image review via a remote multimodal API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout_sec: int = 20,
        max_retry: int = 2,
    ) -> None:
        self.base_url = str(base_url).strip()
        self.api_key = str(api_key).strip()
        self.model_name = str(model_name).strip()
        self.timeout_sec = int(max(1, timeout_sec))
        self.max_retry = int(max(0, max_retry))

    def review_images(
        self,
        image_paths: list[str],
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call the visual API and return raw response dict."""
        if not image_paths:
            raise ValueError("image_paths is empty")
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt is empty")

        normalized_paths = [str(p) for p in image_paths if str(p).strip()]
        if not normalized_paths:
            raise ValueError("image_paths has no valid path")

        encoded_images = [self._encode_image(path) for path in normalized_paths]
        payload = self._build_payload(encoded_images, prompt, metadata=metadata)
        return self._post_with_retry(payload)

    def _encode_image(self, image_path: str) -> dict[str, str]:
        """Read image file and encode to data URL."""
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")

        data = path.read_bytes()
        if not data:
            raise ValueError(f"image is empty: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = "image/jpeg"

        encoded = base64.b64encode(data).decode("utf-8")
        return {
            "path": str(path),
            "mime_type": mime_type,
            "data_url": f"data:{mime_type};base64,{encoded}",
        }

    def _build_payload(
        self,
        encoded_images: list[dict[str, str]],
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a generic OpenAI-style multimodal payload."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in encoded_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item["data_url"]},
                }
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
        }
        if metadata:
            payload["metadata"] = metadata
        return payload

    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST payload with retry and timeout handling."""
        endpoint = self._resolve_endpoint(self.base_url)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for _ in range(self.max_retry + 1):
            try:
                resp = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_sec,
                )
                resp.raise_for_status()
                try:
                    return resp.json()
                except ValueError:
                    return {"text": resp.text}
            except Exception as exc:  # requests exceptions + json parse errors
                last_error = exc

        raise RuntimeError(f"visual API request failed: {last_error}")

    @staticmethod
    def _resolve_endpoint(base_url: str) -> str:
        """Resolve to /chat/completions if base url is root-like."""
        raw = base_url.strip().rstrip("/")
        if raw.endswith("/chat/completions"):
            return raw
        return f"{raw}/chat/completions"
