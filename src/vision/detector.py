"""Person detector module based on Ultralytics YOLO."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore[assignment]


class PersonDetector:
    """Detect person instances from a single image using YOLO.

    Output format:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "score": float,
                "class_id": int,
                "class_name": "person"
            }
        ]
    """

    PERSON_CLASS_ID = 0
    PERSON_CLASS_NAME = "person"

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        device: str = "cuda",
    ) -> None:
        """Initialize detector settings.

        Args:
            model_name: YOLO model path/name.
            conf_threshold: Confidence threshold in [0, 1].
            device: Preferred device (e.g. "cuda", "cpu", "cuda:0").
        """
        self.model_name = model_name
        self.conf_threshold = float(max(0.0, min(1.0, conf_threshold)))
        self.device = device

        self._model: Any | None = None
        self._resolved_device: str = "cpu"

    def load_model(self) -> None:
        """Load YOLO model and resolve runtime device.

        Raises:
            RuntimeError: if ultralytics is not installed or model loading fails.
        """
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Please install with: pip install ultralytics"
            )

        self._resolved_device = self._resolve_device(self.device)

        try:
            self._model = YOLO(self.model_name)
        except Exception as exc:
            self._model = None
            raise RuntimeError(f"Failed to load YOLO model '{self.model_name}': {exc}") from exc

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run person detection on a single image.

        Args:
            image: Input image in ndarray format.

        Returns:
            List of person detection dictionaries.
            Returns empty list for invalid input or inference failure.
        """
        if image is None or not isinstance(image, np.ndarray):
            return []
        if image.size == 0:
            return []
        if self._model is None:
            return []

        try:
            results = self._model.predict(
                source=image,
                conf=self.conf_threshold,
                device=self._resolved_device,
                verbose=False,
            )
        except Exception:
            # Gracefully degrade once to CPU if non-CPU inference fails.
            if self._resolved_device != "cpu":
                self._resolved_device = "cpu"
                try:
                    results = self._model.predict(
                        source=image,
                        conf=self.conf_threshold,
                        device=self._resolved_device,
                        verbose=False,
                    )
                except Exception:
                    return []
            else:
                return []

        return self._parse_person_results(results)

    def _resolve_device(self, preferred: str) -> str:
        """Resolve available device with safe CPU fallback."""
        p = (preferred or "cpu").lower()

        if p.startswith("cuda"):
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                return preferred
            return "cpu"

        if p == "mps":
            if (
                torch is not None
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"
            return "cpu"

        return "cpu" if p not in {"cpu"} else preferred

    def _parse_person_results(self, results: Any) -> list[dict[str, Any]]:
        """Extract and normalize person detections from YOLO outputs."""
        detections: list[dict[str, Any]] = []

        if not results:
            return detections

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy = getattr(boxes, "xyxy", None)
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)
            if xyxy is None or conf is None or cls is None:
                continue

            try:
                xyxy_np = xyxy.cpu().numpy()
                conf_np = conf.cpu().numpy()
                cls_np = cls.cpu().numpy().astype(int)
            except Exception:
                continue

            for i in range(len(cls_np)):
                class_id = int(cls_np[i])
                if class_id != self.PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = [float(v) for v in xyxy_np[i].tolist()]
                score = float(conf_np[i])

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class_id": class_id,
                        "class_name": self.PERSON_CLASS_NAME,
                    }
                )

        return detections
