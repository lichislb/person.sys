"""Lightweight multi-object tracker based on IoU association.

This module provides a stable and dependency-light tracker implementation for
MVP stage. The public output format is unified and can be replaced by another
tracker backend (e.g. ByteTrack) later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.vision.geometry import bbox_center, compute_iou


@dataclass
class _TrackState:
    """Internal track state."""

    track_id: int
    bbox: list[float]
    score: float
    class_name: str
    lost_count: int = 0


class ObjectTracker:
    """Simple IoU-based tracker for person detections.

    Input detections are expected to be compatible with detector output:
        {
            "bbox": [x1, y1, x2, y2],
            "score": float,
            "class_id": int,
            "class_name": "person"
        }
    """

    def __init__(
        self,
        max_lost: int = 30,
        min_box_area: float = 100.0,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_lost = max(0, int(max_lost))
        self.min_box_area = float(max(0.0, min_box_area))
        self.iou_threshold = float(max(0.0, min(1.0, iou_threshold)))

        self._next_track_id: int = 1
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        """Reset tracker state."""
        self._next_track_id = 1
        self._tracks.clear()

    def update(self, detections: list[dict[str, Any]], image: np.ndarray | None = None) -> list[dict[str, Any]]:
        """Update tracker with current frame detections.

        Args:
            detections: Detector outputs from current frame.
            image: Reserved for future backend compatibility.

        Returns:
            Unified track output list:
                [
                    {
                        "track_id": int,
                        "bbox": [x1, y1, x2, y2],
                        "score": float,
                        "class_name": "person",
                        "center": [cx, cy]
                    }
                ]
        """
        _ = image  # not used in current IoU tracker

        valid_detections = self._preprocess_detections(detections or [])

        if not valid_detections:
            self._mark_all_tracks_lost()
            self._remove_expired_tracks()
            return self._export_tracks()

        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(valid_detections)))
        matches: list[tuple[int, int]] = []  # (track_id, det_idx)

        # Greedy match by descending IoU over all pairs.
        pair_candidates: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            track_box = self._tracks[track_id].bbox
            for det_idx, det in enumerate(valid_detections):
                iou = compute_iou(track_box, det["bbox"])
                if iou >= self.iou_threshold:
                    pair_candidates.append((iou, track_id, det_idx))

        pair_candidates.sort(key=lambda x: x[0], reverse=True)
        for _, track_id, det_idx in pair_candidates:
            if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                continue
            matches.append((track_id, det_idx))
            unmatched_tracks.remove(track_id)
            unmatched_dets.remove(det_idx)

        # Update matched tracks.
        for track_id, det_idx in matches:
            det = valid_detections[det_idx]
            state = self._tracks[track_id]
            state.bbox = det["bbox"]
            state.score = det["score"]
            state.class_name = det["class_name"]
            state.lost_count = 0

        # Mark unmatched tracks as lost.
        for track_id in unmatched_tracks:
            self._tracks[track_id].lost_count += 1

        # Spawn new tracks for unmatched detections.
        for det_idx in unmatched_dets:
            det = valid_detections[det_idx]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                bbox=det["bbox"],
                score=det["score"],
                class_name=det["class_name"],
                lost_count=0,
            )

        self._remove_expired_tracks()
        return self._export_tracks()

    def _preprocess_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate and filter raw detections."""
        valid: list[dict[str, Any]] = []
        for det in detections:
            if not isinstance(det, dict):
                continue

            bbox_raw = det.get("bbox")
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in bbox_raw]
            except (TypeError, ValueError):
                continue

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < self.min_box_area:
                continue

            score = det.get("score", 0.0)
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0

            class_name = det.get("class_name", "person")
            if not isinstance(class_name, str):
                class_name = "person"

            valid.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": score_f,
                    "class_name": class_name,
                }
            )
        return valid

    def _mark_all_tracks_lost(self) -> None:
        """Increase lost counter for all active tracks."""
        for state in self._tracks.values():
            state.lost_count += 1

    def _remove_expired_tracks(self) -> None:
        """Remove tracks that have been lost for too long."""
        expired_ids = [tid for tid, state in self._tracks.items() if state.lost_count > self.max_lost]
        for tid in expired_ids:
            self._tracks.pop(tid, None)

    def _export_tracks(self) -> list[dict[str, Any]]:
        """Export active tracks to unified public format."""
        outputs: list[dict[str, Any]] = []
        for track_id in sorted(self._tracks.keys()):
            state = self._tracks[track_id]
            cx, cy = bbox_center(state.bbox)
            outputs.append(
                {
                    "track_id": state.track_id,
                    "bbox": [float(v) for v in state.bbox],
                    "score": float(state.score),
                    "class_name": state.class_name,
                    "center": [float(cx), float(cy)],
                }
            )
        return outputs
