"""Video stream reader module.

This module provides a unified reader interface for local files (MVP stage)
while preserving an API that can be extended to RTSP streams later.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class StreamReader:
    """Read frames from a video source and return a unified frame payload.

    Notes:
    - Current MVP is optimized for local video files (e.g. mp4).
    - The class interface is intentionally generic for future RTSP extension.
    - Timestamp is estimated from FPS and frame index when possible.
    """

    def __init__(self, source: str, loop: bool = False, reconnect: bool = False) -> None:
        """Initialize a stream reader.

        Args:
            source: Video source path/uri.
            loop: Whether to restart from frame 0 when reaching EOF.
            reconnect: Reserved for unstable stream reconnection (future RTSP use).
        """
        self.source: str = source
        self.loop: bool = loop
        self.reconnect: bool = reconnect

        self._cap: cv2.VideoCapture | None = None
        self._frame_id: int = 0
        self._fps: float = 0.0

    def open(self) -> bool:
        """Open the video source.

        Returns:
            True if source opened successfully, otherwise False.
        """
        self.release()
        self._frame_id = 0

        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                cap.release()
                return False

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            # Fallback for invalid FPS metadata.
            if fps <= 0:
                fps = 25.0

            self._cap = cap
            self._fps = fps
            return True
        except Exception:
            self.release()
            return False

    def is_opened(self) -> bool:
        """Check if the source is currently opened."""
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, dict[str, Any] | None]:
        """Read one frame and return standardized frame payload.

        Returns:
            (ok, frame_obj)
            - ok=False means no frame available currently.
            - frame_obj format:
                {
                    "frame_id": int,
                    "timestamp": float,
                    "image": np.ndarray,
                    "source_id": str
                }
        """
        if not self.is_opened():
            return False, None

        try:
            assert self._cap is not None
            ok, image = self._cap.read()

            if not ok or image is None:
                # Reached end of file (or temporary read failure).
                if self.loop:
                    if not self._reset_to_start():
                        return False, None

                    ok, image = self._cap.read()
                    if not ok or image is None:
                        return False, None
                else:
                    # Keep stream state consistent for EOF in local file mode.
                    self.release()
                    return False, None

            frame_id = self._frame_id
            timestamp = frame_id / self._fps if self._fps > 0 else 0.0

            frame_obj: dict[str, Any] = {
                "frame_id": frame_id,
                "timestamp": float(timestamp),
                "image": image,
                "source_id": self.source,
            }
            self._frame_id += 1
            return True, frame_obj
        except Exception:
            # If reconnect is enabled, try reopening once.
            if self.reconnect:
                reopened = self.open()
                if reopened:
                    return self.read()
            return False, None

    def release(self) -> None:
        """Release internal OpenCV capture safely."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            finally:
                self._cap = None

    def _reset_to_start(self) -> bool:
        """Reset stream to the beginning (mainly for local files)."""
        if not self.is_opened() or self._cap is None:
            return False

        try:
            success = self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Some backends may return False but still work on next read.
            if not success:
                self.release()
                return self.open()

            self._frame_id = 0
            return True
        except Exception:
            self.release()
            return self.open()
