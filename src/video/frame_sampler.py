"""Frame sampling module.

This module controls frame processing frequency to reduce downstream load.
"""

from __future__ import annotations


class FrameSampler:
    """Decide whether a frame should be processed based on frame index."""

    def __init__(self, frame_skip: int = 1) -> None:
        """Initialize frame sampler.

        Args:
            frame_skip: Process one frame every `frame_skip` frames.
                - 1: process every frame
                - 2: process every 2nd frame (frame_id 0,2,4,...)
                - <=0: fallback to 1
        """
        self.frame_skip: int = frame_skip if frame_skip > 0 else 1

    def should_process(self, frame_id: int) -> bool:
        """Return whether the input frame should be processed.

        Args:
            frame_id: Non-negative frame index.

        Returns:
            True if this frame should be processed, else False.
        """
        if frame_id < 0:
            return False
        return frame_id % self.frame_skip == 0
