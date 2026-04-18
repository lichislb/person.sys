"""Candidate event generator by scheduling multiple rule engines."""

from __future__ import annotations

from typing import Any


class CandidateEventGenerator:
    """Orchestrate rule engines and aggregate candidate events."""

    def __init__(
        self,
        intrusion_engine: Any,
        dwell_engine: Any,
        state_manager: Any,
        max_missing_sec: float = 2.0,
    ) -> None:
        self.intrusion_engine = intrusion_engine
        self.dwell_engine = dwell_engine
        self.state_manager = state_manager
        self.max_missing_sec = float(max(0.0, max_missing_sec))

    def generate(self, tracks: list[dict[str, Any]], timestamp: float) -> list[dict[str, Any]]:
        """Generate candidate events for current frame."""
        events: list[dict[str, Any]] = []
        if tracks is None:
            tracks = []

        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            return events

        for track in tracks:
            if not isinstance(track, dict):
                continue

            track_id = track.get("track_id")
            bbox = track.get("bbox")
            center = track.get("center")

            # Update generic track state first.
            self.state_manager.update_track(track_id, ts, bbox, center)

            if self.intrusion_engine is not None:
                intrusion_events = self.intrusion_engine.check_track(
                    track=track,
                    timestamp=ts,
                    state_manager=self.state_manager,
                )
                if intrusion_events:
                    events.extend(intrusion_events)

            if self.dwell_engine is not None:
                dwell_events = self.dwell_engine.check_track(
                    track=track,
                    timestamp=ts,
                    state_manager=self.state_manager,
                )
                if dwell_events:
                    events.extend(dwell_events)

        return events

    def cleanup(self, current_time: float) -> None:
        """Cleanup stale track states."""
        self.state_manager.cleanup_stale_tracks(
            current_time=current_time, max_missing_sec=self.max_missing_sec
        )
