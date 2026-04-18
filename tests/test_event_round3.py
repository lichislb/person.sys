"""Round-3 event module tests.

Run:
    python -m unittest tests/test_event_round3.py -v
"""

from __future__ import annotations

import unittest

from src.event.candidate_generator import CandidateEventGenerator
from src.event.dwell_rules import DwellRuleEngine
from src.event.roi_rules import IntrusionRuleEngine
from src.event.state_manager import TrackStateManager


def _make_track(track_id: int, center_x: float, center_y: float) -> dict:
    """Create a minimal tracker-like track payload for tests."""
    return {
        "track_id": track_id,
        "bbox": [center_x - 10.0, center_y - 20.0, center_x + 10.0, center_y + 20.0],
        "score": 0.9,
        "class_name": "person",
        "center": [center_x, center_y],
    }


class TestTrackStateManager(unittest.TestCase):
    def test_auto_init_on_update_track(self) -> None:
        sm = TrackStateManager()
        sm.update_track(track_id=1, timestamp=1.0, bbox=[1, 2, 3, 4], center=[2, 3])
        state = sm.get_track_state(1)
        self.assertIsNotNone(state)
        self.assertEqual(state["last_timestamp"], 1.0)
        self.assertEqual(state["last_bbox"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(state["last_center"], [2.0, 3.0])

    def test_zone_states_are_isolated(self) -> None:
        sm = TrackStateManager()
        sm.mark_zone_enter(track_id=1, zone_name="zone_a", timestamp=1.0)
        sm.mark_zone_enter(track_id=1, zone_name="zone_b", timestamp=2.0)
        sm.mark_zone_exit(track_id=1, zone_name="zone_a")

        zone_a = sm.get_zone_state(1, "zone_a")
        zone_b = sm.get_zone_state(1, "zone_b")
        self.assertFalse(zone_a["inside"])
        self.assertIsNone(zone_a["enter_time"])
        self.assertTrue(zone_b["inside"])
        self.assertEqual(zone_b["enter_time"], 2.0)

    def test_event_flags_intrusion_and_dwell(self) -> None:
        sm = TrackStateManager()
        sm.mark_zone_enter(track_id=7, zone_name="checkout", timestamp=1.0)
        self.assertFalse(sm.is_event_triggered(7, "checkout", "intrusion"))
        self.assertFalse(sm.is_event_triggered(7, "checkout", "dwell"))

        sm.set_event_triggered(7, "checkout", "intrusion")
        self.assertTrue(sm.is_event_triggered(7, "checkout", "intrusion"))
        self.assertFalse(sm.is_event_triggered(7, "checkout", "dwell"))

        sm.set_event_triggered(7, "checkout", "dwell")
        self.assertTrue(sm.is_event_triggered(7, "checkout", "dwell"))

    def test_cleanup_stale_tracks_keeps_active(self) -> None:
        sm = TrackStateManager()
        sm.update_track(track_id=1, timestamp=10.0, bbox=[0, 0, 1, 1], center=[0, 0])  # active
        sm.update_track(track_id=2, timestamp=1.0, bbox=[0, 0, 1, 1], center=[0, 0])   # stale

        sm.cleanup_stale_tracks(current_time=10.5, max_missing_sec=2.0)
        self.assertIsNotNone(sm.get_track_state(1))
        self.assertIsNone(sm.get_track_state(2))


class TestDwellRuleEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.zones = {
            "staff_only_zone": {
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "type": "restricted",
            },
            "checkout_zone": {
                "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]],
                "type": "service",
            },
        }
        self.sm = TrackStateManager()

    def test_only_service_or_dwell_type(self) -> None:
        engine = DwellRuleEngine(zones=self.zones, min_duration_sec=0.0)
        # inside restricted zone only -> should not produce dwell
        track = _make_track(track_id=1, center_x=50.0, center_y=50.0)
        events = engine.check_track(track, timestamp=1.0, state_manager=self.sm)
        self.assertEqual(events, [])

    def test_enter_and_trigger_once_then_no_repeat(self) -> None:
        engine = DwellRuleEngine(zones=self.zones, min_duration_sec=1.0)
        track = _make_track(track_id=2, center_x=250.0, center_y=50.0)  # inside service zone

        events_t1 = engine.check_track(track, timestamp=1.0, state_manager=self.sm)
        self.assertEqual(events_t1, [])
        zone_state = self.sm.get_zone_state(2, "checkout_zone")
        self.assertIsNotNone(zone_state)
        self.assertEqual(zone_state["enter_time"], 1.0)

        events_t2 = engine.check_track(track, timestamp=2.2, state_manager=self.sm)
        self.assertEqual(len(events_t2), 1)
        self.assertEqual(events_t2[0]["event_type"], "dwell")

        events_t3 = engine.check_track(track, timestamp=3.0, state_manager=self.sm)
        self.assertEqual(events_t3, [])

    def test_exit_resets_inside_state(self) -> None:
        engine = DwellRuleEngine(zones=self.zones, min_duration_sec=0.0)
        inside_track = _make_track(track_id=3, center_x=250.0, center_y=50.0)
        outside_track = _make_track(track_id=3, center_x=350.0, center_y=50.0)

        engine.check_track(inside_track, timestamp=1.0, state_manager=self.sm)
        zone_state_inside = self.sm.get_zone_state(3, "checkout_zone")
        self.assertTrue(zone_state_inside["inside"])

        engine.check_track(outside_track, timestamp=2.0, state_manager=self.sm)
        zone_state_out = self.sm.get_zone_state(3, "checkout_zone")
        self.assertFalse(zone_state_out["inside"])
        self.assertIsNone(zone_state_out["enter_time"])


class TestCandidateEventGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.zones = {
            "staff_only_zone": {
                "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "type": "restricted",
            },
            "checkout_zone": {
                "polygon": [[200, 0], [300, 0], [300, 100], [200, 100]],
                "type": "service",
            },
        }

    def test_scheduler_handles_none_engines_and_empty_tracks(self) -> None:
        sm = TrackStateManager()
        gen = CandidateEventGenerator(
            intrusion_engine=None,
            dwell_engine=None,
            state_manager=sm,
        )
        self.assertEqual(gen.generate(tracks=[], timestamp=1.0), [])
        self.assertEqual(gen.generate(tracks=None, timestamp=1.0), [])

    def test_multi_tracks_aggregate_events(self) -> None:
        sm = TrackStateManager()
        intrusion_engine = IntrusionRuleEngine(zones=self.zones, min_duration_sec=0.0)
        dwell_engine = DwellRuleEngine(zones=self.zones, min_duration_sec=0.0)
        gen = CandidateEventGenerator(
            intrusion_engine=intrusion_engine,
            dwell_engine=dwell_engine,
            state_manager=sm,
        )

        tracks = [
            _make_track(track_id=10, center_x=50.0, center_y=50.0),   # restricted -> intrusion
            _make_track(track_id=11, center_x=250.0, center_y=50.0),  # service -> dwell
        ]
        events = gen.generate(tracks=tracks, timestamp=1.0)
        event_types = sorted([e["event_type"] for e in events])
        self.assertEqual(event_types, ["dwell", "intrusion"])

        # same frame context later should not duplicate same track-zone-event pair
        events_again = gen.generate(tracks=tracks, timestamp=2.0)
        self.assertEqual(events_again, [])


if __name__ == "__main__":
    unittest.main()
