"""Tests for EventStore (SQLite persistence)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.service.event_store import EventStore


class TestEventStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp_dir.name) / "events.db")
        self.store = EventStore(db_path=self.db_path)
        self.store.init_db()

        self.sample_event = {
            "event_type": "intrusion",
            "track_id": 42,
            "zone_name": "staff_only_zone",
            "start_time": 10.0,
            "current_time": 12.5,
            "duration": 2.5,
            "confidence_local": 0.8,
        }

    def tearDown(self) -> None:
        self._tmp_dir.cleanup()

    def test_init_db_creates_database(self) -> None:
        self.assertTrue(Path(self.db_path).exists())

    def test_insert_event_generates_event_id_and_get(self) -> None:
        self.store.insert_event(self.sample_event)
        events = self.store.list_events(limit=10)

        self.assertEqual(len(events), 1)
        event_id = events[0]["event_id"]
        self.assertTrue(event_id.startswith("intrusion_42_staff_only_zone_10.000_"))
        self.assertTrue(self.store.event_exists(event_id))

        fetched = self.store.get_event(event_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["event_type"], "intrusion")
        self.assertEqual(fetched["track_id"], 42)
        self.assertEqual(fetched["zone_name"], "staff_only_zone")

    def test_insert_event_deduplicates_same_event(self) -> None:
        self.store.insert_event(self.sample_event)
        self.store.insert_event(dict(self.sample_event))
        events = self.store.list_events(limit=10)
        self.assertEqual(len(events), 1)

    def test_list_events_limit_and_order(self) -> None:
        event_a = dict(self.sample_event)
        event_b = {
            "event_type": "dwell",
            "track_id": 7,
            "zone_name": "checkout_zone",
            "start_time": 20.0,
            "current_time": 31.0,
            "duration": 11.0,
            "confidence_local": 0.92,
        }
        self.store.insert_event(event_a)
        self.store.insert_event(event_b)

        limited = self.store.list_events(limit=1)
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]["event_type"], "dwell")


if __name__ == "__main__":
    unittest.main()
