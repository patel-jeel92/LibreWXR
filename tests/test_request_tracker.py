# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import pytest

pytestmark = pytest.mark.tiles

from librewxr.tiles.request_tracker import TileRequestTracker


class TestTileRequestTracker:
    def test_records_above_min_zoom(self):
        tracker = TileRequestTracker(min_zoom=7)
        tracker.record(7, 1, 2)
        tracker.record(7, 1, 2)
        tracker.record(8, 5, 3)

        stats = tracker.stats()
        assert stats["tracked_tiles"] == 2
        assert stats["total_requests"] == 3

    def test_skips_below_min_zoom(self):
        tracker = TileRequestTracker(min_zoom=7)
        for _ in range(100):
            tracker.record(3, 0, 0)
            tracker.record(6, 1, 1)

        stats = tracker.stats()
        assert stats["tracked_tiles"] == 0
        assert stats["total_requests"] == 0

    def test_top_returns_hottest_tiles(self):
        tracker = TileRequestTracker(min_zoom=7)
        for _ in range(10):
            tracker.record(8, 100, 200)
        for _ in range(3):
            tracker.record(8, 50, 50)
        tracker.record(9, 1, 1)

        stats = tracker.stats(top_n=2)
        assert stats["top"][0] == {"z": 8, "x": 100, "y": 200, "count": 10}
        assert stats["top"][1] == {"z": 8, "x": 50, "y": 50, "count": 3}
        assert len(stats["top"]) == 2

    def test_hot_threshold_count(self):
        tracker = TileRequestTracker(min_zoom=7)
        for _ in range(7):
            tracker.record(8, 1, 1)  # >= 5
        for _ in range(5):
            tracker.record(8, 2, 2)  # >= 5
        for _ in range(2):
            tracker.record(8, 3, 3)  # < 5

        stats = tracker.stats(hot_threshold=5)
        assert stats["hot_tiles"] == 2

    def test_by_zoom_breakdown(self):
        tracker = TileRequestTracker(min_zoom=7)
        tracker.record(7, 1, 1)
        tracker.record(7, 2, 2)
        tracker.record(7, 2, 2)
        tracker.record(9, 5, 5)

        by_zoom = tracker.stats()["by_zoom"]
        assert by_zoom[7] == {"tiles": 2, "requests": 3}
        assert by_zoom[9] == {"tiles": 1, "requests": 1}

    def test_evicts_when_over_cap(self):
        # Cap=4 → eviction triggers on the 5th distinct tile, halving to 2.
        tracker = TileRequestTracker(min_zoom=7, max_entries=4)
        # Hot tiles get many hits — they should survive eviction.
        for _ in range(10):
            tracker.record(8, 0, 0)
        for _ in range(8):
            tracker.record(8, 1, 1)
        # Cold tiles get one hit each.
        tracker.record(8, 2, 2)
        tracker.record(8, 3, 3)
        # This 5th distinct tile triggers the cull down to 2.
        tracker.record(8, 4, 4)

        stats = tracker.stats()
        assert stats["tracked_tiles"] == 2
        kept = {(t["x"], t["y"]) for t in stats["top"]}
        assert kept == {(0, 0), (1, 1)}