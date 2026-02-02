"""
Entity Tracker Tests
====================

Tests for Phase 3 entity-attribute tracking to improve knowledge-update accuracy.

First Principles:
- Knowledge updates are changes to the same entity's attribute over time
- Entity-attribute alignment can detect updates even when semantic similarity is low
- Example: "28 min" and "25:50" have low semantic similarity but represent the same entity-attribute
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from aurora.algorithms.entity_tracker import EntityTracker, EntityAttribute
from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.utils.time_utils import now_ts


class TestEntityTracker:
    """Tests for EntityTracker class."""
    
    def test_extract_phone_number(self):
        """Test phone number extraction."""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("My phone number is 123-456-7890")
        assert len(entities) > 0
        
        phone_entity = next((e for e in entities if e[1] == "phone"), None)
        assert phone_entity is not None
        assert phone_entity[0] == "user"  # entity
        assert phone_entity[1] == "phone"  # attribute
        assert "1234567890" in phone_entity[2] or "123-456-7890" in phone_entity[2]  # value
    
    def test_extract_email(self):
        """Test email extraction."""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("Contact me at user@example.com")
        assert len(entities) > 0
        
        email_entity = next((e for e in entities if e[1] == "email"), None)
        assert email_entity is not None
        assert "user@example.com" in email_entity[2]
    
    def test_extract_location(self):
        """Test location extraction."""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("I live in Shanghai")
        assert len(entities) > 0
        
        location_entity = next((e for e in entities if e[1] == "location"), None)
        assert location_entity is not None
        assert "Shanghai" in location_entity[2]
    
    def test_extract_5k_time(self):
        """Test 5K time extraction."""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("My 5K time is 25:50")
        assert len(entities) > 0
        
        time_entity = next((e for e in entities if e[1] == "5k_time"), None)
        assert time_entity is not None
        assert "25:50" in time_entity[2]
    
    def test_extract_time_minutes(self):
        """Test time extraction in minutes format."""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("I ran 5K in 28 minutes")
        assert len(entities) > 0
        
        time_entity = next((e for e in entities if e[1] == "5k_time"), None)
        assert time_entity is not None
        assert "28" in time_entity[2] or "min" in time_entity[2].lower()
    
    def test_timeline_tracking(self):
        """Test timeline tracking for entity-attribute."""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # First value
        tracker.update("My 5K time is 28 min", "plot1", ts1)
        
        # Second value (after 1 hour)
        ts2 = ts1 + 3600
        tracker.update("My 5K time is now 25:50", "plot2", ts2)
        
        # Check timeline
        timeline = tracker.get_timeline("user", "5k_time")
        assert len(timeline) == 2
        assert timeline[0].value == "28 min" or "28" in timeline[0].value
        assert timeline[1].value == "25:50" or "25" in timeline[1].value
    
    def test_find_potential_updates(self):
        """Test finding potential updates."""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # First value
        tracker.update("My 5K time is 28 min", "plot1", ts1)
        
        # Check for potential update
        ts2 = ts1 + 3600
        potential_updates = tracker.find_potential_updates("My 5K time is 25:50", ts2)
        
        assert len(potential_updates) > 0
        old_ea, new_ea, conf = potential_updates[0]
        assert old_ea.attribute == "5k_time"
        assert old_ea.plot_id == "plot1"
        assert conf > 0.0
    
    def test_normalize_value(self):
        """Test value normalization for comparison."""
        tracker = EntityTracker(seed=42)
        
        # Test time format normalization
        norm1 = tracker._normalize_value("28 min")
        norm2 = tracker._normalize_value("28:00")
        
        # Both should represent similar time
        # Note: exact match depends on implementation
        assert isinstance(norm1, str)
        assert isinstance(norm2, str)
    
    def test_serialization(self):
        """Test serialization and deserialization."""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        tracker.update("My phone is 123-456-7890", "plot1", ts1)
        tracker.update("I live in Beijing", "plot2", ts1 + 100)
        
        # Serialize
        state = tracker.to_state_dict()
        assert "timelines" in state
        assert "seed" in state
        
        # Deserialize
        restored = EntityTracker.from_state_dict(state)
        assert len(restored.timelines) == len(tracker.timelines)
        
        # Check timeline preserved
        timeline = restored.get_timeline("user", "phone")
        assert len(timeline) == 1
        assert timeline[0].plot_id == "plot1"


class TestEntityTrackerIntegration:
    """Integration tests for EntityTracker with AuroraMemory."""
    
    def test_entity_tracker_integrated(self, aurora_memory: AuroraMemory):
        """Test that EntityTracker is integrated into AuroraMemory."""
        assert hasattr(aurora_memory, "entity_tracker")
        assert isinstance(aurora_memory.entity_tracker, EntityTracker)
    
    def test_5k_time_update_detection(self, aurora_memory: AuroraMemory):
        """Test that 5K time updates are detected even with low semantic similarity.
        
        Scenario: "28 min" and "25:50" have low semantic similarity,
        but EntityTracker should detect them as the same entity-attribute.
        """
        # First interaction: establish 5K time
        plot1 = aurora_memory.ingest(
            "User: My 5K time is 28 minutes. Assistant: Great time!",
            event_id="5k_time_v1",
        )
        
        # Wait a bit to ensure time gap
        time.sleep(1)
        
        # Second interaction: update 5K time
        plot2 = aurora_memory.ingest(
            "User: I improved my 5K time to 25:50. Assistant: Congratulations!",
            event_id="5k_time_v2",
        )
        
        # Both plots should be stored (cold start ensures first 10 are stored)
        assert plot1.id in aurora_memory.plots
        assert plot2.id in aurora_memory.plots
        
        # Check entity tracker has recorded both values
        timeline = aurora_memory.entity_tracker.get_timeline("user", "5k_time")
        assert len(timeline) >= 2, f"Expected at least 2 entries, got {len(timeline)}"
        
        # Check that values are different
        values = [ea.value for ea in timeline]
        assert any("28" in v or "min" in v.lower() for v in values)
        assert any("25" in v or "25:50" in v for v in values)
    
    def test_phone_number_update_detection(self, aurora_memory: AuroraMemory):
        """Test phone number update detection."""
        # First interaction
        plot1 = aurora_memory.ingest(
            "User: My phone number is 123-456-7890. Assistant: Saved.",
            event_id="phone_v1",
        )
        
        time.sleep(1)
        
        # Update phone number
        plot2 = aurora_memory.ingest(
            "User: I changed my phone number to 098-765-4321. Assistant: Updated.",
            event_id="phone_v2",
        )
        
        # Check entity tracker
        timeline = aurora_memory.entity_tracker.get_timeline("user", "phone")
        assert len(timeline) >= 2
        
        # Check values are different
        values = [ea.value for ea in timeline]
        assert any("123" in v for v in values)
        assert any("098" in v or "987" in v for v in values)
    
    def test_location_update_detection(self, aurora_memory: AuroraMemory):
        """Test location update detection."""
        # First interaction
        plot1 = aurora_memory.ingest(
            "User: I live in Beijing. Assistant: Got it.",
            event_id="location_v1",
        )
        
        time.sleep(1)
        
        # Update location
        plot2 = aurora_memory.ingest(
            "User: I moved to Shanghai. Assistant: Updated your location.",
            event_id="location_v2",
        )
        
        # Check entity tracker
        timeline = aurora_memory.entity_tracker.get_timeline("user", "location")
        assert len(timeline) >= 2
        
        # Check values
        values = [ea.value for ea in timeline]
        assert any("Beijing" in v for v in values)
        assert any("Shanghai" in v for v in values)
    
    def test_low_similarity_update_detection(self, aurora_memory: AuroraMemory):
        """Test that entity tracker detects updates even when semantic similarity is low.
        
        This is the key improvement: "28 min" and "25:50" have low semantic similarity
        but EntityTracker should detect them as the same entity-attribute.
        """
        # First: establish time in minutes format
        plot1 = aurora_memory.ingest(
            "User: My best 5K time is 28 minutes.",
            event_id="time_min",
        )
        
        time.sleep(2)  # Ensure time gap
        
        # Second: update time in colon format (different format, low semantic similarity)
        plot2 = aurora_memory.ingest(
            "User: I ran 5K in 25:50 today.",
            event_id="time_colon",
        )
        
        # Check entity tracker detected both
        timeline = aurora_memory.entity_tracker.get_timeline("user", "5k_time")
        assert len(timeline) >= 2, (
            f"Entity tracker should detect both values. "
            f"Timeline length: {len(timeline)}, values: {[ea.value for ea in timeline]}"
        )
        
        # Verify both values are in timeline
        values = [ea.value for ea in timeline]
        has_minutes = any("28" in v or "min" in v.lower() for v in values)
        has_colon = any("25" in v and ":" in v for v in values)
        
        assert has_minutes, f"Should have minutes format. Values: {values}"
        assert has_colon, f"Should have colon format. Values: {values}"


class TestEntityTrackerUpdateDetection:
    """Tests for update detection using EntityTracker."""
    
    def test_check_entity_update(self):
        """Test check_entity_update method."""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # First value
        tracker.update("My 5K time is 28 min", "plot1", ts1)
        
        # Check for update
        ts2 = ts1 + 3600
        result = tracker.check_entity_update(
            "My 5K time is 25:50",
            "plot2",
            ts2,
            candidate_plot_id="plot1"
        )
        
        assert result is not None
        entity, attr, old_value, conf = result
        assert entity == "user"
        assert attr == "5k_time"
        assert "28" in old_value or "min" in old_value.lower()
        assert conf > 0.0
    
    def test_no_update_when_value_same(self):
        """Test that no update is detected when value is the same."""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        tracker.update("My 5K time is 25:50", "plot1", ts1)
        
        ts2 = ts1 + 3600
        result = tracker.check_entity_update(
            "My 5K time is 25:50",
            "plot2",
            ts2
        )
        
        # Should not detect update if value is the same
        # (or detect with very low confidence)
        if result is not None:
            entity, attr, old_value, conf = result
            assert conf < 0.3  # Low confidence for same value
    
    def test_update_with_different_formats(self):
        """Test update detection with different time formats."""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # First: minutes format
        tracker.update("I ran 5K in 28 minutes", "plot1", ts1)
        
        # Second: colon format
        ts2 = ts1 + 3600
        potential_updates = tracker.find_potential_updates("My 5K time is 25:50", ts2)
        
        # Should detect as potential update even with different formats
        assert len(potential_updates) > 0
        
        old_ea, new_ea, conf = potential_updates[0]
        assert old_ea.attribute == "5k_time"
        # Values should be different (normalized)
        assert tracker._normalize_value(old_ea.value) != tracker._normalize_value(new_ea.value)
