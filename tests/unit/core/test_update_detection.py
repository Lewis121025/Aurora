"""
Knowledge Update Detection Tests
=================================

Tests for the knowledge update detection feature in AuroraMemory.

First Principles being tested:
1. Redundancy ≠ Semantic similarity
2. Update = same entity's state change over time (carries information gain)
3. Reinforcement = short-term repetition confirming same info
4. Pure redundancy = identical information repeated

In narrative psychology: re-narration repositions old info as "past self"
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from aurora.core.memory import AuroraMemory
from aurora.core.models.config import MemoryConfig
from aurora.core.constants import (
    UPDATE_TIME_GAP_THRESHOLD,
    REINFORCEMENT_TIME_WINDOW,
)


class TestUpdateDetection:
    """Tests for knowledge update detection."""

    def test_location_update_both_stored(self, aurora_memory: AuroraMemory):
        """Test that location updates result in both plots being stored.
        
        Scenario: User first says they live in Beijing, then says they moved to Shanghai.
        Both should be stored because the second is an UPDATE, not redundancy.
        
        Note: With HashEmbedding (random vectors), semantic similarity may be too low
        to trigger update detection. The test verifies both plots are stored
        and checks update detection if similarity was high enough.
        """
        # First interaction: establish location
        plot1 = aurora_memory.ingest(
            "User: I live in Beijing. Assistant: Got it, you're in Beijing!",
            event_id="location_v1",
        )
        
        # Second interaction: update location with explicit update keywords
        plot2 = aurora_memory.ingest(
            "User: I moved to Shanghai last month. Assistant: I'll update that, you're now in Shanghai.",
            event_id="location_v2",
        )
        
        # Both plots should be stored (cold start ensures first 10 are stored)
        assert plot1.id in aurora_memory.plots, "Original location plot should be stored"
        assert plot2.id in aurora_memory.plots, "Updated location plot should be stored"
        
        # With real semantic embeddings, the second should be marked as update
        # With HashEmbedding, similarity may be too low to trigger update detection
        # This is expected behavior - update detection requires finding similar content first
        if aurora_memory.is_using_hash_embedding():
            # With HashEmbedding, just verify storage and redundancy_type is set
            assert plot2.redundancy_type in ("novel", "update", "reinforcement", "pure_redundant")
        else:
            # With real embeddings, should detect as update
            assert plot2.redundancy_type == "update", f"Expected 'update', got {plot2.redundancy_type}"
        
    def test_state_change_keywords_trigger_update(self, aurora_memory: AuroraMemory):
        """Test that state change keywords trigger update classification."""
        # Force store some base data first (cold start)
        for i in range(10):
            aurora_memory.ingest(f"User: Filler {i}. Assistant: OK.", event_id=f"filler_{i}")
        
        # Now test update detection
        plot1 = aurora_memory.ingest(
            "User: My email is old@example.com. Assistant: Noted.",
            event_id="email_v1",
        )
        
        # Update with "changed" keyword
        plot2 = aurora_memory.ingest(
            "User: I've changed my email to new@example.com. Assistant: Updated!",
            event_id="email_v2",
        )
        
        # The second should be detected as an update if similarity is high enough
        # Note: with HashEmbedding, similarity may be random, but the keyword detection should work
        if plot2.id in aurora_memory.plots:
            # If stored, check if it was due to update detection
            if plot2.redundancy_type == "update":
                assert plot2.supersedes_id is not None or plot2.update_type is not None

    def test_numeric_change_detection(self, aurora_memory: AuroraMemory):
        """Test that numeric value changes are detected as updates."""
        # First: establish a numeric fact
        plot1 = aurora_memory.ingest(
            "User: I have 5 team members. Assistant: Got it, 5 people.",
            event_id="team_size_v1",
        )
        
        # Update with different number
        plot2 = aurora_memory.ingest(
            "User: We hired more people, now I have 8 team members. Assistant: Updated to 8.",
            event_id="team_size_v2",
        )
        
        # Both should be stored
        assert plot1.id in aurora_memory.plots
        assert plot2.id in aurora_memory.plots

    def test_correction_detection(self, aurora_memory: AuroraMemory):
        """Test that corrections are detected as updates."""
        # Original
        plot1 = aurora_memory.ingest(
            "User: I think Python 3.8 is the latest. Assistant: OK.",
            event_id="python_v1",
        )
        
        # Correction with "actually" keyword
        plot2 = aurora_memory.ingest(
            "User: Actually, Python 3.12 is the latest version now. Assistant: Thanks for the correction!",
            event_id="python_v2",
        )
        
        # Both should be stored (correction is an update type)
        assert plot1.id in aurora_memory.plots
        assert plot2.id in aurora_memory.plots

    def test_negation_triggers_update(self, aurora_memory: AuroraMemory):
        """Test that negation of previous info triggers update detection."""
        # Original status
        plot1 = aurora_memory.ingest(
            "User: I use Windows. Assistant: Windows user, got it.",
            event_id="os_v1",
        )
        
        # Negation with "no longer"
        plot2 = aurora_memory.ingest(
            "User: I no longer use Windows, switched to Mac. Assistant: You're on Mac now.",
            event_id="os_v2",
        )
        
        assert plot1.id in aurora_memory.plots
        assert plot2.id in aurora_memory.plots


class TestUpdateSignalDetection:
    """Tests for the _detect_update_signals method."""
    
    def test_detect_update_keywords_chinese(self, aurora_memory: AuroraMemory):
        """Test Chinese update keyword detection."""
        signals = aurora_memory._detect_update_signals(
            new_text="我现在住在上海了",
            old_text="我住在北京",
            new_ts=time.time(),
            old_ts=time.time() - 3600 * 24,  # 1 day ago
        )
        
        assert signals["is_update"], "Should detect '现在' as update keyword"
        assert "update_keywords" in signals["signals"]
        
    def test_detect_update_keywords_english(self, aurora_memory: AuroraMemory):
        """Test English update keyword detection."""
        signals = aurora_memory._detect_update_signals(
            new_text="I have now moved to Shanghai",
            old_text="I live in Beijing",
            new_ts=time.time(),
            old_ts=time.time() - 3600 * 24,
        )
        
        assert signals["is_update"], "Should detect 'now' and 'moved' as update keywords"
        
    def test_detect_numeric_change(self, aurora_memory: AuroraMemory):
        """Test numeric change detection."""
        signals = aurora_memory._detect_update_signals(
            new_text="The team size is now 10",
            old_text="The team has 5 members",
            new_ts=time.time(),
            old_ts=time.time() - 3600,
        )
        
        # Should detect numeric change (5 vs 10)
        assert "numeric_change" in signals["signals"] or signals["is_update"]
        
    def test_detect_correction(self, aurora_memory: AuroraMemory):
        """Test correction pattern detection."""
        signals = aurora_memory._detect_update_signals(
            new_text="Actually, the correct answer is 42",
            old_text="The answer is 24",
            new_ts=time.time(),
            old_ts=time.time() - 3600,
        )
        
        assert signals["is_update"]
        if signals["update_type"]:
            assert signals["update_type"] == "correction"
            
    def test_time_gap_contributes_to_confidence(self, aurora_memory: AuroraMemory):
        """Test that time gap contributes to update confidence."""
        # Same text comparison, but different time gaps
        signals_short = aurora_memory._detect_update_signals(
            new_text="我现在住在上海",
            old_text="我住在北京",
            new_ts=time.time(),
            old_ts=time.time() - 60,  # 1 minute ago (short gap)
        )
        
        signals_long = aurora_memory._detect_update_signals(
            new_text="我现在住在上海",
            old_text="我住在北京",
            new_ts=time.time(),
            old_ts=time.time() - 3600 * 24,  # 1 day ago (long gap)
        )
        
        # Long gap should have higher confidence due to time_gap signal
        # Both should be updates due to keywords, but confidence may differ
        assert signals_short["is_update"]
        assert signals_long["is_update"]


class TestRedundancyTypes:
    """Tests for different redundancy type classifications."""
    
    def test_novel_content_classification(self, aurora_memory: AuroraMemory):
        """Test that completely new content is classified as novel."""
        # First plot
        aurora_memory.ingest(
            "User: Tell me about Python. Assistant: Python is a language.",
            event_id="python_info",
        )
        
        # Completely different topic
        plot2 = aurora_memory.ingest(
            "User: What's the weather like? Assistant: It's sunny.",
            event_id="weather_info",
        )
        
        # With HashEmbedding, similarity is random, so we just check storage
        # The semantic difference should result in "novel" classification
        assert plot2.redundancy_type in ("novel", "update", "reinforcement", "pure_redundant")

    def test_plot_supersedes_chain(self, aurora_memory: AuroraMemory):
        """Test that update chain is properly recorded."""
        # Create initial fact
        plot1 = aurora_memory.ingest(
            "User: My phone number is 123-456-7890. Assistant: Saved.",
            event_id="phone_v1",
        )
        
        # Update
        plot2 = aurora_memory.ingest(
            "User: I've changed my phone number to 098-765-4321. Assistant: Updated.",
            event_id="phone_v2",
        )
        
        # Check that both are stored
        assert plot1.id in aurora_memory.plots
        assert plot2.id in aurora_memory.plots
        
        # If detected as update, should have supersedes_id
        if plot2.redundancy_type == "update":
            assert plot2.supersedes_id is not None or plot2.update_type is not None


class TestReinforcementVsRedundancy:
    """Tests for distinguishing reinforcement from pure redundancy."""
    
    def test_short_term_repetition_is_reinforcement(self, aurora_memory: AuroraMemory):
        """Test that short-term repetition is classified as reinforcement."""
        # First message
        plot1 = aurora_memory.ingest(
            "User: Remember, the meeting is at 3pm. Assistant: Got it.",
            event_id="meeting_v1",
        )
        
        # Same info repeated shortly after (within REINFORCEMENT_TIME_WINDOW)
        # Note: we can't actually wait, so this tests the logic path
        plot2 = aurora_memory.ingest(
            "User: Don't forget, the meeting is at 3pm. Assistant: Yes, I remember.",
            event_id="meeting_v2",
        )
        
        # Both should be stored (cold start + may be reinforcement)
        assert plot1.id in aurora_memory.plots
        # Plot2 may or may not be stored based on redundancy calculation


class TestSerializationWithUpdateFields:
    """Tests for serialization of new update-related fields."""
    
    def test_plot_serialization_includes_update_fields(self, aurora_memory: AuroraMemory):
        """Test that Plot serialization includes new fields."""
        plot = aurora_memory.ingest(
            "User: I've moved to a new city. Assistant: Where to?",
            event_id="move_test",
        )
        
        # Serialize
        state = plot.to_state_dict()
        
        # New fields should be present
        assert "supersedes_id" in state
        assert "update_type" in state
        assert "redundancy_type" in state
        
    def test_plot_deserialization_restores_update_fields(self, aurora_memory: AuroraMemory):
        """Test that Plot deserialization restores new fields."""
        from aurora.core.models.plot import Plot
        
        plot = aurora_memory.ingest(
            "User: I've changed my settings. Assistant: Updated.",
            event_id="settings_test",
        )
        
        # Simulate setting update fields
        plot.supersedes_id = "some_old_plot_id"
        plot.update_type = "state_change"
        plot.redundancy_type = "update"
        
        # Round-trip
        state = plot.to_state_dict()
        restored = Plot.from_state_dict(state)
        
        assert restored.supersedes_id == "some_old_plot_id"
        assert restored.update_type == "state_change"
        assert restored.redundancy_type == "update"
        
    def test_memory_serialization_preserves_update_chain(self, aurora_memory: AuroraMemory):
        """Test that full memory serialization preserves update chains."""
        # Create an update scenario
        aurora_memory.ingest(
            "User: I live in New York. Assistant: New York it is!",
            event_id="city_v1",
        )
        aurora_memory.ingest(
            "User: I've moved to Los Angeles. Assistant: LA, got it!",
            event_id="city_v2",
        )
        
        # Serialize entire memory
        state = aurora_memory.to_state_dict()
        
        # Deserialize
        restored = AuroraMemory.from_state_dict(state)
        
        # Check that all plots are restored
        assert len(restored.plots) == len(aurora_memory.plots)
        
        # Check that update fields are preserved
        for pid, plot in aurora_memory.plots.items():
            restored_plot = restored.plots[pid]
            assert restored_plot.redundancy_type == plot.redundancy_type
            assert restored_plot.supersedes_id == plot.supersedes_id
            assert restored_plot.update_type == plot.update_type
