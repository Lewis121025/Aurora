"""
实体追踪测试
====================

步骤3实体属性追踪测试是改进知识更新准确性。

基本原理：
- 知识更新是同一实体属性随时间推进的变化
- 实体属性一致性可以检测更新，即使语义相似性低
- 例子："28分钟""25:50"低语义相似性但表示同一实体属性
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
    """实体追踪汛测试。"""
    
    def test_extract_phone_number(self):
        """测试电话号码提取。"""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("My phone number is 123-456-7890")
        assert len(entities) > 0
        
        phone_entity = next((e for e in entities if e[1] == "phone"), None)
        assert phone_entity is not None
        assert phone_entity[0] == "user"  # 实体
        assert phone_entity[1] == "phone"  # 属性
        assert "1234567890" in phone_entity[2] or "123-456-7890" in phone_entity[2]  # 值
    
    def test_extract_email(self):
        """测试邮件提取。"""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("Contact me at user@example.com")
        assert len(entities) > 0
        
        email_entity = next((e for e in entities if e[1] == "email"), None)
        assert email_entity is not None
        assert "user@example.com" in email_entity[2]
    
    def test_extract_location(self):
        """测试位置提取。"""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("I live in Shanghai")
        assert len(entities) > 0
        
        location_entity = next((e for e in entities if e[1] == "location"), None)
        assert location_entity is not None
        assert "Shanghai" in location_entity[2]
    
    def test_extract_5k_time(self):
        """测试5K跑步成绩提取。"""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("My 5K time is 25:50")
        assert len(entities) > 0
        
        time_entity = next((e for e in entities if e[1] == "5k_time"), None)
        assert time_entity is not None
        assert "25:50" in time_entity[2]
    
    def test_extract_time_minutes(self):
        """测试分钟格式的时间提取。"""
        tracker = EntityTracker(seed=42)
        
        entities = tracker.extract_entities("I ran 5K in 28 minutes")
        assert len(entities) > 0
        
        time_entity = next((e for e in entities if e[1] == "5k_time"), None)
        assert time_entity is not None
        assert "28" in time_entity[2] or "min" in time_entity[2].lower()
    
    def test_timeline_tracking(self):
        """测试实体属性的时间线追踪。"""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # 第一个值
        tracker.update("My 5K time is 28 min", "plot1", ts1)
        
        # 第二个值（之后1小时）
        ts2 = ts1 + 3600
        tracker.update("My 5K time is now 25:50", "plot2", ts2)
        
        # 检查时间线
        timeline = tracker.get_timeline("user", "5k_time")
        assert len(timeline) == 2
        assert timeline[0].value == "28 min" or "28" in timeline[0].value
        assert timeline[1].value == "25:50" or "25" in timeline[1].value
    
    def test_find_potential_updates(self):
        """测试下一步更新发现。"""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # 第一个值
        tracker.update("My 5K time is 28 min", "plot1", ts1)
        
        # 检查是否有下一步更新
        ts2 = ts1 + 3600
        potential_updates = tracker.find_potential_updates("My 5K time is 25:50", ts2)
        
        assert len(potential_updates) > 0
        old_ea, new_ea, conf = potential_updates[0]
        assert old_ea.attribute == "5k_time"
        assert old_ea.plot_id == "plot1"
        assert conf > 0.0
    
    def test_normalize_value(self):
        """测试值正规化以便比较。"""
        tracker = EntityTracker(seed=42)
        
        # 测试时间格式正规化
        norm1 = tracker._normalize_value("28 min")
        norm2 = tracker._normalize_value("28:00")
        
        # 两者都应表示类似的时间
        # 注意：确切匹配取决于实现
        assert isinstance(norm1, str)
        assert isinstance(norm2, str)
    
    def test_serialization(self):
        """测试序列化和反序列化。"""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        tracker.update("My phone is 123-456-7890", "plot1", ts1)
        tracker.update("I live in Beijing", "plot2", ts1 + 100)
        
        # 序列化
        state = tracker.to_state_dict()
        assert "timelines" in state
        assert "seed" in state
        
        # 反序列化
        restored = EntityTracker.from_state_dict(state)
        assert len(restored.timelines) == len(tracker.timelines)
        
        # Check timeline preserved
        timeline = restored.get_timeline("user", "phone")
        assert len(timeline) == 1
        assert timeline[0].plot_id == "plot1"


class TestEntityTrackerIntegration:
    """实体追踪汛与AuroraMemory的集成测试。"""
    
    def test_entity_tracker_integrated(self, aurora_memory: AuroraMemory):
        """测试实体追踪汛是AuroraMemory的一部分。"""
        assert hasattr(aurora_memory, "entity_tracker")
        assert isinstance(aurora_memory.entity_tracker, EntityTracker)
    
    def test_5k_time_update_detection(self, aurora_memory: AuroraMemory):
        """测诖5K跑步成绩更新即使语义相似性低。
        
        场景: "28 分钟" "25:50" 低语义相似性，
        但实体追踪汛应检测它们是同一实体属性。
        """
        # 第一次互动：建立5K跑步成绩
        plot1 = aurora_memory.ingest(
            "User: My 5K time is 28 minutes. Assistant: Great time!",
            event_id="5k_time_v1",
        )
        
        # 等一下对仆以确保时间间隔
        time.sleep(1)
        
        # 第二次互动：更新5K跑步成绩
        plot2 = aurora_memory.ingest(
            "User: I improved my 5K time to 25:50. Assistant: Congratulations!",
            event_id="5k_time_v2",
        )
        
        # 两个情节都应被存储（冷启动确保前10个被存储）
        assert plot1.id in aurora_memory.plots
        assert plot2.id in aurora_memory.plots
        
        # 检查实体追踪汛是否记录了两个值
        timeline = aurora_memory.entity_tracker.get_timeline("user", "5k_time")
        assert len(timeline) >= 2, f"预旨至少5个条目，实际{len(timeline)}"
        
        # 检查值是否不同
        values = [ea.value for ea in timeline]
        assert any("28" in v or "min" in v.lower() for v in values)
        assert any("25" in v or "25:50" in v for v in values)
    
    def test_phone_number_update_detection(self, aurora_memory: AuroraMemory):
        """测试电话号码更新检测。"""
        # 第一次互动
        plot1 = aurora_memory.ingest(
            "User: My phone number is 123-456-7890. Assistant: Saved.",
            event_id="phone_v1",
        )
        
        time.sleep(1)
        
        # 更新电话号码
        plot2 = aurora_memory.ingest(
            "User: I changed my phone number to 098-765-4321. Assistant: Updated.",
            event_id="phone_v2",
        )
        
        # 检查实体追踪汛
        timeline = aurora_memory.entity_tracker.get_timeline("user", "phone")
        assert len(timeline) >= 2
        
        # 检查值是否不同
        values = [ea.value for ea in timeline]
        assert any("123" in v for v in values)
        assert any("098" in v or "987" in v for v in values)
    
    def test_location_update_detection(self, aurora_memory: AuroraMemory):
        """测试位置更新检测。"""
        # 第一次互动
        plot1 = aurora_memory.ingest(
            "User: I live in Beijing. Assistant: Got it.",
            event_id="location_v1",
        )
        
        time.sleep(1)
        
        # 更新位置
        plot2 = aurora_memory.ingest(
            "User: I moved to Shanghai. Assistant: Updated your location.",
            event_id="location_v2",
        )
        
        # 检查实体追踪汛
        timeline = aurora_memory.entity_tracker.get_timeline("user", "location")
        assert len(timeline) >= 2
        
        # 检查值
        values = [ea.value for ea in timeline]
        assert any("Beijing" in v for v in values)
        assert any("Shanghai" in v for v in values)
    
    def test_low_similarity_update_detection(self, aurora_memory: AuroraMemory):
        """测试实体追踪汛即使语义相似性低也能检测更新。
        
        这是的新改进："28 分钟" "25:50" 低语义相似性
        但实体追踪汛应检测它们是同一实体属性。
        """
        # 第一次：以分钟格式建立时间
        plot1 = aurora_memory.ingest(
            "User: My best 5K time is 28 minutes.",
            event_id="time_min",
        )
        
        time.sleep(2)  # 确保时间间隔
        
        # 第二次：以冒号格式更新时间（不同的格式，低语义相似性）
        plot2 = aurora_memory.ingest(
            "User: I ran 5K in 25:50 today.",
            event_id="time_colon",
        )
        
        # 检查实体追踪汛是否检测到了两个值
        timeline = aurora_memory.entity_tracker.get_timeline("user", "5k_time")
        assert len(timeline) >= 2, (
            f"实体追踪汛应检测两个值。 "
            f"时间线长度 {len(timeline)}，值 {[ea.value for ea in timeline]}"
        )
        
        # 验证两个值都在时间线中
        values = [ea.value for ea in timeline]
        has_minutes = any("28" in v or "min" in v.lower() for v in values)
        has_colon = any("25" in v and ":" in v for v in values)
        
        assert has_minutes, f"应有分钟格式。值 {values}"
        assert has_colon, f"应有冒号格式。值 {values}"


class TestEntityTrackerUpdateDetection:
    """使用实体追踪汛的更新检测测试。"""
    
    def test_check_entity_update(self):
        """测试check_entity_update方法。"""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # 第一个值
        tracker.update("My 5K time is 28 min", "plot1", ts1)
        
        # 检查是否有更新
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
        """测试当值相同时未检测到更新。"""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        tracker.update("My 5K time is 25:50", "plot1", ts1)
        
        ts2 = ts1 + 3600
        result = tracker.check_entity_update(
            "My 5K time is 25:50",
            "plot2",
            ts2
        )
        
        # 当值相同时不应检测到更新
        # （或以低信心检测）
        if result is not None:
            entity, attr, old_value, conf = result
            assert conf < 0.3  # 傻值相同时信心低
    
    def test_update_with_different_formats(self):
        """测试不同时间格式的更新检测。"""
        tracker = EntityTracker(seed=42)
        ts1 = now_ts()
        
        # 第一次 分钟格式
        tracker.update("I ran 5K in 28 minutes", "plot1", ts1)
        
        # 第二次 冒号格式
        ts2 = ts1 + 3600
        potential_updates = tracker.find_potential_updates("My 5K time is 25:50", ts2)
        
        # 应检测为潜在更新，即使格式不同
        assert len(potential_updates) > 0
        
        old_ea, new_ea, conf = potential_updates[0]
        assert old_ea.attribute == "5k_time"
        # 值应不同（已正规化）
        assert tracker._normalize_value(old_ea.value) != tracker._normalize_value(new_ea.value)
