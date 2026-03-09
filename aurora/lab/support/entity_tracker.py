"""
Entity-Attribute Tracker for Knowledge Updates
===============================================

Phase 3: Entity-attribute tracking to improve knowledge-update accuracy.

First Principles:
- Knowledge updates are changes to the same entity's attribute over time
- Current semantic similarity detection fails when "28 min" and "25:50" have low similarity
- Need entity-attribute alignment, not pure semantic similarity

Example:
- Entity: "user's 5K time"
- Attribute: "personal_best"
- Timeline: ["28 min" (t1), "25:50" (t2)] → detected as update even if similarity is low
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


@dataclass
class EntityAttribute:
    """实体属性记录"""
    entity: str           # "user" or entity name
    attribute: str        # "5k_time", "phone", "location", etc.
    value: str            # "25:50", "123-456-7890", "Shanghai", etc.
    plot_id: str          # 关联的 plot ID
    timestamp: float      # 记录时间
    confidence: float = 1.0  # 提取置信度
    context: str = ""     # 提取时的上下文文本片段


class EntityTracker:
    """追踪实体属性随时间的变化
    
    核心思想：
    1. 从文本中提取实体-属性-值三元组
    2. 维护每个实体-属性的时间线
    3. 检测同一实体-属性的值变化（即使语义相似度低）
    """
    
    def __init__(self, seed: int = 0):
        """初始化实体追踪器
        
        Args:
            seed: 随机种子（用于确定性）
        """
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        
        # 时间线：key = "entity::attribute", value = List[EntityAttribute] (按时间排序)
        self.timelines: Dict[str, List[EntityAttribute]] = {}
        
        # 预编译正则模式以提高效率
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """预编译正则表达式模式"""
        # 手机号模式（支持多种格式）
        self._phone_pattern = re.compile(
            r'(?:phone|mobile|cell|tel)[\w\s]*?(?:is|:)?\s*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\d{10,11})',
            re.IGNORECASE
        )
        
        # 邮箱模式
        self._email_pattern = re.compile(
            r'[\w.-]+@[\w.-]+\.\w+',
            re.IGNORECASE
        )
        
        # 地址/位置模式
        self._location_patterns = [
            re.compile(r'(?:live|moved?|work|located|reside|stay)\s+(?:in|to|at|near)\s+([A-Z][\w\s]+)', re.IGNORECASE),
            re.compile(r'(?:address|location|city|place)\s+(?:is|:)?\s*([A-Z][\w\s]+)', re.IGNORECASE),
        ]
        
        # 时间/成绩模式（5K time, running time, etc.）
        self._time_patterns = [
            # "25:50", "28:30", "1:23:45"
            re.compile(r'(\d{1,2}:\d{2}(?::\d{2})?)', re.IGNORECASE),
            # "28 min", "30 minutes", "1 hour 20 min"
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|min)', re.IGNORECASE),
            # "5K time: 25:50"
            re.compile(r'(?:5k|5-k|5\s*k)[\s:]+(?:time|pace|result)[\s:]*(\d{1,2}:\d{2}|\d+(?:\.\d+)?\s*min)', re.IGNORECASE),
        ]
        
        # 数值属性模式（年龄、价格、数量等）
        self._numeric_patterns = [
            re.compile(r'(?:age|years? old)[\s:]*(\d+)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(?:dollars?|USD|\$)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*%', re.IGNORECASE),
        ]
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, str, float, str]]:
        """从文本提取实体-属性-值三元组
        
        Args:
            text: 输入文本
            
        Returns:
            List of (entity, attribute, value, confidence, context) tuples
        """
        entities = []
        text_lower = text.lower()
        
        # 1. 手机号
        phone_match = self._phone_pattern.search(text)
        if phone_match:
            value = phone_match.group(1).replace(' ', '').replace('-', '').replace('.', '')
            context = self._extract_context(text, phone_match.start(), phone_match.end())
            entities.append(("user", "phone", value, 0.9, context))
        
        # 2. 邮箱
        email_match = self._email_pattern.search(text)
        if email_match:
            value = email_match.group(0)
            context = self._extract_context(text, email_match.start(), email_match.end())
            entities.append(("user", "email", value, 0.9, context))
        
        # 3. 地址/位置
        for pattern in self._location_patterns:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                context = self._extract_context(text, match.start(), match.end())
                entities.append(("user", "location", value, 0.8, context))
                break  # 只取第一个匹配
        
        # 4. 时间/成绩（5K time, running time等）
        # 检查是否有跑步/比赛上下文
        running_context = any(word in text_lower for word in ['run', 'race', '5k', 'marathon', 'pace', 'time', 'personal', 'best', 'pb'])
        
        for pattern in self._time_patterns:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                context = self._extract_context(text, match.start(), match.end())
                
                # 如果有跑步上下文，标记为 5k_time
                if running_context or '5k' in text_lower:
                    entities.append(("user", "5k_time", value, 0.85, context))
                else:
                    # 通用时间属性
                    entities.append(("user", "time_duration", value, 0.7, context))
                break
        
        # 5. 数值属性
        for pattern in self._numeric_patterns:
            match = pattern.search(text)
            if match:
                value = match.group(1).strip()
                context = self._extract_context(text, match.start(), match.end())
                
                if 'age' in text_lower or 'years old' in text_lower:
                    entities.append(("user", "age", value, 0.85, context))
                elif '$' in text or 'dollar' in text_lower or 'usd' in text_lower:
                    entities.append(("user", "price", value, 0.8, context))
                elif '%' in text:
                    entities.append(("user", "percentage", value, 0.75, context))
                break
        
        # 去重：相同 entity::attribute 只保留置信度最高的
        seen = {}
        for entity, attr, value, conf, ctx in entities:
            key = f"{entity}::{attr}"
            if key not in seen or conf > seen[key][3]:
                seen[key] = (entity, attr, value, conf, ctx)
        
        return list(seen.values())
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """提取匹配位置周围的上下文"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def update(self, text: str, plot_id: str, timestamp: float) -> List[EntityAttribute]:
        """更新实体属性时间线
        
        Args:
            text: 交互文本
            plot_id: 关联的 plot ID
            timestamp: 时间戳
            
        Returns:
            更新的 EntityAttribute 列表
        """
        entities = self.extract_entities(text)
        updated = []
        
        for entity, attr, value, confidence, context in entities:
            key = f"{entity}::{attr}"
            
            ea = EntityAttribute(
                entity=entity,
                attribute=attr,
                value=value,
                plot_id=plot_id,
                timestamp=timestamp,
                confidence=confidence,
                context=context
            )
            
            if key not in self.timelines:
                self.timelines[key] = []
            
            self.timelines[key].append(ea)
            updated.append(ea)
            
            # 按时间戳排序
            self.timelines[key].sort(key=lambda x: x.timestamp)
        
        return updated
    
    def get_latest(self, entity: str, attribute: str) -> Optional[EntityAttribute]:
        """获取实体属性的最新值
        
        Args:
            entity: 实体名称
            attribute: 属性名称
            
        Returns:
            最新的 EntityAttribute，如果不存在则返回 None
        """
        key = f"{entity}::{attribute}"
        if key not in self.timelines or not self.timelines[key]:
            return None
        return max(self.timelines[key], key=lambda x: x.timestamp)
    
    def get_timeline(self, entity: str, attribute: str) -> List[EntityAttribute]:
        """获取实体属性的完整时间线
        
        Args:
            entity: 实体名称
            attribute: 属性名称
            
        Returns:
            按时间排序的 EntityAttribute 列表
        """
        key = f"{entity}::{attribute}"
        return sorted(self.timelines.get(key, []), key=lambda x: x.timestamp)
    
    def find_potential_updates(
        self, 
        text: str, 
        timestamp: float,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[EntityAttribute, EntityAttribute, float]]:
        """查找可能的知识更新
        
        检查新文本中的实体-属性是否与已有实体-属性匹配，
        即使语义相似度较低也能检测到更新。
        
        Args:
            text: 新文本
            timestamp: 新文本的时间戳
            similarity_threshold: 语义相似度阈值（用于过滤）
            
        Returns:
            List of (old_ea, new_ea, confidence) tuples
            其中 old_ea 是旧的 EntityAttribute，new_ea 是新的
        """
        new_entities = self.extract_entities(text)
        potential_updates = []
        
        for entity, attr, value, confidence, context in new_entities:
            key = f"{entity}::{attr}"
            
            # 查找该实体-属性的历史记录
            if key not in self.timelines:
                continue
            
            timeline = self.timelines[key]
            if not timeline:
                continue
            
            # 获取最新的值
            latest = timeline[-1]
            
            # 检查值是否不同
            if self._normalize_value(value) != self._normalize_value(latest.value):
                # 检查时间间隔（避免短期重复）
                time_gap = timestamp - latest.timestamp
                if time_gap > 60:  # 至少1分钟间隔
                    # 计算更新置信度
                    update_confidence = confidence * latest.confidence
                    
                    # 创建新的 EntityAttribute（临时，用于返回）
                    new_ea = EntityAttribute(
                        entity=entity,
                        attribute=attr,
                        value=value,
                        plot_id="",  # 将在调用方设置
                        timestamp=timestamp,
                        confidence=confidence,
                        context=context
                    )
                    
                    potential_updates.append((latest, new_ea, update_confidence))
        
        return potential_updates
    
    def _normalize_value(self, value: str) -> str:
        """标准化值以便比较
        
        例如："28 min" 和 "28:00" 可能表示相同的时间
        """
        value = value.strip().lower()
        
        # 时间格式标准化
        # "28 min" -> "28:00"
        time_min_match = re.match(r'(\d+(?:\.\d+)?)\s*min', value)
        if time_min_match:
            minutes = float(time_min_match.group(1))
            return f"{int(minutes)}:{int((minutes % 1) * 60):02d}"
        
        # 标准化时间格式 "25:50" -> "25:50"
        time_colon_match = re.match(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', value)
        if time_colon_match:
            hours = int(time_colon_match.group(1))
            minutes = int(time_colon_match.group(2))
            seconds = int(time_colon_match.group(3)) if time_colon_match.group(3) else 0
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        
        # 移除空格和标点
        normalized = re.sub(r'[\s\-\.]', '', value)
        return normalized
    
    def check_entity_update(
        self,
        text: str,
        plot_id: str,
        timestamp: float,
        candidate_plot_id: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, float]]:
        """检查文本是否是对已有 plot 的实体-属性更新
        
        Args:
            text: 新文本
            plot_id: 新 plot 的 ID
            timestamp: 新 plot 的时间戳
            candidate_plot_id: 候选的旧 plot ID（如果已知）
            
        Returns:
            Optional tuple of (entity, attribute, old_value, confidence)
            如果检测到更新，返回实体、属性、旧值和置信度
        """
        potential_updates = self.find_potential_updates(text, timestamp)
        
        if not potential_updates:
            return None
        
        # 如果有候选 plot_id，优先匹配
        if candidate_plot_id:
            for old_ea, new_ea, conf in potential_updates:
                if old_ea.plot_id == candidate_plot_id:
                    return (old_ea.entity, old_ea.attribute, old_ea.value, conf)
        
        # 否则返回置信度最高的
        best = max(potential_updates, key=lambda x: x[2])
        old_ea, new_ea, conf = best
        return (old_ea.entity, old_ea.attribute, old_ea.value, conf)
    
    def to_state_dict(self) -> Dict[str, Any]:
        """序列化状态"""
        return {
            "seed": self._seed,
            "timelines": {
                key: [
                    {
                        "entity": ea.entity,
                        "attribute": ea.attribute,
                        "value": ea.value,
                        "plot_id": ea.plot_id,
                        "timestamp": ea.timestamp,
                        "confidence": ea.confidence,
                        "context": ea.context,
                    }
                    for ea in timeline
                ]
                for key, timeline in self.timelines.items()
            }
        }
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "EntityTracker":
        """从状态字典恢复"""
        tracker = cls(seed=d.get("seed", 0))
        
        timelines_dict = d.get("timelines", {})
        for key, timeline_data in timelines_dict.items():
            tracker.timelines[key] = [
                EntityAttribute(
                    entity=ea["entity"],
                    attribute=ea["attribute"],
                    value=ea["value"],
                    plot_id=ea["plot_id"],
                    timestamp=ea["timestamp"],
                    confidence=ea.get("confidence", 1.0),
                    context=ea.get("context", ""),
                )
                for ea in timeline_data
            ]
        
        return tracker
