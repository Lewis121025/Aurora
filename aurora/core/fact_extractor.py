"""
AURORA Fact Extractor
======================

Phase 5: Fact-Enhanced Indexing for Multi-Session Recall

第一性原理：
- multi-session 需要跨会话聚合信息
- 当前问题：单纯语义搜索可能无法覆盖所有相关会话
- 解决方案：为每个 Turn 提取关键事实作为额外索引键

Key insight: Facts provide structured anchors that complement semantic embeddings.
When a query mentions "3 books" or "bought yesterday", fact keys enable precise
matching even if the semantic embedding is slightly different.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class ExtractedFact:
    """提取的事实"""
    fact_text: str           # 事实文本（用于索引）
    fact_type: str           # 事实类型（quantity, action, location, time, preference）
    entities: List[str]      # 涉及的实体
    plot_id: Optional[str] = None  # 关联的 plot ID（可选）
    
    def to_state_dict(self) -> dict:
        """序列化为字典"""
        return {
            "fact_text": self.fact_text,
            "fact_type": self.fact_type,
            "entities": self.entities,
            "plot_id": self.plot_id,
        }
    
    @classmethod
    def from_state_dict(cls, d: dict) -> "ExtractedFact":
        """从字典重建"""
        return cls(
            fact_text=d["fact_text"],
            fact_type=d["fact_type"],
            entities=d.get("entities", []),
            plot_id=d.get("plot_id"),
        )


class FactExtractor:
    """从对话中提取关键事实
    
    事实类型：
    - quantity: 数量信息（"3 books", "5 items"）
    - action: 动作信息（"bought", "returned", "ordered"）
    - location: 地点信息（"at store", "from shop"）
    - time: 时间信息（"yesterday", "last week"）
    - preference: 偏好信息（"like Python", "prefer coffee"）
    """
    
    # 事实模式（支持中英文）
    FACT_PATTERNS = {
        'quantity': [
            r'(\d+)\s*(items?|books?|clothes?|things?|件|本|个|条)',
            r'(几个|多少|几本|几件)\s*(\w+)',
        ],
        'action': [
            r'\b(bought|returned|picked up|ordered|received|purchased|bought|returned)\b',
            r'\b(买|退|订|收|购买|退货)\b',
        ],
        'location': [
            r'\b(at|from|to)\s+(store|shop|restaurant|place|library|bookstore)\s+(\w+)',
            r'\b(在|从|到)\s*(商店|书店|餐厅|图书馆|地方)\s*(\w+)',
        ],
        'time': [
            r'\b(yesterday|last week|tomorrow|next week|today|recently)\b',
            r'\b(昨天|上周|明天|下周|今天|最近|之前|以后)\b',
        ],
        'preference': [
            r'\b(like|love|prefer|hate|enjoy|dislike)\s+(\w+)',
            r'\b(喜欢|爱|偏好|讨厌|享受|不喜欢)\s+(\w+)',
        ],
    }
    
    def __init__(self, min_fact_length: int = 3):
        """初始化事实提取器
        
        Args:
            min_fact_length: 最小事实文本长度（过滤太短的事实）
        """
        self.min_fact_length = min_fact_length
        # 编译正则表达式以提高性能
        self._compiled_patterns = {}
        for fact_type, patterns in self.FACT_PATTERNS.items():
            self._compiled_patterns[fact_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract(self, text: str) -> List[ExtractedFact]:
        """从文本提取事实
        
        Args:
            text: 输入文本
            
        Returns:
            提取的事实列表
        """
        if not text or not text.strip():
            return []
        
        facts = []
        text_lower = text.lower()
        
        # 数量事实
        for pattern in self._compiled_patterns.get('quantity', []):
            for match in pattern.finditer(text_lower):
                quantity = match.group(1) if match.lastindex >= 1 else ""
                item = match.group(2) if match.lastindex >= 2 else match.group(1) if match.lastindex >= 1 else ""
                fact_text = f"quantity:{quantity} {item}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(
                        fact_text=fact_text,
                        fact_type="quantity",
                        entities=[item] if item else []
                    ))
        
        # 动作事实
        for pattern in self._compiled_patterns.get('action', []):
            for match in pattern.finditer(text_lower):
                action = match.group(0)
                # 获取上下文（前后各20-30字符）
                start = max(0, match.start() - 20)
                end = min(len(text_lower), match.end() + 30)
                context = text_lower[start:end].strip()
                fact_text = f"action:{action} {context}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(
                        fact_text=fact_text,
                        fact_type="action",
                        entities=[action]
                    ))
        
        # 地点事实
        for pattern in self._compiled_patterns.get('location', []):
            for match in pattern.finditer(text_lower):
                location_parts = [g for g in match.groups() if g]
                fact_text = f"location:{' '.join(location_parts)}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(
                        fact_text=fact_text,
                        fact_type="location",
                        entities=location_parts
                    ))
        
        # 时间事实
        for pattern in self._compiled_patterns.get('time', []):
            for match in pattern.finditer(text_lower):
                time_expr = match.group(0)
                fact_text = f"time:{time_expr}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(
                        fact_text=fact_text,
                        fact_type="time",
                        entities=[time_expr]
                    ))
        
        # 偏好事实
        for pattern in self._compiled_patterns.get('preference', []):
            for match in pattern.finditer(text_lower):
                preference = match.group(1) if match.lastindex >= 1 else match.group(0)
                target = match.group(2) if match.lastindex >= 2 else ""
                fact_text = f"preference:{preference} {target}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(
                        fact_text=fact_text,
                        fact_type="preference",
                        entities=[target] if target else []
                    ))
        
        # 去重：相同的事实文本只保留一个
        seen = set()
        unique_facts = []
        for fact in facts:
            if fact.fact_text not in seen:
                seen.add(fact.fact_text)
                unique_facts.append(fact)
        
        return unique_facts
    
    def augment_plot(self, plot) -> List[ExtractedFact]:
        """为 plot 添加事实索引
        
        提取事实并添加到 plot.fact_keys 字段。

        这里刻意不再为 fact key 额外生成 embedding。当前检索链路只使用
        `fact_keys` 文本匹配，历史实现里生成的 `_fact_embeddings`
        没有任何读取方，却会在每次 ingest/replay 时引入额外远程嵌入调用。
        去掉这条死路径可以显著降低真实服务延迟，并保持现有记忆行为不变。
        
        Args:
            plot: Plot 对象（需要添加 fact_keys 字段）
        """
        facts = self.extract(plot.text)
        
        # 提取事实键（用于文本匹配）
        fact_keys = [f.fact_text for f in facts]
        
        # 设置到 plot 对象（需要 Plot 模型支持 fact_keys 字段）
        if hasattr(plot, 'fact_keys'):
            plot.fact_keys = fact_keys
        else:
            # 如果 Plot 模型还没有 fact_keys 字段，先存储到临时属性
            plot._fact_keys = fact_keys
        
        return facts
