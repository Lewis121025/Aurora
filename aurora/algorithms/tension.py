"""
AURORA Tension Manager
======================

Functional contradiction management - managing contradictions, not eliminating them.

Key insight from philosophy:
- Healthy humans contain contradictions
- These contradictions are sources of adaptability
- Over-consistent identities are fragile

Tension Types:
1. Action-blocking: Must be resolved (prevents action)
2. Identity-threatening: Must be resolved (threatens core values)
3. Adaptive: Should be preserved (provides flexibility)
4. Developmental: Should be accepted (evidence of growth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts
from aurora.utils.math_utils import cosine_sim


class TensionType(Enum):
    """Classification of tension/contradiction types."""
    ACTION_BLOCKING = "action_blocking"       # Prevents action, must resolve
    IDENTITY_THREATENING = "identity_threatening"  # Threatens core values, must resolve
    ADAPTIVE = "adaptive"                     # Provides flexibility, preserve
    DEVELOPMENTAL = "developmental"           # Evidence of growth, accept
    UNKNOWN = "unknown"                       # Needs more analysis


@dataclass
class Tension:
    """
    A tension (potential contradiction) between two elements.
    
    Rather than being a "problem to solve", tensions can be:
    - Necessary for flexibility
    - Signs of growth
    - Sources of adaptability
    """
    id: str
    element_a_id: str
    element_a_type: str  # "plot", "story", "theme"
    element_b_id: str
    element_b_type: str
    
    # The nature of the tension
    description: str
    tension_type: TensionType = TensionType.UNKNOWN
    
    # Severity and context
    severity: float = 0.5  # [0, 1]
    context: str = ""      # When/where this tension manifests
    
    # Resolution tracking
    resolution_status: str = "unresolved"  # "unresolved", "resolved", "preserved", "accepted"
    resolution_notes: str = ""
    
    # Timestamps
    detected_ts: float = field(default_factory=now_ts)
    resolved_ts: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "element_a_id": self.element_a_id,
            "element_a_type": self.element_a_type,
            "element_b_id": self.element_b_id,
            "element_b_type": self.element_b_type,
            "description": self.description,
            "tension_type": self.tension_type.value,
            "severity": self.severity,
            "context": self.context,
            "resolution_status": self.resolution_status,
            "resolution_notes": self.resolution_notes,
            "detected_ts": self.detected_ts,
            "resolved_ts": self.resolved_ts,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tension":
        return cls(
            id=d["id"],
            element_a_id=d["element_a_id"],
            element_a_type=d["element_a_type"],
            element_b_id=d["element_b_id"],
            element_b_type=d["element_b_type"],
            description=d["description"],
            tension_type=TensionType(d.get("tension_type", "unknown")),
            severity=d.get("severity", 0.5),
            context=d.get("context", ""),
            resolution_status=d.get("resolution_status", "unresolved"),
            resolution_notes=d.get("resolution_notes", ""),
            detected_ts=d.get("detected_ts", now_ts()),
            resolved_ts=d.get("resolved_ts"),
        )


@dataclass
class TensionResolution:
    """
    The outcome of handling a tension.
    
    Not all tensions need to be "resolved" in the traditional sense.
    Some should be preserved or accepted.
    """
    tension_id: str
    action: str  # "resolve", "preserve", "accept", "defer"
    rationale: str
    changes_made: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tension_id": self.tension_id,
            "action": self.action,
            "rationale": self.rationale,
            "changes_made": self.changes_made,
        }


class TensionManager:
    """
    Manager for functional contradiction handling.
    
    Philosophy: "The goal is not to eliminate contradictions,
    but to manage them wisely."
    
    - Some contradictions need resolution (they block action)
    - Some contradictions should be preserved (they provide flexibility)
    - Some contradictions should be accepted (they mark growth)
    """
    
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        
        # Track all detected tensions
        self.tensions: Dict[str, Tension] = {}
        
        # Core identity dimensions that should not be contradicted
        self.core_identity_values: List[str] = [
            "帮助性",
            "准确性", 
            "诚实",
        ]
        
        # Action keywords that indicate action-blocking potential
        self.action_keywords: List[str] = [
            "应该", "必须", "不能", "禁止", "should", "must", "cannot"
        ]
    
    def detect_tension(
        self,
        element_a: Dict[str, Any],
        element_b: Dict[str, Any],
        embedding_a: Optional[np.ndarray] = None,
        embedding_b: Optional[np.ndarray] = None,
    ) -> Optional[Tension]:
        """
        Detect if there's a tension between two elements.
        
        Returns None if no significant tension is detected.
        """
        text_a = element_a.get("text", "")
        text_b = element_b.get("text", "")
        
        # Check for semantic opposition
        if embedding_a is not None and embedding_b is not None:
            sim = cosine_sim(embedding_a, embedding_b)
            
            # Very low similarity might indicate opposition
            if sim < -0.2:
                return self._create_tension(
                    element_a, element_b,
                    "语义对立",
                    severity=abs(sim),
                )
        
        # Check for logical contradiction patterns
        contradiction = self._check_logical_contradiction(text_a, text_b)
        if contradiction:
            return self._create_tension(
                element_a, element_b,
                contradiction,
                severity=0.7,
            )
        
        return None
    
    def _check_logical_contradiction(self, text_a: str, text_b: str) -> Optional[str]:
        """Check for logical contradiction patterns."""
        # Simple heuristic checks
        negation_pairs = [
            ("是", "不是"), ("能", "不能"), ("会", "不会"),
            ("应该", "不应该"), ("可以", "不可以"),
            ("yes", "no"), ("can", "cannot"), ("should", "should not"),
        ]
        
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()
        
        for pos, neg in negation_pairs:
            if pos in text_a_lower and neg in text_b_lower:
                return f"逻辑矛盾：'{pos}' vs '{neg}'"
            if neg in text_a_lower and pos in text_b_lower:
                return f"逻辑矛盾：'{neg}' vs '{pos}'"
        
        return None
    
    def _create_tension(
        self,
        element_a: Dict[str, Any],
        element_b: Dict[str, Any],
        description: str,
        severity: float,
    ) -> Tension:
        """Create a new tension object."""
        import uuid
        
        tension = Tension(
            id=str(uuid.uuid4()),
            element_a_id=element_a.get("id", "unknown"),
            element_a_type=element_a.get("type", "unknown"),
            element_b_id=element_b.get("id", "unknown"),
            element_b_type=element_b.get("type", "unknown"),
            description=description,
            severity=severity,
        )
        
        self.tensions[tension.id] = tension
        return tension
    
    def classify_tension(self, tension: Tension) -> TensionType:
        """
        Classify a tension into one of four types.
        
        This is the key philosophical decision point:
        - Not all contradictions are bad
        - Some should be preserved for flexibility
        - Some are signs of growth
        """
        # 1. Check if it blocks action
        if self._blocks_action(tension):
            tension.tension_type = TensionType.ACTION_BLOCKING
            return TensionType.ACTION_BLOCKING
        
        # 2. Check if it threatens core identity
        if self._threatens_core_identity(tension):
            tension.tension_type = TensionType.IDENTITY_THREATENING
            return TensionType.IDENTITY_THREATENING
        
        # 3. Check if it provides flexibility (context-dependent validity)
        if self._provides_flexibility(tension):
            tension.tension_type = TensionType.ADAPTIVE
            return TensionType.ADAPTIVE
        
        # 4. Check if it indicates growth (past vs present self)
        if self._indicates_growth(tension):
            tension.tension_type = TensionType.DEVELOPMENTAL
            return TensionType.DEVELOPMENTAL
        
        tension.tension_type = TensionType.UNKNOWN
        return TensionType.UNKNOWN
    
    def _blocks_action(self, tension: Tension) -> bool:
        """Check if the tension prevents taking action."""
        # Tensions involving direct action imperatives
        desc_lower = tension.description.lower()
        
        for keyword in self.action_keywords:
            if keyword in desc_lower:
                return True
        
        # High severity tensions often block action
        if tension.severity > 0.8:
            return True
        
        return False
    
    def _threatens_core_identity(self, tension: Tension) -> bool:
        """Check if the tension threatens core identity values."""
        desc_lower = tension.description.lower()
        
        for value in self.core_identity_values:
            if value.lower() in desc_lower:
                return True
        
        return False
    
    def _provides_flexibility(self, tension: Tension) -> bool:
        """Check if the tension provides useful flexibility."""
        # Moderate tensions between different contexts often provide flexibility
        # Example: "I am patient" vs "I am efficient"
        # Both can be true in different situations
        
        if 0.3 < tension.severity < 0.6:
            # Check for context-dependent language
            context_indicators = [
                "有时", "在某些情况下", "取决于", "sometimes", "depends", "context"
            ]
            desc_lower = tension.description.lower()
            
            for indicator in context_indicators:
                if indicator in desc_lower:
                    return True
            
            # If elements are from different time periods, likely adaptive
            # (This would need access to the actual elements' timestamps)
            return True
        
        return False
    
    def _indicates_growth(self, tension: Tension) -> bool:
        """Check if the tension indicates growth (past vs present)."""
        # Tensions between "I was" and "I am now" indicate growth
        growth_indicators = [
            "以前", "曾经", "过去", "现在", "now", "used to", "before", "now"
        ]
        
        desc_lower = tension.description.lower()
        
        past_found = any(ind in desc_lower for ind in ["以前", "曾经", "过去", "used to", "before"])
        present_found = any(ind in desc_lower for ind in ["现在", "now", "currently"])
        
        if past_found and present_found:
            return True
        
        # Low severity temporal tensions often indicate growth
        if tension.severity < 0.4:
            return True
        
        return False
    
    def handle_tension(self, tension: Tension) -> TensionResolution:
        """
        Handle a tension based on its type.
        
        This is where the philosophy materializes:
        - Action-blocking and identity-threatening: RESOLVE
        - Adaptive: PRESERVE (mark as context-dependent)
        - Developmental: ACCEPT (mark as growth evidence)
        """
        tension_type = self.classify_tension(tension)
        
        if tension_type == TensionType.ACTION_BLOCKING:
            return self._resolve_tension(tension)
        
        elif tension_type == TensionType.IDENTITY_THREATENING:
            return self._resolve_tension(tension)
        
        elif tension_type == TensionType.ADAPTIVE:
            return self._preserve_tension(tension)
        
        elif tension_type == TensionType.DEVELOPMENTAL:
            return self._accept_tension(tension)
        
        else:
            return self._defer_tension(tension)
    
    def _resolve_tension(self, tension: Tension) -> TensionResolution:
        """Resolve an action-blocking or identity-threatening tension."""
        tension.resolution_status = "resolved"
        tension.resolved_ts = now_ts()
        tension.resolution_notes = "需要解决：会阻碍行动或威胁核心身份"
        
        return TensionResolution(
            tension_id=tension.id,
            action="resolve",
            rationale=f"矛盾类型为{tension.tension_type.value}，需要明确解决以维持行动能力",
            changes_made=["降低其中一个元素的置信度", "添加条件限制"],
        )
    
    def _preserve_tension(self, tension: Tension) -> TensionResolution:
        """Preserve an adaptive tension (provides flexibility)."""
        tension.resolution_status = "preserved"
        tension.resolution_notes = "保留为适应性矛盾：在不同情境下都有效"
        
        return TensionResolution(
            tension_id=tension.id,
            action="preserve",
            rationale="这是一个适应性矛盾，在不同情境下激活不同的一面，保留它能提供灵活性",
            changes_made=["标记为情境依赖", "两个元素都保留"],
        )
    
    def _accept_tension(self, tension: Tension) -> TensionResolution:
        """Accept a developmental tension (evidence of growth)."""
        tension.resolution_status = "accepted"
        tension.resolution_notes = "接受为成长证据：过去的我 vs 现在的我"
        
        return TensionResolution(
            tension_id=tension.id,
            action="accept",
            rationale="这是成长的标志，表明'过去的我'和'现在的我'的差异，这是健康的身份演化",
            changes_made=["标记为成长轨迹", "作为身份演化的证据"],
        )
    
    def _defer_tension(self, tension: Tension) -> TensionResolution:
        """Defer handling of an unknown tension."""
        tension.resolution_status = "unresolved"
        tension.resolution_notes = "需要更多信息来判断"
        
        return TensionResolution(
            tension_id=tension.id,
            action="defer",
            rationale="无法明确分类，暂时保留并继续观察",
            changes_made=[],
        )
    
    def get_unresolved_tensions(self) -> List[Tension]:
        """Get all unresolved tensions."""
        return [t for t in self.tensions.values() if t.resolution_status == "unresolved"]
    
    def get_preserved_tensions(self) -> List[Tension]:
        """Get all preserved (adaptive) tensions."""
        return [t for t in self.tensions.values() if t.resolution_status == "preserved"]
    
    def get_accepted_tensions(self) -> List[Tension]:
        """Get all accepted (developmental) tensions."""
        return [t for t in self.tensions.values() if t.resolution_status == "accepted"]
    
    def get_tension_summary(self) -> Dict[str, Any]:
        """Get a summary of all tensions."""
        by_type = {}
        by_status = {}
        
        for tension in self.tensions.values():
            # By type
            ttype = tension.tension_type.value
            by_type[ttype] = by_type.get(ttype, 0) + 1
            
            # By status
            status = tension.resolution_status
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            "total": len(self.tensions),
            "by_type": by_type,
            "by_status": by_status,
            "unresolved_count": len(self.get_unresolved_tensions()),
            "preserved_count": len(self.get_preserved_tensions()),
            "accepted_count": len(self.get_accepted_tensions()),
        }
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to state dict."""
        return {
            "tensions": {tid: t.to_dict() for tid, t in self.tensions.items()},
            "core_identity_values": self.core_identity_values,
        }
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any], seed: int = 0) -> "TensionManager":
        """Restore from state dict."""
        obj = cls(seed=seed)
        obj.tensions = {
            tid: Tension.from_dict(td) 
            for tid, td in d.get("tensions", {}).items()
        }
        obj.core_identity_values = d.get("core_identity_values", obj.core_identity_values)
        return obj
