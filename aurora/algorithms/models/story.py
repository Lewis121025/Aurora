"""
AURORA StoryArc Model
======================

Mesoscale narrative unit - now organized around RELATIONSHIPS, not just semantic clusters.

Key insight: Identity is defined in relationships. A Story is no longer just
"semantically similar events" but "the narrative of a relationship".

The primary organizational dimension is now `relationship_with`, not semantic similarity.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


# -----------------------------------------------------------------------------
# Relationship Trajectory Data Structure
# -----------------------------------------------------------------------------

@dataclass
class RelationshipMoment:
    """
    A moment in the relationship trajectory.
    
    Captures how the relationship evolved at a specific point in time.
    """
    ts: float                   # Timestamp
    event_summary: str          # Brief summary of what happened
    trust_level: float          # Trust level at this moment [0, 1]
    my_role: str                # My role in this moment
    quality_delta: float = 0.0  # Change in relationship quality
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "ts": self.ts,
            "event_summary": self.event_summary,
            "trust_level": self.trust_level,
            "my_role": self.my_role,
            "quality_delta": self.quality_delta,
        }
    
    # Backward compatibility alias
    to_dict = to_state_dict
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "RelationshipMoment":
        """Reconstruct from state dict."""
        return cls(
            ts=d["ts"],
            event_summary=d["event_summary"],
            trust_level=d.get("trust_level", 0.5),
            my_role=d.get("my_role", "assistant"),
            quality_delta=d.get("quality_delta", 0.0),
        )
    
    # Backward compatibility alias
    from_dict = from_state_dict


# -----------------------------------------------------------------------------
# StoryArc Model
# -----------------------------------------------------------------------------

@dataclass
class StoryArc:
    """
    Mesoscale narrative unit: now a RELATIONSHIP NARRATIVE, not just a semantic cluster.

    Key paradigm shift:
    - Old: Stories group semantically similar plots
    - New: Stories represent relationships, and identity emerges within them

    The primary organizational dimension is `relationship_with`.
    "Who I am" is answered per-relationship via `my_identity_in_this_relationship`.

    Attributes:
        id: Unique identifier
        created_ts: Creation timestamp
        updated_ts: Last update timestamp
        plot_ids: List of plot IDs belonging to this story

    Relationship-centric:
        relationship_with: The entity ID this story is about (primary key)
        relationship_type: Type of relationship ("user", "system", "concept")
        relationship_arc: Trajectory of relationship moments
        my_identity_in_this_relationship: "Who I am in this relationship"
        lessons_from_relationship: What this relationship has taught me
        relationship_health: Current health/quality of the relationship [0, 1]

    Narrative Structure (叙事阶段):
        setup: 开端 - 故事的起始情境
        rising_action: 发展 - 情节推进事件列表
        climax: 高潮 - 张力最高点
        falling_action: 收尾 - 高潮后的情节
        resolution: 结局 - 最终解决
        central_conflict: 核心冲突
        turning_points: 转折点列表 (timestamp, description)
        moral: 寓意 - 从故事中提取的意义

    Generative parameters:
        centroid: Mean embedding of plots in this story
        dist_mean, dist_m2, dist_n: Welford stats for semantic dispersion
        gap_mean, gap_m2, gap_n: Welford stats for temporal gaps

    Metadata:
        actor_counts: Count of each actor's appearances
        tension_curve: History of tension values
        status: Story lifecycle status
        reference_count: How often this story is referenced
    """

    id: str
    created_ts: float
    updated_ts: float
    plot_ids: List[str] = field(default_factory=list)

    # === Relationship-centric fields (NEW - primary organizational dimension) ===
    relationship_with: Optional[str] = None         # Entity ID (primary key for organization)
    relationship_type: str = "user"                 # "user", "system", "concept"
    relationship_arc: List[RelationshipMoment] = field(default_factory=list)
    my_identity_in_this_relationship: str = ""      # "Who I am in this relationship"
    lessons_from_relationship: List[str] = field(default_factory=list)
    relationship_health: float = 0.5                # Current relationship quality [0, 1]

    # === Narrative Structure (叙事阶段) ===
    setup: Optional[str] = None                     # 开端 - 故事的起始情境
    rising_action: List[str] = field(default_factory=list)  # 发展 - 情节推进
    climax: Optional[str] = None                    # 高潮 - 张力最高点
    falling_action: List[str] = field(default_factory=list)  # 收尾
    resolution: Optional[str] = None                # 结局 - 最终解决
    
    # === Narrative Elements (叙事元素) ===
    central_conflict: Optional[str] = None          # 核心冲突
    turning_points: List[Tuple[float, str]] = field(default_factory=list)  # 转折点 (timestamp, description)
    moral: Optional[str] = None                     # 寓意 - 从故事中提取的意义

    # Online generative parameters (kept for compatibility)
    centroid: Optional[np.ndarray] = None

    # Stats for semantic dispersion (Welford's algorithm)
    dist_mean: float = 0.0
    dist_m2: float = 0.0
    dist_n: int = 0

    # Stats for temporal gaps (Welford's algorithm)
    gap_mean: float = 0.0
    gap_m2: float = 0.0
    gap_n: int = 0

    actor_counts: Dict[str, int] = field(default_factory=dict)
    tension_curve: List[float] = field(default_factory=list)

    status: Literal["developing", "resolved", "abandoned"] = "developing"
    reference_count: int = 0

    def _update_stats(self, name: str, x: float) -> None:
        """
        Update running statistics using Welford's algorithm.

        Args:
            name: Either "dist" (distance) or "gap" (temporal gap)
            x: New observation value
        """
        if name == "dist":
            self.dist_n += 1
            delta = x - self.dist_mean
            self.dist_mean += delta / self.dist_n
            self.dist_m2 += delta * (x - self.dist_mean)
        elif name == "gap":
            self.gap_n += 1
            delta = x - self.gap_mean
            self.gap_mean += delta / self.gap_n
            self.gap_m2 += delta * (x - self.gap_mean)
        else:
            raise ValueError(f"Unknown stat name: {name}")

    def dist_var(self) -> float:
        """Get variance of distance statistics."""
        return self.dist_m2 / (self.dist_n - 1) if self.dist_n > 1 else 1.0

    def gap_mean_safe(self, default: float = 3600.0) -> float:
        """Get gap mean with safe default."""
        return self.gap_mean if self.gap_n > 0 and self.gap_mean > 0 else default

    def activity_probability(self, ts: Optional[float] = None) -> float:
        """
        Probability the story is still active under a learned temporal hazard model.

        If a story usually gets updates every ~gap_mean seconds, then being idle
        >> gap_mean should reduce activity probability smoothly (not via a fixed
        threshold).

        Args:
            ts: Current timestamp (defaults to now)

        Returns:
            Activity probability in (0, 1)
        """
        ts = ts or now_ts()
        idle = max(0.0, ts - self.updated_ts)
        tau = self.gap_mean_safe()
        # Survival function of exponential: P(active) ~ exp(-idle/tau)
        return math.exp(-idle / max(tau, 1e-6))

    def mass(self) -> float:
        """
        Emergent importance at story level.

        Returns:
            Mass value combining freshness, size, and references
        """
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        size = math.log1p(len(self.plot_ids))
        return freshness * (size + math.log1p(self.reference_count + 1))
    
    # -------------------------------------------------------------------------
    # Relationship-centric Methods
    # -------------------------------------------------------------------------
    
    def is_relationship_story(self) -> bool:
        """Check if this story is organized around a relationship."""
        return self.relationship_with is not None
    
    def add_relationship_moment(
        self,
        event_summary: str,
        trust_level: float,
        my_role: str,
        quality_delta: float = 0.0,
        ts: Optional[float] = None
    ) -> None:
        """Add a moment to the relationship trajectory."""
        moment = RelationshipMoment(
            ts=ts or now_ts(),
            event_summary=event_summary,
            trust_level=trust_level,
            my_role=my_role,
            quality_delta=quality_delta,
        )
        self.relationship_arc.append(moment)
        
        # Update relationship health based on quality delta
        self.relationship_health = max(0.0, min(1.0, 
            self.relationship_health + quality_delta * 0.1
        ))
    
    def get_trust_trend(self, window: int = 10) -> float:
        """
        Get the trend of trust level over recent interactions.
        
        Returns:
            Positive if trust is increasing, negative if decreasing.
        """
        if len(self.relationship_arc) < 2:
            return 0.0
        
        recent = self.relationship_arc[-window:]
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        avg_first = sum(m.trust_level for m in first_half) / len(first_half)
        avg_second = sum(m.trust_level for m in second_half) / len(second_half)
        
        return avg_second - avg_first
    
    def get_role_consistency(self, window: int = 10) -> float:
        """
        Get the consistency of my role in this relationship.
        
        Returns:
            1.0 if perfectly consistent, lower if role varies.
        """
        if len(self.relationship_arc) < 2:
            return 1.0
        
        recent = self.relationship_arc[-window:]
        roles = [m.my_role for m in recent]
        
        if not roles:
            return 1.0
        
        # Count most common role using Counter
        role_counts = Counter(roles)
        max_count = role_counts.most_common(1)[0][1] if role_counts else 0
        return max_count / len(roles)

    # -------------------------------------------------------------------------
    # Temporal Methods (时间作为一等公民)
    # -------------------------------------------------------------------------
    
    def get_temporal_span(self, plots_dict: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """Get the temporal span of this story.
        
        Time as First-Class Citizen: A story's time span is essential for
        understanding its narrative arc and temporal context.
        
        Args:
            plots_dict: Optional dict of plot_id -> Plot for precise timestamps.
                        If not provided, uses created_ts and updated_ts.
        
        Returns:
            Tuple of (earliest_timestamp, latest_timestamp).
            If no plots, returns (created_ts, updated_ts).
        """
        if not self.plot_ids:
            return (self.created_ts, self.updated_ts)
        
        if plots_dict is None:
            # Fallback to story timestamps
            return (self.created_ts, self.updated_ts)
        
        # Get timestamps from actual plots
        timestamps = []
        for pid in self.plot_ids:
            plot = plots_dict.get(pid)
            if plot is not None:
                ts = getattr(plot, 'ts', None)
                if ts is not None:
                    timestamps.append(ts)
        
        if not timestamps:
            return (self.created_ts, self.updated_ts)
        
        return (min(timestamps), max(timestamps))
    
    def get_temporal_narrative(
        self, 
        plots_dict: Optional[Dict[str, Any]] = None,
        locale: str = "zh"
    ) -> str:
        """Generate a temporal narrative summary of this story.
        
        Time as First-Class Citizen: Generates a human-readable description
        of the story's temporal dimension, helping users understand the
        timeline of events.
        
        Example outputs:
        - Chinese: "从2024年1月开始，用户和我讨论了Python学习，历经3个月，共12次交互..."
        - English: "Starting from January 2024, discussed Python learning over 3 months, 12 interactions..."
        
        Args:
            plots_dict: Optional dict of plot_id -> Plot for event details.
            locale: Language for the narrative ("zh" for Chinese, "en" for English).
        
        Returns:
            A natural language description of the story's temporal arc.
        """
        import datetime
        
        # Get temporal span
        start_ts, end_ts = self.get_temporal_span(plots_dict)
        
        # Convert to datetime
        start_dt = datetime.datetime.fromtimestamp(start_ts)
        end_dt = datetime.datetime.fromtimestamp(end_ts)
        
        # Calculate duration
        duration_days = (end_ts - start_ts) / (24 * 3600)
        interaction_count = len(self.plot_ids)
        
        # Format dates
        if locale == "zh":
            start_str = start_dt.strftime("%Y年%m月%d日")
            end_str = end_dt.strftime("%Y年%m月%d日")
            
            # Build narrative
            parts = []
            
            if self.relationship_with:
                parts.append(f"与{self.relationship_with}的关系故事")
            else:
                parts.append(f"故事{self.id[:8]}")
            
            parts.append(f"从{start_str}开始")
            
            if duration_days < 1:
                parts.append(f"在同一天内")
            elif duration_days < 7:
                parts.append(f"历经{int(duration_days)}天")
            elif duration_days < 30:
                weeks = int(duration_days / 7)
                parts.append(f"历经约{weeks}周")
            elif duration_days < 365:
                months = int(duration_days / 30)
                parts.append(f"历经约{months}个月")
            else:
                years = duration_days / 365
                parts.append(f"历经约{years:.1f}年")
            
            parts.append(f"共{interaction_count}次交互")
            
            # Add relationship context if available
            if self.my_identity_in_this_relationship:
                parts.append(f"我作为{self.my_identity_in_this_relationship}")
            
            # Add narrative phase
            phase = self.get_narrative_phase()
            phase_names = {
                "setup": "处于开端阶段",
                "rising": "正在发展中",
                "climax": "达到高潮",
                "falling": "进入收尾",
                "resolution": "已经完结",
                "unknown": "阶段未明",
            }
            parts.append(phase_names.get(phase, ""))
            
            return "，".join(parts) + "。"
        
        else:  # English
            start_str = start_dt.strftime("%B %d, %Y")
            end_str = end_dt.strftime("%B %d, %Y")
            
            parts = []
            
            if self.relationship_with:
                parts.append(f"Story with {self.relationship_with}")
            else:
                parts.append(f"Story {self.id[:8]}")
            
            parts.append(f"starting from {start_str}")
            
            if duration_days < 1:
                parts.append("on the same day")
            elif duration_days < 7:
                parts.append(f"spanning {int(duration_days)} days")
            elif duration_days < 30:
                weeks = int(duration_days / 7)
                parts.append(f"over about {weeks} weeks")
            elif duration_days < 365:
                months = int(duration_days / 30)
                parts.append(f"over about {months} months")
            else:
                years = duration_days / 365
                parts.append(f"over about {years:.1f} years")
            
            parts.append(f"with {interaction_count} interactions")
            
            if self.my_identity_in_this_relationship:
                parts.append(f"acting as {self.my_identity_in_this_relationship}")
            
            phase = self.get_narrative_phase()
            phase_names = {
                "setup": "in setup phase",
                "rising": "currently developing",
                "climax": "at climax",
                "falling": "in falling action",
                "resolution": "resolved",
                "unknown": "phase unknown",
            }
            parts.append(phase_names.get(phase, ""))
            
            return ", ".join(parts) + "."
    
    def get_temporal_density(self, plots_dict: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the temporal density of interactions.
        
        Higher density means more frequent interactions in the time span.
        
        Args:
            plots_dict: Optional dict of plot_id -> Plot.
        
        Returns:
            Interactions per day, or 0 if no temporal span.
        """
        start_ts, end_ts = self.get_temporal_span(plots_dict)
        duration_days = (end_ts - start_ts) / (24 * 3600)
        
        if duration_days < 0.001:  # Less than a minute
            return float(len(self.plot_ids))  # All interactions in single moment
        
        return len(self.plot_ids) / duration_days
    
    def update_identity_in_relationship(self, new_identity: str) -> None:
        """Update my identity in this relationship."""
        self.my_identity_in_this_relationship = new_identity
    
    def add_lesson(self, lesson: str) -> None:
        """Add a lesson learned from this relationship."""
        if lesson not in self.lessons_from_relationship:
            self.lessons_from_relationship.append(lesson)
    
    def get_current_trust(self) -> float:
        """Get the current trust level in this relationship."""
        if not self.relationship_arc:
            return 0.5  # Neutral default
        return self.relationship_arc[-1].trust_level
    
    def to_relationship_narrative(self) -> str:
        """Generate a natural language narrative of this relationship."""
        if not self.is_relationship_story():
            return f"Story {self.id} (not a relationship story)"
        
        parts = []
        
        # Relationship identity
        if self.my_identity_in_this_relationship:
            parts.append(f"在与 {self.relationship_with} 的关系中，我是{self.my_identity_in_this_relationship}。")
        
        # Trust and health
        trust = self.get_current_trust()
        if trust > 0.7:
            parts.append("我们建立了良好的信任关系。")
        elif trust > 0.4:
            parts.append("我们的关系正在发展中。")
        else:
            parts.append("我们的关系还需要培养。")
        
        # Lessons
        if self.lessons_from_relationship:
            lessons = "、".join(self.lessons_from_relationship[:3])
            parts.append(f"这段关系教会我：{lessons}。")
        
        return "".join(parts)
    
    # -------------------------------------------------------------------------
    # Narrative Structure Methods (叙事结构方法)
    # -------------------------------------------------------------------------
    
    def detect_climax(self, threshold_percentile: float = 0.9) -> Optional[int]:
        """
        Detect the climax point based on tension_curve.
        
        The climax is identified as the point with maximum tension that exceeds
        a threshold percentile of all tension values.
        
        Args:
            threshold_percentile: Percentile threshold for considering a point
                                  as climax (default 0.9 = 90th percentile)
        
        Returns:
            Index of the climax point in tension_curve, or None if no clear climax.
        """
        if not self.tension_curve or len(self.tension_curve) < 3:
            return None
        
        tensions = np.array(self.tension_curve)
        max_idx = int(np.argmax(tensions))
        max_tension = tensions[max_idx]
        
        # Check if the max tension exceeds the threshold percentile
        threshold = np.percentile(tensions, threshold_percentile * 100)
        if max_tension >= threshold:
            return max_idx
        
        return None
    
    def extract_moral(self) -> Optional[str]:
        """
        Extract the moral/meaning from this story.
        
        This is a placeholder implementation that generates a moral based on
        the story's structure and lessons learned. In production, this would
        be augmented by LLM-based meaning extraction.
        
        Returns:
            A string describing the moral/meaning, or None if insufficient data.
        """
        # If already set, return it
        if self.moral:
            return self.moral
        
        # Synthesize from available information
        parts = []
        
        # From relationship lessons
        if self.lessons_from_relationship:
            parts.append(f"从关系中学到：{self.lessons_from_relationship[0]}")
        
        # From central conflict
        if self.central_conflict and self.resolution:
            parts.append(f"面对「{self.central_conflict}」，最终{self.resolution}")
        
        # From tension pattern
        if self.tension_curve and len(self.tension_curve) >= 3:
            trend = self.tension_curve[-1] - self.tension_curve[0]
            if trend > 0.2:
                parts.append("张力逐渐升高，故事仍在发展中")
            elif trend < -0.2:
                parts.append("经历起伏后趋于平静")
            else:
                parts.append("故事维持相对稳定的节奏")
        
        if parts:
            self.moral = "；".join(parts)
            return self.moral
        
        return None
    
    def add_turning_point(self, ts: float, description: str) -> None:
        """
        Add a turning point to the narrative.
        
        Turning points are significant moments where the story direction changes.
        They are stored with timestamps for chronological ordering.
        
        Args:
            ts: Timestamp of the turning point
            description: Description of what changed
        """
        self.turning_points.append((ts, description))
        # Keep sorted by timestamp
        self.turning_points.sort(key=lambda x: x[0])
    
    def get_narrative_phase(self) -> Literal["setup", "rising", "climax", "falling", "resolution", "unknown"]:
        """
        Determine the current narrative phase based on story structure.
        
        Returns:
            The current phase of the narrative arc.
        """
        if self.resolution:
            return "resolution"
        if self.falling_action:
            return "falling"
        if self.climax:
            return "climax"
        if self.rising_action:
            return "rising"
        if self.setup:
            return "setup"
        return "unknown"
    
    def get_narrative_completeness(self) -> float:
        """
        Calculate how complete the narrative structure is.
        
        Returns:
            A score from 0.0 to 1.0 indicating narrative completeness.
        """
        score = 0.0
        weights = {
            "setup": 0.15,
            "rising_action": 0.20,
            "climax": 0.25,
            "falling_action": 0.15,
            "resolution": 0.15,
            "central_conflict": 0.05,
            "moral": 0.05,
        }
        
        if self.setup:
            score += weights["setup"]
        if self.rising_action:
            score += weights["rising_action"]
        if self.climax:
            score += weights["climax"]
        if self.falling_action:
            score += weights["falling_action"]
        if self.resolution:
            score += weights["resolution"]
        if self.central_conflict:
            score += weights["central_conflict"]
        if self.moral:
            score += weights["moral"]
        
        return min(1.0, score)
    
    def to_narrative_summary(self) -> str:
        """
        Generate a structured narrative summary of this story.
        
        Returns:
            A formatted string summarizing the narrative arc.
        """
        parts = []
        
        if self.central_conflict:
            parts.append(f"【核心冲突】{self.central_conflict}")
        
        if self.setup:
            parts.append(f"【开端】{self.setup}")
        
        if self.rising_action:
            rising = "→".join(self.rising_action[:3])
            if len(self.rising_action) > 3:
                rising += f"...（共{len(self.rising_action)}个发展）"
            parts.append(f"【发展】{rising}")
        
        if self.climax:
            parts.append(f"【高潮】{self.climax}")
        
        if self.falling_action:
            falling = "→".join(self.falling_action[:2])
            parts.append(f"【收尾】{falling}")
        
        if self.resolution:
            parts.append(f"【结局】{self.resolution}")
        
        if self.turning_points:
            tp_str = "; ".join([f"{desc}" for _, desc in self.turning_points[:3]])
            parts.append(f"【转折点】{tp_str}")
        
        if self.moral:
            parts.append(f"【寓意】{self.moral}")
        
        if not parts:
            return f"Story {self.id}（叙事结构待完善）"
        
        return "\n".join(parts)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "plot_ids": self.plot_ids,
            # Relationship-centric fields
            "relationship_with": self.relationship_with,
            "relationship_type": self.relationship_type,
            "relationship_arc": [m.to_state_dict() for m in self.relationship_arc],
            "my_identity_in_this_relationship": self.my_identity_in_this_relationship,
            "lessons_from_relationship": self.lessons_from_relationship,
            "relationship_health": self.relationship_health,
            # Narrative structure (叙事阶段)
            "setup": self.setup,
            "rising_action": self.rising_action,
            "climax": self.climax,
            "falling_action": self.falling_action,
            "resolution": self.resolution,
            # Narrative elements (叙事元素)
            "central_conflict": self.central_conflict,
            "turning_points": self.turning_points,  # List[Tuple[float, str]] is JSON-serializable
            "moral": self.moral,
            # Generative parameters
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "dist_mean": self.dist_mean,
            "dist_m2": self.dist_m2,
            "dist_n": self.dist_n,
            "gap_mean": self.gap_mean,
            "gap_m2": self.gap_m2,
            "gap_n": self.gap_n,
            "actor_counts": self.actor_counts,
            "tension_curve": self.tension_curve,
            "status": self.status,
            "reference_count": self.reference_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "StoryArc":
        """Reconstruct from state dict.
        
        Backward compatible: handles old data without narrative structure fields.
        """
        centroid = d.get("centroid")
        
        # Parse relationship arc
        relationship_arc = []
        if "relationship_arc" in d:
            relationship_arc = [
                RelationshipMoment.from_state_dict(m) for m in d["relationship_arc"]
            ]
        
        # Parse turning points - convert lists back to tuples for type consistency
        raw_turning_points = d.get("turning_points", [])
        turning_points: List[Tuple[float, str]] = [
            (float(tp[0]), str(tp[1])) for tp in raw_turning_points
        ] if raw_turning_points else []
        
        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            plot_ids=d.get("plot_ids", []),
            # Relationship-centric fields
            relationship_with=d.get("relationship_with"),
            relationship_type=d.get("relationship_type", "user"),
            relationship_arc=relationship_arc,
            my_identity_in_this_relationship=d.get("my_identity_in_this_relationship", ""),
            lessons_from_relationship=d.get("lessons_from_relationship", []),
            relationship_health=d.get("relationship_health", 0.5),
            # Narrative structure (backward compatible defaults)
            setup=d.get("setup"),
            rising_action=d.get("rising_action", []),
            climax=d.get("climax"),
            falling_action=d.get("falling_action", []),
            resolution=d.get("resolution"),
            # Narrative elements (backward compatible defaults)
            central_conflict=d.get("central_conflict"),
            turning_points=turning_points,
            moral=d.get("moral"),
            # Generative parameters
            centroid=np.array(centroid, dtype=np.float32) if centroid is not None else None,
            dist_mean=d.get("dist_mean", 0.0),
            dist_m2=d.get("dist_m2", 0.0),
            dist_n=d.get("dist_n", 0),
            gap_mean=d.get("gap_mean", 0.0),
            gap_m2=d.get("gap_m2", 0.0),
            gap_n=d.get("gap_n", 0),
            actor_counts=d.get("actor_counts", {}),
            tension_curve=d.get("tension_curve", []),
            status=d.get("status", "developing"),
            reference_count=d.get("reference_count", 0),
        )
