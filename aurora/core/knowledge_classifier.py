"""
AURORA Knowledge Type Classifier
================================

Implements first-principle-based knowledge classification for intelligent conflict resolution.

Core insight: Not all contradictions need elimination.

| Knowledge Type    | Example                    | Conflict Handling        |
|-------------------|----------------------------|--------------------------|
| FACTUAL_STATE     | address, phone, job        | UPDATE (keep latest)     |
| FACTUAL_STATIC    | birthday, birthplace       | CORRECT (old is wrong)   |
| IDENTITY_TRAIT    | patient, efficient         | PRESERVE_BOTH (facets)   |
| IDENTITY_VALUE    | honesty, integrity         | PRESERVE_BOTH (core)     |
| PREFERENCE        | likes coffee, likes sports | EVOLVE (track changes)   |
| BEHAVIOR_PATTERN  | works late, exercises      | EVOLVE (patterns change) |

In narrative psychology: Healthy identity contains tensions and contradictions.
"I am patient" and "I am efficient" are not contradictions - they are different
facets activated in different contexts.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from aurora.utils.math_utils import cosine_sim
from aurora.utils.time_utils import now_ts


# =============================================================================
# Knowledge Type Enum
# =============================================================================

class KnowledgeType(Enum):
    """
    Classification of knowledge types for intelligent conflict resolution.
    
    Each type has different implications for how contradictions should be handled:
    - States change over time → UPDATE
    - Static facts don't change → CORRECT
    - Identity traits can coexist → PRESERVE_BOTH
    - Values are foundational → PRESERVE_BOTH
    - Preferences evolve → EVOLVE
    - Behaviors are patterns → EVOLVE
    """
    FACTUAL_STATE = "factual_state"      # Mutable facts (address, job, status)
    FACTUAL_STATIC = "factual_static"    # Immutable facts (birthday, birthplace)
    IDENTITY_TRAIT = "identity_trait"    # Personality traits (patient, efficient)
    IDENTITY_VALUE = "identity_value"    # Core values (honesty, integrity)
    PREFERENCE = "preference"            # Likes/dislikes (coffee, sports)
    BEHAVIOR_PATTERN = "behavior"        # Behavioral patterns (habits, routines)
    UNKNOWN = "unknown"                  # Cannot be classified


class ConflictResolution(Enum):
    """
    Resolution strategies for knowledge conflicts.
    
    The key insight: Different knowledge types require different strategies.
    Not all conflicts need "resolution" - some should be preserved.
    """
    UPDATE = "update"           # Replace old with new (states, transient facts)
    PRESERVE_BOTH = "preserve"  # Keep both as valid (traits, values)
    CORRECT = "correct"         # Mark old as incorrect (static facts)
    EVOLVE = "evolve"           # Track timeline of changes (preferences, behaviors)
    NO_ACTION = "no_action"     # No conflict exists, no action needed


# =============================================================================
# Classification Result
# =============================================================================

@dataclass
class ClassificationResult:
    """Result of knowledge type classification."""
    knowledge_type: KnowledgeType
    confidence: float  # [0, 1]
    matched_patterns: List[str] = field(default_factory=list)
    subject: Optional[str] = None  # Extracted subject (e.g., "user", "Alice")
    predicate: Optional[str] = None  # Extracted predicate (e.g., "lives in", "likes")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "knowledge_type": self.knowledge_type.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "subject": self.subject,
            "predicate": self.predicate,
        }


@dataclass
class ConflictAnalysis:
    """Analysis of a conflict between two pieces of knowledge."""
    resolution: ConflictResolution
    rationale: str
    confidence: float
    
    # Context about the conflict
    knowledge_type_a: KnowledgeType
    knowledge_type_b: KnowledgeType
    is_complementary: bool = False
    requires_human_review: bool = False
    
    # Action recommendations
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution.value,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "knowledge_type_a": self.knowledge_type_a.value,
            "knowledge_type_b": self.knowledge_type_b.value,
            "is_complementary": self.is_complementary,
            "requires_human_review": self.requires_human_review,
            "recommended_actions": self.recommended_actions,
        }


# =============================================================================
# Knowledge Patterns (Keywords and Regex)
# =============================================================================

# State-indicating keywords (Chinese + English)
STATE_KEYWORDS: Set[str] = {
    # Chinese
    "住在", "住址", "地址", "电话", "手机", "邮箱", "工作在", "现在是", 
    "目前是", "正在", "现居", "任职于", "就职于", "担任", "在做", "邮编",
    "在谷歌", "在微软", "在阿里", "在腾讯", "在百度", "在公司",  # Common tech companies
    # English
    "live in", "lives in", "living in", "address", "phone", "email",
    "work at", "works at", "working at", "currently", "now is", "located in",
    "employed at", "job is", "position is", "residing", "stay at",
}

# Static fact keywords
STATIC_KEYWORDS: Set[str] = {
    # Chinese
    "生日", "出生日期", "出生地", "籍贯", "故乡", "血型", "身份证",
    "国籍", "民族", "性别", "出生于", "生于",
    # English
    "birthday", "birth date", "birthdate", "born on", "born in",
    "birthplace", "place of birth", "nationality", "blood type",
    "native place", "hometown", "gender", "sex",
}

# Identity trait keywords
TRAIT_KEYWORDS: Set[str] = {
    # Chinese
    "是...的人", "性格", "特点是", "擅长", "不擅长", "善于", "不善于",
    "天生", "本性", "个性", "脾气", "气质", "为人", "性情",
    # English
    "i am", "personality", "good at", "bad at", "tends to be", "tend to be",
    "naturally", "by nature", "character", "temperament", "disposition",
    "trait", "inherently", "fundamentally",
}

# Trait words (adjectives that describe identity)
TRAIT_WORDS: Set[str] = {
    # Chinese (positive)
    "耐心", "高效", "细心", "认真", "勤奋", "创意", "逻辑",
    "理性", "感性", "开朗", "内向", "外向", "热情", "冷静",
    "严谨", "灵活", "坚持", "果断", "温和", "敏感", "乐观",
    "谨慎", "大胆", "独立", "合作", "专注", "好奇",
    # English
    "patient", "efficient", "careful", "diligent", "creative", "logical",
    "rational", "emotional", "outgoing", "introverted", "extroverted",
    "enthusiastic", "calm", "rigorous", "flexible", "persistent",
    "decisive", "gentle", "sensitive", "optimistic", "cautious",
    "bold", "independent", "collaborative", "focused", "curious",
}

# Identity value keywords
VALUE_KEYWORDS: Set[str] = {
    # Chinese
    "价值观", "相信", "坚信", "信仰", "原则", "底线", "准则",
    "重要的是", "最重要", "核心", "根本", "本质", "追求", "理想",
    # English
    "value", "believe", "belief", "principle", "bottom line", "standard",
    "important to me", "core", "fundamental", "essence", "pursue", "ideal",
    "integrity", "honesty", "fairness", "justice", "respect", "trust",
}

# Preference keywords
PREFERENCE_KEYWORDS: Set[str] = {
    # Chinese
    "喜欢", "不喜欢", "偏好", "习惯", "喜爱", "讨厌", "爱好",
    "热爱", "痴迷", "钟爱", "厌恶", "反感", "偏爱", "喜好",
    "最爱", "最喜欢", "最讨厌", "倾向于",
    # English
    "like", "likes", "prefer", "prefers", "prefer to", "enjoy", "enjoys",
    "hate", "hates", "love", "loves", "dislike", "dislikes", "fond of",
    "favorite", "favourite", "obsessed", "addicted to", "into",
}

# Behavior pattern keywords
BEHAVIOR_KEYWORDS: Set[str] = {
    # Chinese
    "通常", "一般", "总是", "经常", "每天", "每周", "习惯于",
    "倾向", "模式", "规律", "作息", "日常", "例行", "定期",
    # English
    "usually", "typically", "always", "often", "every day", "every week",
    "tend to", "pattern", "routine", "habit", "regularly", "daily",
    "weekly", "monthly", "schedule",
}

# =============================================================================
# Complementary Trait Pairs
# =============================================================================

# These pairs are NOT contradictory - they are different facets activated in different contexts
COMPLEMENTARY_TRAIT_PAIRS: List[Tuple[Set[str], Set[str]]] = [
    # Patience vs Efficiency - different situations require different approaches
    ({"耐心", "patient", "慢", "slow", "仔细", "careful"}, 
     {"高效", "efficient", "快", "fast", "迅速", "quick"}),
    
    # Introverted vs Social - different contexts, not contradiction
    ({"内向", "introverted", "安静", "quiet", "独处", "solitary"},
     {"社交", "social", "外向", "extroverted", "活跃", "active"}),
    
    # Rational vs Emotional - both can coexist
    ({"理性", "rational", "逻辑", "logical", "分析", "analytical"},
     {"感性", "emotional", "直觉", "intuitive", "感受", "feeling"}),
    
    # Rigorous vs Flexible - different needs in different situations
    ({"严谨", "rigorous", "精确", "precise", "严格", "strict"},
     {"灵活", "flexible", "变通", "adaptable", "弹性", "elastic"}),
    
    # Independent vs Collaborative - both are strengths
    ({"独立", "independent", "自主", "autonomous", "单独", "solo"},
     {"合作", "collaborative", "团队", "team", "协作", "cooperative"}),
    
    # Detail-oriented vs Big-picture
    ({"细节", "detail", "细心", "meticulous", "微观", "micro"},
     {"全局", "big picture", "宏观", "macro", "整体", "holistic"}),
    
    # Cautious vs Bold
    ({"谨慎", "cautious", "保守", "conservative", "稳妥", "prudent"},
     {"大胆", "bold", "冒险", "adventurous", "激进", "aggressive"}),
    
    # Serious vs Humorous
    ({"严肃", "serious", "正经", "formal", "庄重", "solemn"},
     {"幽默", "humorous", "风趣", "witty", "轻松", "relaxed"}),
]

# Truly contradictory pairs - these cannot coexist for the same subject/context
CONTRADICTORY_PAIRS: List[Tuple[Set[str], Set[str]]] = [
    # Honest vs Dishonest - direct moral contradiction
    ({"诚实", "honest", "真诚", "sincere", "坦诚", "truthful"},
     {"说谎", "lying", "liar", "欺骗", "deceptive", "虚伪", "dishonest"}),
    
    # Helpful vs Harmful
    ({"帮助", "helpful", "有益", "beneficial", "有用", "useful"},
     {"伤害", "harmful", "有害", "detrimental", "破坏", "destructive"}),
    
    # Alive vs Dead (for living entities)
    ({"活着", "alive", "生存", "living", "存活", "surviving"},
     {"死亡", "dead", "去世", "deceased", "死去", "passed away"}),
    
    # Present vs Absent (same time)
    ({"在场", "present", "在", "here", "出席", "attending"},
     {"缺席", "absent", "不在", "away", "离开", "left"}),
]


# =============================================================================
# Knowledge Classifier
# =============================================================================

class KnowledgeClassifier:
    """
    First-principles-based knowledge type classifier.
    
    Philosophy: Not all contradictions need elimination.
    - States update (住址变了)
    - Static facts can be corrected (生日错了)
    - Traits can coexist (耐心 AND 高效)
    - Values are foundational (诚实)
    - Preferences evolve (以前喜欢咖啡，现在喜欢茶)
    - Behaviors are patterns (习惯)
    
    The classifier uses:
    1. Keyword matching (high precision)
    2. Pattern matching (structural analysis)
    3. Semantic similarity (when embeddings provided)
    
    All decisions return probabilities, not hard thresholds.
    """
    
    def __init__(self, seed: int = 0):
        """
        Initialize the knowledge classifier.
        
        Args:
            seed: Random seed for reproducibility
        """
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Precompile regex patterns for efficiency
        self._compile_patterns()
        
        # Statistics for online learning (optional future extension)
        self._classification_stats: Dict[str, int] = {kt.value: 0 for kt in KnowledgeType}
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for knowledge type detection."""
        # Subject-predicate patterns for state detection
        self._state_patterns = [
            # Chinese: X住在Y, X的地址是Y
            re.compile(r"(.+?)(?:住在|住址是|地址是|电话是|邮箱是|工作在)(.+)", re.IGNORECASE),
            # English: X lives in Y, X's address is Y
            re.compile(r"(.+?)(?:lives? in|works? at|\'s address is|\'s phone is|\'s email is)(.+)", re.IGNORECASE),
            # Current state: X现在是Y, X目前是Y
            re.compile(r"(.+?)(?:现在是|目前是|currently is|now is)(.+)", re.IGNORECASE),
        ]
        
        # Static fact patterns
        self._static_patterns = [
            # Chinese: X的生日是Y, X出生于Y
            re.compile(r"(.+?)(?:的生日是|出生日期是|出生于|生于)(.+)", re.IGNORECASE),
            # English: X's birthday is Y, X was born on Y
            re.compile(r"(.+?)(?:\'s birthday is|was born on|born in|birthplace is)(.+)", re.IGNORECASE),
        ]
        
        # Trait patterns
        self._trait_patterns = [
            # Chinese: X是一个Y的人, X很Y
            re.compile(r"(.+?)(?:是一个|是个)(.+?)(?:的人)", re.IGNORECASE),
            re.compile(r"(.+?)(?:很|非常|特别|比较)(\w+)", re.IGNORECASE),
            # English: X is a Y person, X is very Y
            re.compile(r"(.+?)(?:is a|is an|am a|am an)(.+?)(?:person)", re.IGNORECASE),
            re.compile(r"(.+?)(?:is very|am very|is quite|am quite)(.+)", re.IGNORECASE),
            # I am X: 我是Y的人
            re.compile(r"(?:我|i)(?:是|am)(.+)", re.IGNORECASE),
        ]
        
        # Preference patterns
        self._preference_patterns = [
            # Chinese: X喜欢Y, X偏好Y
            re.compile(r"(.+?)(?:喜欢|偏好|热爱|讨厌|不喜欢)(.+)", re.IGNORECASE),
            # English: X likes Y, X prefers Y
            re.compile(r"(.+?)(?:likes?|prefers?|loves?|hates?|dislikes?)(.+)", re.IGNORECASE),
        ]
        
        # Behavior patterns
        self._behavior_patterns = [
            # Chinese: X通常Y, X习惯Y
            re.compile(r"(.+?)(?:通常|一般|总是|经常|习惯)(.+)", re.IGNORECASE),
            # English: X usually Y, X tends to Y
            re.compile(r"(.+?)(?:usually|typically|always|often|tends? to)(.+)", re.IGNORECASE),
        ]
    
    # =========================================================================
    # Main Classification Method
    # =========================================================================
    
    def classify(
        self, 
        text: str,
        embedding: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """
        Classify the knowledge type of a text.
        
        Uses multiple signals combined probabilistically:
        1. Keyword matching (strongest signal)
        2. Pattern matching (structural)
        3. Semantic features (when embedding provided)
        
        Args:
            text: The text to classify
            embedding: Optional embedding for semantic analysis
            context: Optional context dict with additional hints
            
        Returns:
            ClassificationResult with type, confidence, and extracted info
        """
        text_lower = text.lower()
        
        # Compute scores for each type
        scores: Dict[KnowledgeType, float] = {kt: 0.0 for kt in KnowledgeType}
        matched_patterns: Dict[KnowledgeType, List[str]] = {kt: [] for kt in KnowledgeType}
        
        # 1. Keyword matching
        keyword_results = self._match_keywords(text_lower)
        for kt, (score, patterns) in keyword_results.items():
            scores[kt] += score
            matched_patterns[kt].extend(patterns)
        
        # 2. Pattern matching
        pattern_results = self._match_patterns(text)
        subject, predicate = None, None
        for kt, (score, patterns, subj, pred) in pattern_results.items():
            scores[kt] += score
            matched_patterns[kt].extend(patterns)
            if subj and subject is None:
                subject = subj
            if pred and predicate is None:
                predicate = pred
        
        # 3. Trait word detection (special case for identity traits)
        trait_score = self._detect_trait_words(text_lower)
        if trait_score > 0:
            scores[KnowledgeType.IDENTITY_TRAIT] += trait_score
            matched_patterns[KnowledgeType.IDENTITY_TRAIT].append("trait_word_detected")
        
        # Normalize scores to probabilities using softmax with temperature
        total_score = sum(scores.values())
        if total_score < 0.1:
            # No strong signals - return UNKNOWN
            return ClassificationResult(
                knowledge_type=KnowledgeType.UNKNOWN,
                confidence=0.3,
                matched_patterns=[],
                subject=subject,
                predicate=predicate,
            )
        
        # Find best type
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        # Confidence based on score margin
        second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        margin = best_score - second_best
        confidence = min(1.0, 0.5 + margin * 0.5)  # Map margin to [0.5, 1.0]
        
        # Update statistics
        self._classification_stats[best_type.value] += 1
        
        return ClassificationResult(
            knowledge_type=best_type,
            confidence=confidence,
            matched_patterns=matched_patterns[best_type],
            subject=subject,
            predicate=predicate,
        )
    
    def _match_keywords(
        self, text_lower: str
    ) -> Dict[KnowledgeType, Tuple[float, List[str]]]:
        """
        Match keywords for each knowledge type.
        
        Returns dict mapping type to (score, matched_keywords).
        """
        results: Dict[KnowledgeType, Tuple[float, List[str]]] = {}
        
        # State keywords
        state_matches = [kw for kw in STATE_KEYWORDS if kw in text_lower]
        results[KnowledgeType.FACTUAL_STATE] = (len(state_matches) * 0.8, state_matches)
        
        # Static keywords
        static_matches = [kw for kw in STATIC_KEYWORDS if kw in text_lower]
        results[KnowledgeType.FACTUAL_STATIC] = (len(static_matches) * 0.9, static_matches)
        
        # Trait keywords
        trait_matches = [kw for kw in TRAIT_KEYWORDS if kw in text_lower]
        results[KnowledgeType.IDENTITY_TRAIT] = (len(trait_matches) * 0.7, trait_matches)
        
        # Value keywords
        value_matches = [kw for kw in VALUE_KEYWORDS if kw in text_lower]
        results[KnowledgeType.IDENTITY_VALUE] = (len(value_matches) * 0.85, value_matches)
        
        # Preference keywords
        pref_matches = [kw for kw in PREFERENCE_KEYWORDS if kw in text_lower]
        results[KnowledgeType.PREFERENCE] = (len(pref_matches) * 0.75, pref_matches)
        
        # Behavior keywords
        behavior_matches = [kw for kw in BEHAVIOR_KEYWORDS if kw in text_lower]
        results[KnowledgeType.BEHAVIOR_PATTERN] = (len(behavior_matches) * 0.7, behavior_matches)
        
        return results
    
    def _match_patterns(
        self, text: str
    ) -> Dict[KnowledgeType, Tuple[float, List[str], Optional[str], Optional[str]]]:
        """
        Match structural patterns for each knowledge type.
        
        Returns dict mapping type to (score, patterns, subject, predicate).
        """
        results: Dict[KnowledgeType, Tuple[float, List[str], Optional[str], Optional[str]]] = {}
        
        for kt in KnowledgeType:
            results[kt] = (0.0, [], None, None)
        
        # State patterns
        for pattern in self._state_patterns:
            match = pattern.search(text)
            if match:
                subject = match.group(1).strip() if len(match.groups()) > 0 else None
                predicate = match.group(2).strip() if len(match.groups()) > 1 else None
                results[KnowledgeType.FACTUAL_STATE] = (1.2, [pattern.pattern[:30]], subject, predicate)
                break
        
        # Static patterns
        for pattern in self._static_patterns:
            match = pattern.search(text)
            if match:
                subject = match.group(1).strip() if len(match.groups()) > 0 else None
                predicate = match.group(2).strip() if len(match.groups()) > 1 else None
                results[KnowledgeType.FACTUAL_STATIC] = (1.3, [pattern.pattern[:30]], subject, predicate)
                break
        
        # Trait patterns
        for pattern in self._trait_patterns:
            match = pattern.search(text)
            if match:
                subject = match.group(1).strip() if len(match.groups()) > 0 else None
                trait = match.group(2).strip() if len(match.groups()) > 1 else match.group(1).strip()
                results[KnowledgeType.IDENTITY_TRAIT] = (1.0, [pattern.pattern[:30]], subject, trait)
                break
        
        # Preference patterns
        for pattern in self._preference_patterns:
            match = pattern.search(text)
            if match:
                subject = match.group(1).strip() if len(match.groups()) > 0 else None
                predicate = match.group(2).strip() if len(match.groups()) > 1 else None
                results[KnowledgeType.PREFERENCE] = (1.1, [pattern.pattern[:30]], subject, predicate)
                break
        
        # Behavior patterns
        for pattern in self._behavior_patterns:
            match = pattern.search(text)
            if match:
                subject = match.group(1).strip() if len(match.groups()) > 0 else None
                predicate = match.group(2).strip() if len(match.groups()) > 1 else None
                results[KnowledgeType.BEHAVIOR_PATTERN] = (1.0, [pattern.pattern[:30]], subject, predicate)
                break
        
        return results
    
    def _detect_trait_words(self, text_lower: str) -> float:
        """Detect trait words that indicate identity traits."""
        matches = [tw for tw in TRAIT_WORDS if tw in text_lower]
        return len(matches) * 0.5
    
    # =========================================================================
    # Conflict Resolution
    # =========================================================================
    
    def resolve_conflict(
        self,
        type_a: KnowledgeType,
        type_b: KnowledgeType,
        time_relation: str,  # "sequential" | "concurrent" | "unknown"
        text_a: Optional[str] = None,
        text_b: Optional[str] = None,
        embedding_a: Optional[np.ndarray] = None,
        embedding_b: Optional[np.ndarray] = None,
    ) -> ConflictAnalysis:
        """
        Determine how to resolve a conflict between two pieces of knowledge.
        
        Core philosophy: Not all conflicts need elimination.
        - States: UPDATE (newer replaces older)
        - Static facts: CORRECT (one is wrong)
        - Traits: PRESERVE_BOTH (complementary facets)
        - Values: PRESERVE_BOTH (core identity)
        - Preferences: EVOLVE (track change)
        - Behaviors: EVOLVE (patterns change)
        
        Args:
            type_a: Knowledge type of first item
            type_b: Knowledge type of second item
            time_relation: Temporal relationship ("sequential", "concurrent", "unknown")
            text_a: Optional text of first item
            text_b: Optional text of second item
            embedding_a: Optional embedding of first item
            embedding_b: Optional embedding of second item
            
        Returns:
            ConflictAnalysis with resolution strategy and rationale
        """
        # Check if they're the same knowledge type
        same_type = type_a == type_b
        
        # Special case: Identity traits - check for complementary pairs
        if type_a == KnowledgeType.IDENTITY_TRAIT and type_b == KnowledgeType.IDENTITY_TRAIT:
            if text_a and text_b:
                is_complementary = self.are_complementary_traits(text_a, text_b)
                if is_complementary:
                    return ConflictAnalysis(
                        resolution=ConflictResolution.PRESERVE_BOTH,
                        rationale="这两个特质是互补的而非矛盾的——它们在不同情境下激活不同的面向",
                        confidence=0.85,
                        knowledge_type_a=type_a,
                        knowledge_type_b=type_b,
                        is_complementary=True,
                        recommended_actions=[
                            "保留两个特质",
                            "标记为情境依赖的特质",
                            "在检索时根据上下文激活",
                        ],
                    )
                
                # Check for true contradiction
                is_contradictory = self._are_truly_contradictory(text_a, text_b)
                if is_contradictory:
                    return ConflictAnalysis(
                        resolution=ConflictResolution.CORRECT if time_relation == "sequential" else ConflictResolution.UPDATE,
                        rationale="这两个特质存在真正的矛盾，需要解决",
                        confidence=0.75,
                        knowledge_type_a=type_a,
                        knowledge_type_b=type_b,
                        is_complementary=False,
                        requires_human_review=True,
                        recommended_actions=[
                            "标记为需要人工审核",
                            "保留更新的特质",
                            "记录变化历史",
                        ],
                    )
        
        # Resolution based on knowledge type
        if same_type:
            return self._resolve_same_type_conflict(type_a, time_relation, text_a, text_b)
        else:
            return self._resolve_different_type_conflict(type_a, type_b, time_relation)
    
    def _resolve_same_type_conflict(
        self,
        ktype: KnowledgeType,
        time_relation: str,
        text_a: Optional[str],
        text_b: Optional[str],
    ) -> ConflictAnalysis:
        """Resolve conflict when both items have the same knowledge type."""
        
        if ktype == KnowledgeType.FACTUAL_STATE:
            # States update: newer replaces older
            return ConflictAnalysis(
                resolution=ConflictResolution.UPDATE,
                rationale="状态性事实会随时间变化，保留最新状态",
                confidence=0.9,
                knowledge_type_a=ktype,
                knowledge_type_b=ktype,
                recommended_actions=[
                    "用新状态替换旧状态",
                    "保留旧状态作为历史记录（可选）",
                ],
            )
        
        elif ktype == KnowledgeType.FACTUAL_STATIC:
            # Static facts: one is wrong, need correction
            return ConflictAnalysis(
                resolution=ConflictResolution.CORRECT,
                rationale="静态事实不应该变化，旧的可能是错误的",
                confidence=0.85,
                knowledge_type_a=ktype,
                knowledge_type_b=ktype,
                requires_human_review=True,
                recommended_actions=[
                    "需要验证哪个是正确的",
                    "标记旧的为可能错误",
                    "考虑信息来源可靠性",
                ],
            )
        
        elif ktype == KnowledgeType.IDENTITY_TRAIT:
            # Traits: default to preserve both (unless explicitly contradictory)
            return ConflictAnalysis(
                resolution=ConflictResolution.PRESERVE_BOTH,
                rationale="身份特质可以共存，它们是不同情境下的不同面向",
                confidence=0.7,
                knowledge_type_a=ktype,
                knowledge_type_b=ktype,
                is_complementary=True,  # Assume complementary by default
                recommended_actions=[
                    "保留两个特质",
                    "分析它们是否在不同情境下有效",
                    "标记为多面性身份",
                ],
            )
        
        elif ktype == KnowledgeType.IDENTITY_VALUE:
            # Values: preserve both (core to identity)
            return ConflictAnalysis(
                resolution=ConflictResolution.PRESERVE_BOTH,
                rationale="价值观是身份的核心，不同价值观可以共存",
                confidence=0.85,
                knowledge_type_a=ktype,
                knowledge_type_b=ktype,
                recommended_actions=[
                    "保留所有价值观",
                    "分析它们之间的关系",
                    "识别潜在的价值观冲突",
                ],
            )
        
        elif ktype == KnowledgeType.PREFERENCE:
            # Preferences: evolve (track change timeline)
            return ConflictAnalysis(
                resolution=ConflictResolution.EVOLVE,
                rationale="偏好会随时间演化，保留变化轨迹",
                confidence=0.8,
                knowledge_type_a=ktype,
                knowledge_type_b=ktype,
                recommended_actions=[
                    "保留两个偏好作为时间线",
                    "标记时间戳",
                    "分析偏好演化趋势",
                ],
            )
        
        elif ktype == KnowledgeType.BEHAVIOR_PATTERN:
            # Behavior: evolve (patterns change)
            return ConflictAnalysis(
                resolution=ConflictResolution.EVOLVE,
                rationale="行为模式会随时间变化，记录变化有助于理解",
                confidence=0.75,
                knowledge_type_a=ktype,
                knowledge_type_b=ktype,
                recommended_actions=[
                    "保留两个行为模式",
                    "标记时间段",
                    "分析行为变化原因",
                ],
            )
        
        else:
            # Unknown: no action, needs more info
            return ConflictAnalysis(
                resolution=ConflictResolution.NO_ACTION,
                rationale="无法确定知识类型，需要更多信息",
                confidence=0.3,
                knowledge_type_a=ktype,
                knowledge_type_b=ktype,
                requires_human_review=True,
                recommended_actions=[
                    "收集更多上下文",
                    "人工审核",
                ],
            )
    
    def _resolve_different_type_conflict(
        self,
        type_a: KnowledgeType,
        type_b: KnowledgeType,
        time_relation: str,
    ) -> ConflictAnalysis:
        """Resolve conflict when items have different knowledge types."""
        
        # Priority: Traits and Values > State > Preference > Behavior
        # This reflects the importance hierarchy in identity
        
        priority = {
            KnowledgeType.IDENTITY_VALUE: 5,
            KnowledgeType.IDENTITY_TRAIT: 4,
            KnowledgeType.FACTUAL_STATIC: 3,
            KnowledgeType.FACTUAL_STATE: 2,
            KnowledgeType.PREFERENCE: 1,
            KnowledgeType.BEHAVIOR_PATTERN: 0,
            KnowledgeType.UNKNOWN: -1,
        }
        
        higher_type = type_a if priority.get(type_a, -1) >= priority.get(type_b, -1) else type_b
        
        # Cross-type conflicts usually should preserve both
        return ConflictAnalysis(
            resolution=ConflictResolution.PRESERVE_BOTH,
            rationale=f"不同类型的知识（{type_a.value} vs {type_b.value}）通常可以共存",
            confidence=0.65,
            knowledge_type_a=type_a,
            knowledge_type_b=type_b,
            recommended_actions=[
                "保留两条知识",
                f"优先参考{higher_type.value}类型",
                "分析它们之间的关系",
            ],
        )
    
    # =========================================================================
    # Complementary Trait Detection
    # =========================================================================
    
    def are_complementary_traits(
        self, 
        text_a: str, 
        text_b: str,
        embedding_a: Optional[np.ndarray] = None,
        embedding_b: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Check if two traits are complementary rather than contradictory.
        
        Complementary traits:
        - "I am patient" and "I am efficient" → complementary (different contexts)
        - "I am introverted" and "I am social" → complementary (different situations)
        
        Contradictory traits:
        - "I am honest" and "I am a liar" → contradictory (cannot coexist)
        
        Args:
            text_a: First trait text
            text_b: Second trait text
            embedding_a: Optional embedding for semantic analysis
            embedding_b: Optional embedding for semantic analysis
            
        Returns:
            True if traits are complementary, False if contradictory
        """
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()
        
        # Check against known complementary pairs
        for set_a, set_b in COMPLEMENTARY_TRAIT_PAIRS:
            a_in_set_a = any(w in text_a_lower for w in set_a)
            a_in_set_b = any(w in text_a_lower for w in set_b)
            b_in_set_a = any(w in text_b_lower for w in set_a)
            b_in_set_b = any(w in text_b_lower for w in set_b)
            
            # Cross-match indicates complementary
            if (a_in_set_a and b_in_set_b) or (a_in_set_b and b_in_set_a):
                return True
        
        # Check against known contradictory pairs
        if self._are_truly_contradictory(text_a, text_b):
            return False
        
        # Use embedding similarity if available
        if embedding_a is not None and embedding_b is not None:
            sim = cosine_sim(embedding_a, embedding_b)
            # Very low or negative similarity suggests contradiction
            # Moderate similarity suggests complementary
            if sim < -0.3:
                return False  # Likely contradictory
            elif 0.2 < sim < 0.7:
                return True  # Likely complementary (different but not opposite)
        
        # Default: assume complementary for safety
        # (Better to preserve than lose information)
        return True
    
    def _are_truly_contradictory(self, text_a: str, text_b: str) -> bool:
        """Check if two texts are truly contradictory (cannot coexist)."""
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()
        
        for set_a, set_b in CONTRADICTORY_PAIRS:
            a_in_set_a = any(w in text_a_lower for w in set_a)
            a_in_set_b = any(w in text_a_lower for w in set_b)
            b_in_set_a = any(w in text_b_lower for w in set_a)
            b_in_set_b = any(w in text_b_lower for w in set_b)
            
            # Cross-match indicates true contradiction
            if (a_in_set_a and b_in_set_b) or (a_in_set_b and b_in_set_a):
                return True
        
        return False
    
    # =========================================================================
    # Batch Classification
    # =========================================================================
    
    def classify_batch(
        self, 
        texts: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.
        
        Args:
            texts: List of texts to classify
            embeddings: Optional list of embeddings
            
        Returns:
            List of ClassificationResult
        """
        results = []
        for i, text in enumerate(texts):
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            results.append(self.classify(text, embedding=emb))
        return results
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize classifier state."""
        return {
            "seed": self._seed,
            "classification_stats": self._classification_stats,
        }
    
    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "KnowledgeClassifier":
        """Restore classifier from state dict."""
        obj = cls(seed=d.get("seed", 0))
        obj._classification_stats = d.get("classification_stats", obj._classification_stats)
        return obj
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        total = sum(self._classification_stats.values())
        return {
            "total_classifications": total,
            "by_type": self._classification_stats.copy(),
            "distribution": {
                k: v / total if total > 0 else 0
                for k, v in self._classification_stats.items()
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def classify_knowledge(text: str, seed: int = 0) -> ClassificationResult:
    """Convenience function for one-off classification."""
    classifier = KnowledgeClassifier(seed=seed)
    return classifier.classify(text)


def resolve_knowledge_conflict(
    text_a: str,
    text_b: str,
    time_relation: str = "sequential",
    seed: int = 0,
) -> ConflictAnalysis:
    """Convenience function for resolving conflicts between two texts."""
    classifier = KnowledgeClassifier(seed=seed)
    type_a = classifier.classify(text_a)
    type_b = classifier.classify(text_b)
    return classifier.resolve_conflict(
        type_a.knowledge_type,
        type_b.knowledge_type,
        time_relation,
        text_a,
        text_b,
    )
