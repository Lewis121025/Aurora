"""
AURORA 连贯性守护模块
================================

在没有硬编码规则的情况下维护叙述和事实连贯性。
所有一致性判断都是概率性的。

关键组件：
- ContradictionDetector：概率性矛盾检测
- CoherenceScorer：计算全局连贯性分数
- ConflictResolver：自动冲突解决策略
- BeliefNetwork：概率性信念传播
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import math
import numpy as np
import networkx as nx

from aurora.core.graph.memory_graph import MemoryGraph
from aurora.core.knowledge_classifier import (
    KnowledgeClassifier,
    KnowledgeType,
    ConflictResolution as KnowledgeConflictResolution,
    ConflictAnalysis,
)
from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.core.types import MemoryElement
from aurora.core.components.metric import LowRankMetric
from aurora.core.causal import CausalEdgeBelief, CausalMemoryGraph
from aurora.core.tension import TensionManager, Tension, TensionType, TensionResolution
from aurora.core.constants import (
    OPPOSITION_SCORE_THRESHOLD,
    HIGH_SIMILARITY_THRESHOLD,
    ANTI_CORRELATION_THRESHOLD,
    UNFINISHED_STORY_HOURS,
    MAX_COHERENCE_PAIRS,
    BELIEF_PROPAGATION_ITERATIONS,
    COHERENCE_WEIGHTS,
)
from aurora.utils.math_utils import l2_normalize, cosine_sim, sigmoid, softmax
from aurora.utils.time_utils import now_ts
from aurora.utils.embedding_utils import get_embedding_from_object


# -----------------------------------------------------------------------------
# 数据结构
# -----------------------------------------------------------------------------

class ConflictType(Enum):
    """连贯性冲突的类型"""
    FACTUAL = "factual"           # 矛盾的事实
    TEMPORAL = "temporal"         # 时间线不一致
    CAUSAL = "causal"             # 因果循环或悖论
    THEMATIC = "thematic"         # 冲突的主题
    SELF_NARRATIVE = "self"       # 自我叙述不一致


@dataclass
class Conflict:
    """检测到的连贯性冲突"""
    type: ConflictType
    node_a: str
    node_b: str

    # 概率性严重程度（不是硬阈值）
    severity: float  # 0-1，越高越严重
    confidence: float  # 0-1，对此检测的置信度

    description: str
    evidence: List[str] = field(default_factory=list)

    # 潜在的解决方案
    resolutions: List['Resolution'] = field(default_factory=list)


@dataclass
class Resolution:
    """冲突的建议解决方案"""
    strategy: str  # "weaken", "condition", "merge", "remove"
    target_node: str
    action_description: str

    # 预期结果
    expected_coherence_gain: float
    cost: float  # 信息丢失或复杂性

    # 此解决方案适用的条件
    condition: Optional[str] = None


@dataclass
class CoherenceReport:
    """完整的连贯性分析报告"""
    overall_score: float  # 0-1，越高越连贯

    conflicts: List[Conflict]
    unfinished_stories: List[str]
    orphan_plots: List[str]

    # 按类型的分数
    factual_coherence: float
    temporal_coherence: float
    causal_coherence: float
    thematic_coherence: float

    recommended_actions: List[Resolution]


# -----------------------------------------------------------------------------
# 信念网络（概率性连贯性）
# -----------------------------------------------------------------------------

class BeliefNetwork:
    """
    用于连贯性推理的概率性信念网络。

    每个节点都有一个信念状态（概率分布）。
    当连接的信念不兼容时会产生冲突。
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.beliefs: Dict[str, BeliefState] = {}
    
    def add_belief(
        self,
        node_id: str,
        prior: float,
        evidence_strength: float = 1.0,
    ) -> None:
        """添加信念节点"""
        self.graph.add_node(node_id)
        self.beliefs[node_id] = BeliefState(
            prior=prior,
            evidence_strength=evidence_strength,
        )

    def add_dependency(
        self,
        from_id: str,
        to_id: str,
        dependency_type: str,  # "supports", "contradicts"
        strength: float,
    ) -> None:
        """添加信念之间的依赖关系"""
        self.graph.add_edge(
            from_id, to_id,
            type=dependency_type,
            strength=strength,
        )
    
    def propagate_beliefs(self, iterations: int = BELIEF_PROPAGATION_ITERATIONS) -> Dict[str, float]:
        """
        通过网络传播信念。
        返回最终的信念概率。
        """
        probabilities = {
            node_id: state.prior
            for node_id, state in self.beliefs.items()
        }

        for _ in range(iterations):
            new_probs = {}

            for node_id in self.graph.nodes():
                if node_id not in self.beliefs:
                    continue

                base_prob = self.beliefs[node_id].prior
                evidence = self.beliefs[node_id].evidence_strength

                # 聚合来自邻居的影响
                support = 0.0
                contradiction = 0.0

                for pred in self.graph.predecessors(node_id):
                    edge = self.graph.edges[pred, node_id]
                    pred_prob = probabilities.get(pred, 0.5)

                    if edge['type'] == 'supports':
                        support += edge['strength'] * pred_prob
                    elif edge['type'] == 'contradicts':
                        contradiction += edge['strength'] * pred_prob

                # 更新概率
                influence = support - contradiction
                updated = sigmoid(
                    math.log(base_prob / (1 - base_prob + 1e-9) + 1e-9) +
                    evidence * influence
                )

                new_probs[node_id] = updated

            probabilities = new_probs

        return probabilities


@dataclass
class BeliefState:
    """网络中信念的状态"""
    prior: float
    evidence_strength: float
    last_updated: float = field(default_factory=now_ts)


# -----------------------------------------------------------------------------
# 矛盾检测器
# -----------------------------------------------------------------------------

class ContradictionDetector:
    """
    检测内存元素之间的矛盾。

    使用多个信号概率性地组合：
    - 语义对立
    - 时间不可能性
    - 因果不一致
    - 声明冲突
    """
    
    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)

        # 对立模式（从数据学习，不是硬编码）
        self.opposition_patterns: List[Tuple[np.ndarray, np.ndarray]] = []

    def detect_contradiction(
        self,
        a: MemoryElement,
        b: MemoryElement,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """
        检测 a 和 b 是否相互矛盾。

        返回：
            (概率，解释)
        """
        log_odds_contradiction = 0.0
        explanations = []

        # 获取嵌入
        emb_a = self._get_embedding(a)
        emb_b = self._get_embedding(b)

        if emb_a is None or emb_b is None:
            return 0.0, "Cannot compare: missing embeddings"

        # 1. 语义对立检测
        opposition_score = self._semantic_opposition_score(emb_a, emb_b)
        if opposition_score > OPPOSITION_SCORE_THRESHOLD:
            log_odds_contradiction += opposition_score * 2
            explanations.append(f"Semantic opposition ({opposition_score:.2f})")

        # 2. 极性反转（用于声明/主题）
        polarity_conflict = self._check_polarity_conflict(a, b)
        if polarity_conflict > 0:
            log_odds_contradiction += polarity_conflict
            explanations.append("Polarity conflict detected")

        # 3. 时间不可能性
        temporal_conflict = self._check_temporal_conflict(a, b)
        if temporal_conflict > 0:
            log_odds_contradiction += temporal_conflict
            explanations.append("Temporal inconsistency")

        # 4. 行为者矛盾（同一行为者，不兼容的状态）
        actor_conflict = self._check_actor_conflict(a, b)
        if actor_conflict > 0:
            log_odds_contradiction += actor_conflict
            explanations.append("Actor state conflict")

        # 转换为概率
        p_contradiction = sigmoid(log_odds_contradiction)
        explanation = "; ".join(explanations) if explanations else "No contradiction detected"

        return p_contradiction, explanation
    
    def _get_embedding(self, obj: MemoryElement) -> Optional[np.ndarray]:
        """从各种对象类型提取嵌入。"""
        return get_embedding_from_object(obj)

    def _semantic_opposition_score(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray,
    ) -> float:
        """
        检测语义对立。

        对立不仅仅是低相似性 - 它是主动矛盾。
        我们在某些子空间中寻找指向"相反"方向的向量。
        """
        # 基本相似性
        sim = cosine_sim(emb_a, emb_b)

        # 如果高度相似，则不矛盾
        if sim > HIGH_SIMILARITY_THRESHOLD:
            return 0.0

        # 如果反相关（指向相反），更强的矛盾信号
        if sim < ANTI_CORRELATION_THRESHOLD:
            return abs(sim)

        # 检查学习的对立模式
        for pos_pattern, neg_pattern in self.opposition_patterns:
            proj_a = np.dot(emb_a, pos_pattern)
            proj_b = np.dot(emb_b, neg_pattern)

            if proj_a > 0.5 and proj_b > 0.5:
                return 0.7  # 匹配对立模式

        # 中等相似性但不同的簇
        if 0.2 < sim < 0.5:
            return 0.3 * (0.5 - sim)

        return 0.0
    
    def _check_polarity_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        """检查声明/主题中的极性冲突"""
        # 提取极性（如果可用）
        polarity_a = getattr(a, 'polarity', None) or getattr(a, 'emotion_valence', 0)
        polarity_b = getattr(b, 'polarity', None) or getattr(b, 'emotion_valence', 0)

        if isinstance(polarity_a, str):
            polarity_a = 1.0 if polarity_a == 'positive' else -1.0
        if isinstance(polarity_b, str):
            polarity_b = 1.0 if polarity_b == 'positive' else -1.0

        # 检查主语/谓语匹配且极性相反
        subject_a = getattr(a, 'subject', '') or ''
        subject_b = getattr(b, 'subject', '') or ''
        predicate_a = getattr(a, 'predicate', '') or ''
        predicate_b = getattr(b, 'predicate', '') or ''

        if subject_a and subject_b:
            # 相同主语，相同谓语，相反极性
            if subject_a.lower() == subject_b.lower():
                if predicate_a.lower() == predicate_b.lower():
                    if polarity_a * polarity_b < 0:
                        return 1.5

        return 0.0
    
    def _check_temporal_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        """检查时间不可能性"""
        ts_a = getattr(a, 'ts', None) or getattr(a, 'timestamp', None)
        ts_b = getattr(b, 'ts', None) or getattr(b, 'timestamp', None)

        if ts_a is None or ts_b is None:
            return 0.0

        # 检查关于同一时间段的声明是否存在冲突
        # 这需要更复杂的时间推理
        # 目前返回 0
        return 0.0

    def _check_actor_conflict(self, a: MemoryElement, b: MemoryElement) -> float:
        """检查同一行为者是否具有不兼容的状态"""
        actors_a = set(getattr(a, 'actors', []) or [])
        actors_b = set(getattr(b, 'actors', []) or [])

        shared = actors_a & actors_b
        if not shared:
            return 0.0

        # 如果相同行为者且嵌入差异很大，可能存在冲突
        emb_a = self._get_embedding(a)
        emb_b = self._get_embedding(b)

        if emb_a is not None and emb_b is not None:
            sim = cosine_sim(emb_a, emb_b)
            if sim < 0.3:
                return 0.5 * len(shared) * (0.3 - sim)

        return 0.0

    def learn_opposition_pattern(
        self,
        positive_examples: List[np.ndarray],
        negative_examples: List[np.ndarray],
    ) -> None:
        """从示例学习对立模式"""
        if not positive_examples or not negative_examples:
            return

        pos_mean = np.mean(positive_examples, axis=0)
        neg_mean = np.mean(negative_examples, axis=0)

        pos_pattern = l2_normalize(pos_mean)
        neg_pattern = l2_normalize(neg_mean)

        self.opposition_patterns.append((pos_pattern, neg_pattern))


# -----------------------------------------------------------------------------
# 连贯性评分器
# -----------------------------------------------------------------------------

class CoherenceScorer:
    """
    计算内存系统的整体连贯性分数。

    连贯性在多个维度上测量：
    - 事实一致性
    - 时间一致性
    - 因果一致性
    - 主题一致性
    """

    def __init__(
        self,
        metric: LowRankMetric,
        detector: ContradictionDetector,
        seed: int = 0,
    ):
        self.metric = metric
        self.detector = detector
        self.rng = np.random.default_rng(seed)

    def compute_coherence(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> CoherenceReport:
        """计算完整的连贯性报告"""
        conflicts = []

        # 1. 检查事实连贯性（情节级矛盾）
        factual_conflicts, factual_score = self._check_factual_coherence(plots)
        conflicts.extend(factual_conflicts)

        # 2. 检查时间连贯性
        temporal_conflicts, temporal_score = self._check_temporal_coherence(plots, stories)
        conflicts.extend(temporal_conflicts)

        # 3. 检查因果连贯性
        if causal_beliefs:
            causal_conflicts, causal_score = self._check_causal_coherence(causal_beliefs)
            conflicts.extend(causal_conflicts)
        else:
            causal_score = 1.0

        # 4. 检查主题连贯性
        thematic_conflicts, thematic_score = self._check_thematic_coherence(themes)
        conflicts.extend(thematic_conflicts)

        # 5. 查找未完成的故事
        unfinished = [
            s.id for s in stories.values()
            if s.status == 'developing' and
            (now_ts() - s.updated_ts) > UNFINISHED_STORY_HOURS * 3600
        ]

        # 6. 查找孤立的情节
        orphans = [
            p.id for p in plots.values()
            if p.story_id is None and p.status == 'active'
        ]

        # 计算整体分数（加权几何平均）
        weights = [
            COHERENCE_WEIGHTS["factual"],
            COHERENCE_WEIGHTS["temporal"],
            COHERENCE_WEIGHTS["causal"],
            COHERENCE_WEIGHTS["thematic"],
        ]
        scores = [factual_score, temporal_score, causal_score, thematic_score]

        log_score = sum(w * math.log(s + 1e-9) for w, s in zip(weights, scores))
        overall_score = math.exp(log_score)

        # 生成建议的行动
        recommendations = self._generate_recommendations(conflicts)

        return CoherenceReport(
            overall_score=overall_score,
            conflicts=conflicts,
            unfinished_stories=unfinished,
            orphan_plots=orphans,
            factual_coherence=factual_score,
            temporal_coherence=temporal_score,
            causal_coherence=causal_score,
            thematic_coherence=thematic_score,
            recommended_actions=recommendations,
        )
    
    def _check_factual_coherence(
        self,
        plots: Dict[str, Plot],
    ) -> Tuple[List[Conflict], float]:
        """检查情节之间的事实矛盾"""
        conflicts = []
        total_pairs = 0
        contradiction_sum = 0.0

        plot_list = list(plots.values())

        # 为了效率对对进行采样
        max_pairs = min(MAX_COHERENCE_PAIRS, len(plot_list) * (len(plot_list) - 1) // 2)

        for i, p1 in enumerate(plot_list):
            for p2 in plot_list[i+1:]:
                if total_pairs >= max_pairs:
                    break

                total_pairs += 1
                prob, explanation = self.detector.detect_contradiction(p1, p2)
                contradiction_sum += prob

                if prob > 0.6:  # 报告的软阈值
                    conflicts.append(Conflict(
                        type=ConflictType.FACTUAL,
                        node_a=p1.id,
                        node_b=p2.id,
                        severity=prob,
                        confidence=0.7,
                        description=explanation,
                    ))

        # 分数 = 1 - 平均矛盾概率
        avg_contradiction = contradiction_sum / max(total_pairs, 1)
        score = 1.0 - avg_contradiction

        return conflicts, score

    def _check_temporal_coherence(
        self,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
    ) -> Tuple[List[Conflict], float]:
        """检查故事内的时间一致性"""
        conflicts = []
        inconsistency_count = 0
        total_checks = 0

        for story in stories.values():
            if len(story.plot_ids) < 2:
                continue

            # 获取故事中的情节
            story_plots = [plots[pid] for pid in story.plot_ids if pid in plots]
            story_plots.sort(key=lambda p: p.ts)

            # 检查序列是否合理
            for i in range(len(story_plots) - 1):
                total_checks += 1

                p1, p2 = story_plots[i], story_plots[i+1]

                # 检查不合理的时间间隙
                gap = p2.ts - p1.ts
                if gap < 0:
                    # 时间旅行！明确的不一致
                    inconsistency_count += 1
                    conflicts.append(Conflict(
                        type=ConflictType.TEMPORAL,
                        node_a=p1.id,
                        node_b=p2.id,
                        severity=1.0,
                        confidence=1.0,
                        description="Temporal ordering violation",
                    ))

        score = 1.0 - (inconsistency_count / max(total_checks, 1))
        return conflicts, score
    
    def _check_causal_coherence(
        self,
        causal_beliefs: Dict[Tuple[str, str], CausalEdgeBelief],
    ) -> Tuple[List[Conflict], float]:
        """检查因果图中的循环和悖论"""
        conflicts = []

        # 构建因果有向无环图（DAG）
        dag = nx.DiGraph()
        for (src, tgt), belief in causal_beliefs.items():
            if belief.direction_belief() > 0.5:
                dag.add_edge(src, tgt, weight=belief.effective_causal_weight())

        # 检查循环
        cycles = list(nx.simple_cycles(dag))

        for cycle in cycles[:10]:  # 限制报告的循环
            conflicts.append(Conflict(
                type=ConflictType.CAUSAL,
                node_a=cycle[0],
                node_b=cycle[-1],
                severity=0.8,
                confidence=1.0,
                description=f"Causal cycle detected: {' → '.join(cycle[:5])}...",
                evidence=cycle,
            ))

        # 基于循环数的分数
        cycle_penalty = len(cycles) * 0.1
        score = max(0.0, 1.0 - cycle_penalty)

        return conflicts, score
    
    def _check_thematic_coherence(
        self,
        themes: Dict[str, Theme],
    ) -> Tuple[List[Conflict], float]:
        """检查冲突的主题"""
        conflicts = []
        conflict_count = 0
        total_pairs = 0

        theme_list = list(themes.values())

        for i, t1 in enumerate(theme_list):
            for t2 in theme_list[i+1:]:
                total_pairs += 1

                prob, explanation = self.detector.detect_contradiction(t1, t2)

                if prob > 0.5:
                    conflict_count += 1

                    # 检查它们是否共享支持的故事
                    shared_stories = set(t1.story_ids) & set(t2.story_ids)

                    if shared_stories:
                        severity = prob * 1.2  # 如果有共享证据，严重程度更高
                    else:
                        severity = prob * 0.8

                    conflicts.append(Conflict(
                        type=ConflictType.THEMATIC,
                        node_a=t1.id,
                        node_b=t2.id,
                        severity=min(severity, 1.0),
                        confidence=0.6,
                        description=f"Themes may conflict: {explanation}",
                        evidence=list(shared_stories),
                    ))

        score = 1.0 - (conflict_count / max(total_pairs, 1))
        return conflicts, score

    def _generate_recommendations(
        self,
        conflicts: List[Conflict],
    ) -> List[Resolution]:
        """为冲突生成解决方案建议"""
        recommendations = []

        # 按严重程度排序
        sorted_conflicts = sorted(conflicts, key=lambda c: c.severity, reverse=True)

        for conflict in sorted_conflicts[:5]:  # 前 5 个建议
            if conflict.type == ConflictType.FACTUAL:
                # 建议条件化或削弱
                recommendations.append(Resolution(
                    strategy="condition",
                    target_node=conflict.node_a,
                    action_description=f"Add condition to {conflict.node_a} to resolve conflict with {conflict.node_b}",
                    expected_coherence_gain=conflict.severity * 0.7,
                    cost=0.1,
                    condition="Different contexts may apply",
                ))

            elif conflict.type == ConflictType.CAUSAL:
                # 建议移除循环中最弱的边
                recommendations.append(Resolution(
                    strategy="remove",
                    target_node=conflict.node_a,
                    action_description=f"Remove weakest causal link in cycle involving {conflict.node_a}",
                    expected_coherence_gain=conflict.severity * 0.8,
                    cost=0.2,
                ))

            elif conflict.type == ConflictType.THEMATIC:
                # 建议合并为条件主题
                recommendations.append(Resolution(
                    strategy="merge",
                    target_node=conflict.node_a,
                    action_description=f"Merge {conflict.node_a} and {conflict.node_b} into conditional theme",
                    expected_coherence_gain=conflict.severity * 0.6,
                    cost=0.3,
                ))

        return recommendations


# -----------------------------------------------------------------------------
# 冲突解决器
# -----------------------------------------------------------------------------

class ConflictResolver:
    """
    自动解决连贯性冲突。

    策略：
    - Weaken：降低冲突元素的置信度
    - Condition：添加条件使两者都为真
    - Merge：合并为更一般的元素
    - Remove：归档低置信度元素
    """

    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.rng = np.random.default_rng(seed)

    def resolve(
        self,
        conflict: Conflict,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> bool:
        """
        尝试解决冲突。
        如果应用了解决方案，返回 True。
        """
        if not conflict.resolutions:
            return False

        # 选择最佳解决方案（成本最低且满足收益阈值）
        best = min(
            conflict.resolutions,
            key=lambda r: r.cost - r.expected_coherence_gain
        )

        if best.expected_coherence_gain < 0.1:
            return False  # 不值得

        return self._apply_resolution(best, plots, stories, themes)

    def _apply_resolution(
        self,
        resolution: Resolution,
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> bool:
        """应用特定的解决策略"""

        if resolution.strategy == "weaken":
            # 降低置信度/证据
            if resolution.target_node in themes:
                theme = themes[resolution.target_node]
                theme.b += 1.0  # 增加负证据
                return True

        elif resolution.strategy == "condition":
            # 添加条件注释
            if resolution.target_node in themes:
                theme = themes[resolution.target_node]
                theme.description += f" (Condition: {resolution.condition})"
                return True

        elif resolution.strategy == "remove":
            # 标记为归档（软删除）
            if resolution.target_node in plots:
                plots[resolution.target_node].status = "archived"
                return True

        return False


# -----------------------------------------------------------------------------
# 连贯性守护者（主接口）
# -----------------------------------------------------------------------------

class CoherenceGuardian:
    """
    连贯性维护的主接口，具有功能性矛盾管理。

    关键哲学变化：
    - 并非所有矛盾都需要解决
    - 某些矛盾提供灵活性（适应性）
    - 某些矛盾表示增长（发展性）
    - 只有阻止行动或威胁身份的矛盾必须解决

    集成 TensionManager 进行智能矛盾处理。
    """

    def __init__(self, metric: LowRankMetric, seed: int = 0):
        self.metric = metric
        self.detector = ContradictionDetector(metric, seed)
        self.scorer = CoherenceScorer(metric, self.detector, seed)
        self.resolver = ConflictResolver(metric, seed)
        self.belief_network = BeliefNetwork()

        # 用于功能性矛盾管理的 TensionManager
        self.tension_manager = TensionManager(seed=seed)

        # 用于智能冲突解决的 KnowledgeClassifier
        # 并非所有矛盾都需要消除 - 某些应该保留
        self.knowledge_classifier = KnowledgeClassifier(seed=seed)

    def full_check(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> CoherenceReport:
        """运行完整的连贯性检查"""
        return self.scorer.compute_coherence(
            graph, plots, stories, themes, causal_beliefs
        )

    def full_check_with_tension_analysis(
        self,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> Tuple[CoherenceReport, Dict[str, Any]]:
        """
        运行完整的连贯性检查并进行张力分析。

        返回：
            (CoherenceReport, tension_analysis)

        tension_analysis 包括：
        - conflicts_to_resolve：必须解决的冲突
        - conflicts_to_preserve：提供灵活性的冲突
        - conflicts_to_accept：表示增长的冲突
        """
        report = self.scorer.compute_coherence(
            graph, plots, stories, themes, causal_beliefs
        )

        # 通过张力视角分析每个冲突
        conflicts_to_resolve = []
        conflicts_to_preserve = []
        conflicts_to_accept = []
        conflicts_to_defer = []

        for conflict in report.conflicts:
            # 将 Conflict 转换为 Tension 进行分析
            tension = self._conflict_to_tension(conflict, plots, stories, themes)
            if tension is None:
                continue

            # 分类并决定做什么
            tension_type = self.tension_manager.classify_tension(tension)

            if tension_type in [TensionType.ACTION_BLOCKING, TensionType.IDENTITY_THREATENING]:
                conflicts_to_resolve.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": f"必须解决：{tension_type.value}"
                })
            elif tension_type == TensionType.ADAPTIVE:
                conflicts_to_preserve.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": "保留：提供灵活性的适应性矛盾"
                })
            elif tension_type == TensionType.DEVELOPMENTAL:
                conflicts_to_accept.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": "接受：成长的标志"
                })
            else:
                conflicts_to_defer.append({
                    "conflict": conflict,
                    "tension": tension,
                    "reason": "需要更多信息"
                })

        tension_analysis = {
            "conflicts_to_resolve": conflicts_to_resolve,
            "conflicts_to_preserve": conflicts_to_preserve,
            "conflicts_to_accept": conflicts_to_accept,
            "conflicts_to_defer": conflicts_to_defer,
            "summary": {
                "total": len(report.conflicts),
                "to_resolve": len(conflicts_to_resolve),
                "to_preserve": len(conflicts_to_preserve),
                "to_accept": len(conflicts_to_accept),
                "to_defer": len(conflicts_to_defer),
            }
        }

        return report, tension_analysis

    def _conflict_to_tension(
        self,
        conflict: Conflict,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> Optional[Tension]:
        """将 Conflict 转换为 Tension 进行分析。"""
        # 获取涉及的元素
        element_a = self._get_element(conflict.node_a, plots, stories, themes)
        element_b = self._get_element(conflict.node_b, plots, stories, themes)

        if element_a is None or element_b is None:
            return None

        # 获取嵌入
        emb_a = self._get_embedding(element_a)
        emb_b = self._get_embedding(element_b)

        # 使用 TensionManager 检测并创建张力
        tension = self.tension_manager.detect_tension(
            {"id": conflict.node_a, "type": conflict.type.value, "text": conflict.description},
            {"id": conflict.node_b, "type": conflict.type.value, "text": ""},
            emb_a, emb_b
        )

        if tension is None:
            # 从冲突创建张力
            import uuid
            tension = Tension(
                id=str(uuid.uuid4()),
                element_a_id=conflict.node_a,
                element_a_type=conflict.type.value,
                element_b_id=conflict.node_b,
                element_b_type=conflict.type.value,
                description=conflict.description,
                severity=conflict.severity,
            )
            self.tension_manager.tensions[tension.id] = tension

        return tension

    def _get_element(
        self,
        node_id: str,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
    ) -> Optional[MemoryElement]:
        """按 ID 从任何集合中获取元素。"""
        if node_id in plots:
            return plots[node_id]
        if node_id in stories:
            return stories[node_id]
        if node_id in themes:
            return themes[node_id]
        return None

    def _get_embedding(self, element: MemoryElement) -> Optional[np.ndarray]:
        """从元素获取嵌入。"""
        return get_embedding_from_object(element)

    def auto_resolve(
        self,
        report: CoherenceReport,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        max_resolutions: int = 3,
    ) -> int:
        """
        自动解决顶部冲突。
        返回成功解决的数量。
        """
        resolved = 0

        # 按严重程度排序冲突
        sorted_conflicts = sorted(
            report.conflicts,
            key=lambda c: c.severity * c.confidence,
            reverse=True
        )

        for conflict in sorted_conflicts[:max_resolutions]:
            # 如果不存在，生成解决方案
            if not conflict.resolutions:
                conflict.resolutions = self._generate_resolutions(conflict)

            if self.resolver.resolve(conflict, graph, plots, stories, themes):
                resolved += 1

        return resolved

    def smart_resolve(
        self,
        report: CoherenceReport,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        max_resolutions: int = 3,
    ) -> Dict[str, Any]:
        """
        使用 TensionManager 进行智能冲突解决。

        仅解决需要解决的冲突。
        保留适应性和发展性张力。

        返回采取的行动摘要。
        """
        _, tension_analysis = self.full_check_with_tension_analysis(
            graph, plots, stories, themes
        )

        actions_taken = {
            "resolved": [],
            "preserved": [],
            "accepted": [],
        }

        # 仅解决阻止行动或威胁身份的冲突
        for item in tension_analysis["conflicts_to_resolve"][:max_resolutions]:
            conflict = item["conflict"]
            tension = item["tension"]

            # 生成并应用解决方案
            if not conflict.resolutions:
                conflict.resolutions = self._generate_resolutions(conflict)

            if self.resolver.resolve(conflict, graph, plots, stories, themes):
                resolution = self.tension_manager.handle_tension(tension)
                actions_taken["resolved"].append({
                    "conflict_id": f"{conflict.node_a}-{conflict.node_b}",
                    "description": conflict.description,
                    "action": resolution.action,
                    "rationale": resolution.rationale,
                })

        # 将适应性张力标记为已保留
        for item in tension_analysis["conflicts_to_preserve"]:
            tension = item["tension"]
            resolution = self.tension_manager.handle_tension(tension)
            actions_taken["preserved"].append({
                "conflict_description": item["conflict"].description,
                "reason": item["reason"],
                "action": resolution.action,
            })

        # 将发展性张力标记为已接受
        for item in tension_analysis["conflicts_to_accept"]:
            tension = item["tension"]
            resolution = self.tension_manager.handle_tension(tension)
            actions_taken["accepted"].append({
                "conflict_description": item["conflict"].description,
                "reason": item["reason"],
                "action": resolution.action,
            })

        return actions_taken

    def _generate_resolutions(self, conflict: Conflict) -> List[Resolution]:
        """为冲突生成解决方案选项"""
        resolutions = []

        if conflict.type == ConflictType.FACTUAL:
            resolutions.append(Resolution(
                strategy="weaken",
                target_node=conflict.node_b,  # 削弱较新的
                action_description=f"Reduce confidence in {conflict.node_b}",
                expected_coherence_gain=conflict.severity * 0.5,
                cost=0.2,
            ))
            resolutions.append(Resolution(
                strategy="condition",
                target_node=conflict.node_a,
                action_description=f"Add context condition to {conflict.node_a}",
                expected_coherence_gain=conflict.severity * 0.7,
                cost=0.1,
                condition="May depend on context",
            ))

        elif conflict.type == ConflictType.THEMATIC:
            resolutions.append(Resolution(
                strategy="merge",
                target_node=conflict.node_a,
                action_description=f"Merge conflicting themes",
                expected_coherence_gain=conflict.severity * 0.6,
                cost=0.3,
            ))

        elif conflict.type == ConflictType.CAUSAL:
            resolutions.append(Resolution(
                strategy="remove",
                target_node=conflict.node_b,
                action_description=f"Remove edge to break cycle",
                expected_coherence_gain=conflict.severity * 0.8,
                cost=0.15,
            ))

        return resolutions

    def update_belief_network(
        self,
        themes: Dict[str, Theme],
        causal_beliefs: Optional[Dict[Tuple[str, str], CausalEdgeBelief]] = None,
    ) -> Dict[str, float]:
        """
        使用当前主题更新信念网络并返回传播的信念。
        """
        self.belief_network = BeliefNetwork()

        # 添加主题信念
        for theme in themes.values():
            self.belief_network.add_belief(
                theme.id,
                prior=theme.confidence(),
                evidence_strength=len(theme.story_ids) * 0.1,
            )

        # 添加因果依赖
        if causal_beliefs:
            for (src, tgt), belief in causal_beliefs.items():
                if src in themes and tgt in themes:
                    dep_type = "supports" if belief.effective_causal_weight() > 0.5 else "contradicts"
                    self.belief_network.add_dependency(
                        src, tgt,
                        dependency_type=dep_type,
                        strength=belief.effective_causal_weight(),
                    )

        return self.belief_network.propagate_beliefs()

    def get_tension_summary(self) -> Dict[str, Any]:
        """获取所有跟踪张力的摘要。"""
        return self.tension_manager.get_tension_summary()

    def analyze_knowledge_conflict(
        self,
        plot_a: Plot,
        plot_b: Plot,
    ) -> ConflictAnalysis:
        """
        使用知识类型分类分析两个情节之间的冲突。

        第一原理的关键见解：
        - 并非所有矛盾都需要消除
        - 状态更新（较新的替换较旧的）
        - 静态事实需要更正（其中一个是错误的）
        - 特征可以共存（互补的方面）
        - 价值观应该被保留（核心身份）
        - 偏好演变（跟踪时间线）
        - 行为改变（模式是可塑的）

        参数：
            plot_a：冲突中的第一个情节
            plot_b：冲突中的第二个情节

        返回：
            ConflictAnalysis，包含解决策略和理由
        """
        # 获取或分类知识类型
        type_a = self._get_plot_knowledge_type(plot_a)
        type_b = self._get_plot_knowledge_type(plot_b)

        # 确定时间关系
        if abs(plot_a.ts - plot_b.ts) < 3600:  # 一小时内
            time_relation = "concurrent"
        elif plot_a.ts < plot_b.ts:
            time_relation = "sequential"
        else:
            time_relation = "sequential"  # b 在 a 之前

        # 使用知识分类器解决冲突
        return self.knowledge_classifier.resolve_conflict(
            type_a=type_a,
            type_b=type_b,
            time_relation=time_relation,
            text_a=plot_a.text,
            text_b=plot_b.text,
            embedding_a=plot_a.embedding,
            embedding_b=plot_b.embedding,
        )

    def _get_plot_knowledge_type(self, plot: Plot) -> KnowledgeType:
        """获取或分类情节的知识类型。"""
        if plot.knowledge_type is not None:
            # 使用现有分类
            type_map = {
                "factual_state": KnowledgeType.FACTUAL_STATE,
                "factual_static": KnowledgeType.FACTUAL_STATIC,
                "identity_trait": KnowledgeType.IDENTITY_TRAIT,
                "identity_value": KnowledgeType.IDENTITY_VALUE,
                "preference": KnowledgeType.PREFERENCE,
                "behavior": KnowledgeType.BEHAVIOR_PATTERN,
                "unknown": KnowledgeType.UNKNOWN,
            }
            return type_map.get(plot.knowledge_type, KnowledgeType.UNKNOWN)

        # 如果尚未完成，进行分类
        result = self.knowledge_classifier.classify(plot.text, embedding=plot.embedding)
        return result.knowledge_type

    def check_complementary_traits(
        self,
        plot_a: Plot,
        plot_b: Plot,
    ) -> bool:
        """
        检查两个情节是否包含互补（而非矛盾）的特征。

        示例：
        - "我很耐心" 和 "我很高效" → 互补（不同的背景）
        - "我诚实" 和 "我撒谎" → 矛盾（无法共存）

        这实现了叙述心理学原则，即健康的身份
        包含在不同情况下激活的功能性张力。

        参数：
            plot_a：第一个情节
            plot_b：第二个情节

        返回：
            如果特征互补则为 True，如果矛盾则为 False
        """
        return self.knowledge_classifier.are_complementary_traits(
            text_a=plot_a.text,
            text_b=plot_b.text,
            embedding_a=plot_a.embedding,
            embedding_b=plot_b.embedding,
        )

    def get_conflict_resolution_recommendation(
        self,
        conflict: Conflict,
        plots: Dict[str, Plot],
    ) -> Dict[str, Any]:
        """
        获取冲突的智能解决建议。

        使用知识类型分类来确定最佳策略：
        - UPDATE：用于状态变化（较新的信息替换较旧的）
        - CORRECT：用于静态事实更正（验证哪个是正确的）
        - PRESERVE_BOTH：用于互补的特征/价值观（保留两者）
        - EVOLVE：用于偏好/行为变化（跟踪时间线）

        参数：
            conflict：检测到的冲突
            plots：所有情节的字典

        返回：
            包含策略、理由和行动的建议字典
        """
        # 获取涉及的情节
        plot_a = plots.get(conflict.node_a)
        plot_b = plots.get(conflict.node_b)

        if plot_a is None or plot_b is None:
            return {
                "strategy": "unknown",
                "rationale": "Could not find one or both plots",
                "actions": ["Manual review required"],
                "confidence": 0.0,
            }

        # 使用知识分类器进行分析
        analysis = self.analyze_knowledge_conflict(plot_a, plot_b)

        return {
            "strategy": analysis.resolution.value,
            "rationale": analysis.rationale,
            "actions": analysis.recommended_actions,
            "confidence": analysis.confidence,
            "is_complementary": analysis.is_complementary,
            "requires_human_review": analysis.requires_human_review,
            "knowledge_type_a": analysis.knowledge_type_a.value,
            "knowledge_type_b": analysis.knowledge_type_b.value,
        }
