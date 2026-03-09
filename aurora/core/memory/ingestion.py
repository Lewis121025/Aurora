"""
AURORA 摄入模块
================

封装 plot 摄入、批量摄入、冲突处理与存储编织。
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aurora.core.coherence import Conflict, ConflictType
from aurora.core.config.coherence import (
    CONCURRENT_TIME_THRESHOLD,
    CONFLICT_CHECK_K,
    CONFLICT_CHECK_SIMILARITY_THRESHOLD,
    CONFLICT_PROBABILITY_THRESHOLD,
    MAX_CONFLICTS_PER_INGEST,
)
from aurora.core.config.identity import IDENTITY_RELEVANCE_WEIGHT, VOI_DECISION_WEIGHT
from aurora.core.config.knowledge import (
    KNOWLEDGE_TYPE_WEIGHT_BEHAVIOR,
    KNOWLEDGE_TYPE_WEIGHT_PREFERENCE,
    KNOWLEDGE_TYPE_WEIGHT_STATE,
    KNOWLEDGE_TYPE_WEIGHT_STATIC,
    KNOWLEDGE_TYPE_WEIGHT_TRAIT,
    KNOWLEDGE_TYPE_WEIGHT_VALUE,
    NUMERIC_CHANGE_INDICATORS,
    REINFORCEMENT_TIME_WINDOW,
    UPDATE_HIGH_SIMILARITY_THRESHOLD,
    UPDATE_KEYWORDS,
    UPDATE_MODERATE_SIMILARITY_THRESHOLD,
    UPDATE_TIME_GAP_THRESHOLD,
)
from aurora.core.config.numeric import (
    EPSILON_PRIOR,
    EVENT_SUMMARY_MAX_LENGTH,
    TEXT_LENGTH_NORMALIZATION,
    TRUST_BASE,
)
from aurora.core.config.retrieval import SEMANTIC_NEIGHBORS_K
from aurora.core.config.storage import COLD_START_FORCE_STORE_COUNT, MIN_STORE_PROB
from aurora.core.knowledge import ConflictAnalysis, ConflictResolution
from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.exceptions import ValidationError
from aurora.utils.id_utils import det_id
from aurora.utils.math_utils import cosine_sim, l2_normalize, sigmoid
from aurora.utils.time_utils import now_ts

logger = logging.getLogger(__name__)

INTERACTION_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "text": ("text", "文本"),
    "actors": ("actors", "参与者"),
    "context_text": ("context_text", "上下文", "context"),
    "event_id": ("event_id", "事件ID", "事件id"),
    "date": ("date", "日期"),
}


class IngestionMixin:
    """提供 ingest、批量摄入、冲突处理与图编织能力。"""

    def _update_centroid_online(
        self, current: Optional[np.ndarray], new_emb: np.ndarray, count: int
    ) -> np.ndarray:
        """使用在线平均算法更新质心/原型。"""
        if current is None:
            return new_emb.copy()
        return l2_normalize(current * ((count - 1) / count) + new_emb / count)

    def _create_bidirectional_edge(
        self, from_id: str, to_id: str, forward_type: str, backward_type: str
    ) -> None:
        """在内存图中创建双向边。"""
        self.graph.ensure_edge(from_id, to_id, forward_type)
        self.graph.ensure_edge(to_id, from_id, backward_type)

    def _compute_redundancy(
        self, emb: np.ndarray, text: str, ts: float
    ) -> Tuple[float, str, Optional[str]]:
        """计算与现有记忆的冗余度，区分更新和冗余。"""
        if self.benchmark_mode:
            return 0.0, "novel", None

        hits = self.vindex.search(emb, k=8, kind="plot")
        if not hits:
            return 0.0, "novel", None

        max_sim = 0.0
        most_similar_id: Optional[str] = None
        most_similar_plot: Optional[Plot] = None

        for pid, sim in hits:
            if sim > max_sim:
                max_sim = sim
                most_similar_id = pid
                most_similar_plot = self.plots.get(pid)

        potential_updates = self.entity_tracker.find_potential_updates(text, ts)

        entity_update = None
        if potential_updates and most_similar_id:
            for old_ea, new_ea, conf in potential_updates:
                if old_ea.plot_id == most_similar_id and conf > 0.5:
                    entity_update = (old_ea.entity, old_ea.attribute, old_ea.value, conf)
                    break

        if entity_update is not None:
            entity, attr, old_value, entity_conf = entity_update
            logger.debug(
                f"检测到实体属性更新：{entity}::{attr} "
                f"({old_value} -> 新值)，置信度={entity_conf:.2f}"
            )
            if entity_conf > 0.5:
                return 0.0, "update", most_similar_id

        if most_similar_plot is not None:
            update_signals = self._detect_update_signals(
                text, most_similar_plot.text, ts, most_similar_plot.ts
            )
            if update_signals["is_update"] and max_sim >= 0.3:
                return 0.0, "update", most_similar_id

        if max_sim < UPDATE_MODERATE_SIMILARITY_THRESHOLD:
            return 0.0, "novel", None

        if max_sim >= UPDATE_HIGH_SIMILARITY_THRESHOLD and most_similar_plot is not None:
            update_signals = self._detect_update_signals(
                text, most_similar_plot.text, ts, most_similar_plot.ts
            )

            if update_signals["is_update"]:
                return 0.0, "update", most_similar_id

            time_gap = abs(ts - most_similar_plot.ts)
            if time_gap < REINFORCEMENT_TIME_WINDOW:
                return 0.5 * max_sim, "reinforcement", most_similar_id

            return max_sim, "pure_redundant", most_similar_id

        if most_similar_plot is not None:
            time_gap = abs(ts - most_similar_plot.ts)
            if time_gap < REINFORCEMENT_TIME_WINDOW:
                return 0.3 * max_sim, "reinforcement", most_similar_id

        return 0.3 * max_sim, "novel", None

    def _detect_update_signals(
        self, new_text: str, old_text: str, new_ts: float, old_ts: float
    ) -> Dict[str, Any]:
        """检测 new_text 是否代表对 old_text 的更新。"""
        signals: List[str] = []
        update_type: Optional[str] = None
        confidence = 0.0

        new_lower = new_text.lower()
        old_lower = old_text.lower()

        keyword_count = sum(1 for kw in UPDATE_KEYWORDS if kw in new_lower)
        if keyword_count > 0:
            signals.append("update_keywords")
            confidence += min(0.3 * keyword_count, 0.6)

            correction_indicators = {"其实", "实际上", "纠正", "更正", "actually", "correction"}
            if any(ind in new_lower for ind in correction_indicators):
                update_type = "correction"
            else:
                update_type = "state_change"

        time_gap = new_ts - old_ts
        if time_gap > UPDATE_TIME_GAP_THRESHOLD:
            signals.append("time_gap")
            gap_factor = min(time_gap / (24 * 3600), 1.0)
            confidence += 0.2 * gap_factor

        entity_update = self.entity_tracker.check_entity_update(new_text, "", new_ts)
        if entity_update is not None:
            entity, attr, old_value, entity_conf = entity_update
            signals.append("entity_attribute_alignment")
            confidence += min(0.4 * entity_conf, 0.5)
            if update_type is None:
                update_type = "state_change"
            logger.debug(
                f"实体属性对齐：{entity}::{attr} "
                f"从 {old_value} 改变（置信度={entity_conf:.2f}）"
            )

        import re

        new_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", new_text))
        old_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", old_text))

        if new_numbers and old_numbers and new_numbers != old_numbers:
            has_change_indicator = any(ind in new_text for ind in NUMERIC_CHANGE_INDICATORS)
            if has_change_indicator or len(new_numbers.symmetric_difference(old_numbers)) > 0:
                signals.append("numeric_change")
                confidence += 0.2
                if update_type is None:
                    update_type = "state_change"

        negation_patterns = [
            "不再", "不是", "没有", "不用", "no longer", "not anymore", "don't", "doesn't"
        ]
        if any(neg in new_lower for neg in negation_patterns):
            signals.append("negation")
            confidence += 0.25
            if update_type is None:
                update_type = "state_change"

        refinement_patterns = ["具体来说", "详细地", "补充", "更准确", "specifically", "to be precise", "additionally"]
        if any(ref in new_lower for ref in refinement_patterns):
            signals.append("refinement")
            confidence += 0.2
            if update_type is None:
                update_type = "refinement"

        is_update = confidence >= 0.3 and len(signals) >= 1

        return {
            "is_update": is_update,
            "update_type": update_type if is_update else None,
            "confidence": confidence,
            "signals": signals,
        }

    def _compute_goal_relevance(self, emb: np.ndarray, context_emb: Optional[np.ndarray]) -> float:
        """计算与当前目标/上下文的相关性。"""
        if context_emb is None:
            return 0.0
        return float(max(0.0, min(1.0, cosine_sim(emb, context_emb))))

    def _normalize_interaction_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """将本地化字段名归一化到批量摄入的标准 schema。"""
        normalized: Dict[str, Any] = {}
        for canonical, aliases in INTERACTION_FIELD_ALIASES.items():
            for alias in aliases:
                if alias in item:
                    normalized[canonical] = item[alias]
                    break
        return normalized

    def _compute_pred_error(self, emb: np.ndarray) -> float:
        """计算与最佳匹配 story 质心的预测误差。"""
        best_sim = -1.0
        for story in self.stories.values():
            if story.centroid is None:
                continue
            sim = self.metric.sim(emb, story.centroid)
            if sim > best_sim:
                best_sim = sim
        return 1.0 if best_sim < 0 else 1.0 - best_sim

    def _compute_voi_features(self, plot: Plot) -> np.ndarray:
        """计算用于编码决策的信息价值特征。"""
        return np.array([
            plot.surprise,
            plot.pred_error,
            1.0 - plot.redundancy,
            plot.goal_relevance,
            math.tanh(len(plot.text) / TEXT_LENGTH_NORMALIZATION),
            1.0,
        ], dtype=np.float32)

    def _compute_knowledge_type_weight(self, plot: Plot) -> float:
        """根据知识类型计算存储权重。"""
        if plot.knowledge_type is None:
            return 0.6

        type_weights = {
            "identity_value": KNOWLEDGE_TYPE_WEIGHT_VALUE,
            "factual_static": KNOWLEDGE_TYPE_WEIGHT_STATIC,
            "identity_trait": KNOWLEDGE_TYPE_WEIGHT_TRAIT,
            "factual_state": KNOWLEDGE_TYPE_WEIGHT_STATE,
            "preference": KNOWLEDGE_TYPE_WEIGHT_PREFERENCE,
            "behavior": KNOWLEDGE_TYPE_WEIGHT_BEHAVIOR,
            "unknown": 0.6,
        }

        base_weight = type_weights.get(plot.knowledge_type, 0.6)
        confidence_factor = 0.5 + 0.5 * plot.knowledge_confidence
        return base_weight * confidence_factor

    def ingest(
        self,
        interaction_text: str,
        actors: Optional[Sequence[str]] = None,
        context_text: Optional[str] = None,
        event_id: Optional[str] = None,
        interaction_embedding: Optional[np.ndarray] = None,
        context_embedding: Optional[np.ndarray] = None,
        ts: Optional[float] = None,
    ) -> Plot:
        """使用关系中心处理摄入交互/事件。"""
        if not interaction_text or not interaction_text.strip():
            raise ValidationError("interaction_text cannot be empty")
        actors = tuple(actors) if actors else ("user", "agent")
        emb = self._coerce_embedding(interaction_embedding) if interaction_embedding is not None else self.embedder.embed(interaction_text)

        plot = self._prepare_plot(interaction_text, actors, emb, event_id, ts)
        self.kde.add(emb)

        context_emb: Optional[np.ndarray]
        if context_embedding is not None:
            context_emb = self._coerce_embedding(context_embedding)
        elif context_text:
            context_emb = self.embedder.embed(context_text)
        else:
            context_emb = None
        self._compute_plot_signals(plot, emb, context_emb)

        encode = self._compute_storage_decision(plot)

        if encode:
            self._store_plot(plot)
            self._recent_encoded_plot_ids.append(plot.id)
            self.entity_tracker.update(interaction_text, plot.id, plot.ts)
            logger.debug(f"编码 plot {plot.id}，combined_prob={plot._storage_prob:.3f}")
        else:
            logger.debug(f"丢弃 plot，combined_prob={plot._storage_prob:.3f}")

        self._pressure_manage()
        return plot

    def ingest_batch(
        self,
        interactions: Sequence[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int, Plot], None]] = None,
        batch_size: int = 25,
    ) -> List[Plot]:
        """使用优化的嵌入批量摄入多个交互。"""
        if not interactions:
            return []

        normalized_items = [self._normalize_interaction_item(item) for item in interactions]

        for i, item in enumerate(normalized_items):
            text = item.get("text", "")
            if not text or not text.strip():
                raise ValidationError(f"interaction[{i}].text cannot be empty")

        total = len(interactions)
        logger.info(f"开始批量摄入 {total} 个交互（batch_size={batch_size}）")

        def _prepare_text_with_date(item: Dict[str, Any]) -> str:
            text = item["text"]
            date = item.get("date")
            if date:
                return f"[{date}] {text}"
            return text

        texts_to_embed = [_prepare_text_with_date(item) for item in normalized_items]

        context_texts = []
        context_indices = []
        for i, item in enumerate(normalized_items):
            ctx = item.get("context_text")
            if ctx:
                context_texts.append(ctx)
                context_indices.append(i)

        logger.info(f"以 {batch_size} 的批次嵌入 {len(texts_to_embed)} 个文本...")
        all_embeddings = self._batch_embed_texts(texts_to_embed, batch_size)

        context_embeddings: Dict[int, np.ndarray] = {}
        if context_texts:
            logger.info(f"嵌入 {len(context_texts)} 个上下文文本...")
            ctx_embs = self._batch_embed_texts(context_texts, batch_size)
            for idx, emb in zip(context_indices, ctx_embs):
                context_embeddings[idx] = emb

        logger.info("嵌入完成。处理 plot...")

        plots: List[Plot] = []
        stored_count = 0

        for i, item in enumerate(normalized_items):
            text = texts_to_embed[i]
            actors = tuple(item.get("actors", ("user", "agent")))
            event_id = item.get("event_id")
            emb = all_embeddings[i]
            context_emb = context_embeddings.get(i)

            plot = self._prepare_plot(text, actors, emb, event_id, None)
            self.kde.add(emb)
            self._compute_plot_signals(plot, emb, context_emb)

            encode = self._compute_storage_decision(plot)

            if encode:
                self._store_plot(plot)
                self._recent_encoded_plot_ids.append(plot.id)
                self.entity_tracker.update(text, plot.id, plot.ts)
                stored_count += 1
                logger.debug(
                    f"[{i+1}/{total}] 编码 plot {plot.id[:8]}...，"
                    f"combined_prob={plot._storage_prob:.3f}"
                )
            else:
                logger.debug(
                    f"[{i+1}/{total}] 丢弃 plot，combined_prob={plot._storage_prob:.3f}"
                )

            plots.append(plot)

            if progress_callback is not None:
                try:
                    progress_callback(i + 1, total, plot)
                except Exception as exc:
                    logger.warning(f"进度回调错误：{exc}")

            if (i + 1) % 50 == 0:
                self._pressure_manage()

        self._pressure_manage()

        logger.info(
            f"批量摄入完成：{total} 个已处理，{stored_count} 个已存储 "
            f"（{stored_count * 100 / total:.1f}% 存储率）"
        )
        return plots

    def _batch_embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 25,
    ) -> List[np.ndarray]:
        """使用嵌入器的批处理能力批量嵌入文本。"""
        if not texts:
            return []

        if hasattr(self.embedder, "embed_batch"):
            all_embeddings: List[np.ndarray] = []
            for batch_start in range(0, len(texts), batch_size):
                batch_end = min(batch_start + batch_size, len(texts))
                batch_texts = texts[batch_start:batch_end]
                batch_embs = self.embedder.embed_batch(batch_texts)
                all_embeddings.extend(batch_embs)
                logger.debug(
                    f"嵌入批次 {batch_start//batch_size + 1}/"
                    f"{(len(texts) + batch_size - 1)//batch_size}"
                )
            return all_embeddings

        logger.warning("嵌入器不支持 embed_batch，回退到顺序嵌入")
        return [self.embedder.embed(t) for t in texts]

    def _prepare_plot(
        self,
        interaction_text: str,
        actors: Tuple[str, ...],
        emb: np.ndarray,
        event_id: Optional[str],
        ts: Optional[float],
    ) -> Plot:
        """使用关系中心上下文和知识类型分类准备 plot。"""
        relationship_entity = self._identify_relationship_entity(actors, interaction_text)
        identity_relevance = self._assess_identity_relevance(interaction_text, relationship_entity, emb)
        relational_context = self._extract_relational_context(
            interaction_text, relationship_entity, actors, identity_relevance
        )
        identity_impact = self._extract_identity_impact(
            interaction_text, relational_context, identity_relevance
        )

        classification = self.knowledge_classifier.classify(interaction_text, embedding=emb)
        knowledge_type = classification.knowledge_type.value
        knowledge_confidence = classification.confidence

        plot = Plot(
            id=det_id("plot", event_id) if event_id else str(uuid.uuid4()),
            ts=float(ts) if ts is not None else now_ts(),
            text=interaction_text,
            actors=tuple(actors),
            embedding=emb,
            relational=relational_context,
            identity_impact=identity_impact,
            knowledge_type=knowledge_type,
            knowledge_confidence=knowledge_confidence,
        )

        self.fact_extractor.augment_plot(plot)
        return plot

    def _coerce_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """将外部提供的 embedding 规范化为内部 float32 向量。"""
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim != 1:
            raise ValidationError("embedding must be a one-dimensional vector")
        return arr

    def _compute_plot_signals(
        self, plot: Plot, emb: np.ndarray, context_emb: Optional[np.ndarray]
    ) -> None:
        """计算 plot 的传统信号，进行更新检测。"""
        plot.surprise = float(self.kde.surprise(emb))
        plot.pred_error = float(self._compute_pred_error(emb))

        redundancy_score, redundancy_type, supersedes_id = self._compute_redundancy(
            emb, plot.text, plot.ts
        )

        plot.redundancy = float(redundancy_score)
        plot.redundancy_type = redundancy_type

        if redundancy_type == "update" and supersedes_id is not None:
            if supersedes_id in self.plots:
                old_plot = self.plots[supersedes_id]
                actors_compatible = self._actors_compatible_for_update(plot.actors, old_plot.actors)

                if actors_compatible:
                    plot.supersedes_id = supersedes_id
                    update_signals = self._detect_update_signals(
                        plot.text, old_plot.text, plot.ts, old_plot.ts
                    )
                    plot.update_type = update_signals.get("update_type")
                    old_plot.status = "superseded"
                    old_plot.superseded_by_id = plot.id
                    logger.info(
                        f"检测到更新：{plot.id[:8]}... 替代 {supersedes_id[:8]}... "
                        f"(update_type={plot.update_type})"
                    )
                else:
                    plot.redundancy_type = "reinforcement"
                    logger.debug(
                        f"跳过替代：参与者不兼容。"
                        f"新：{plot.actors}，旧：{old_plot.actors}"
                    )

        plot.goal_relevance = float(self._compute_goal_relevance(emb, context_emb))
        plot.tension = plot.surprise * (1.0 + plot.pred_error)

    def _actors_compatible_for_update(
        self, new_actors: Tuple[str, ...], old_actors: Tuple[str, ...]
    ) -> bool:
        """检查参与者是否兼容以进行替代。"""

        def get_primary_speaker(actors: Tuple[str, ...]) -> Optional[str]:
            for actor in actors:
                actor_lower = actor.lower()
                if actor_lower in ("user", "human", "customer"):
                    return "user"
                if actor_lower in ("assistant", "agent", "ai", "bot"):
                    return "assistant"
            return actors[0].lower() if actors else None

        new_speaker = get_primary_speaker(new_actors)
        old_speaker = get_primary_speaker(old_actors)

        if new_speaker == old_speaker:
            return True

        return False

    def _compute_storage_decision(self, plot: Plot) -> bool:
        """计算是否存储此 plot。"""
        if self.benchmark_mode:
            plot._storage_prob = 1.0
            logger.debug(f"基准模式：强制存储 plot {plot.id[:8]}...")
            return True

        if len(self.plots) < COLD_START_FORCE_STORE_COUNT:
            plot._storage_prob = 1.0
            logger.debug(f"冷启动：强制存储 plot {len(self.plots) + 1}/{COLD_START_FORCE_STORE_COUNT}")
            return True

        if plot.redundancy_type == "update":
            plot._storage_prob = 1.0
            logger.debug(
                f"检测到知识更新：强制存储 plot，"
                f"supersedes={plot.supersedes_id}，update_type={plot.update_type}"
            )
            return True

        x = self._compute_voi_features(plot)
        voi_decision = self.gate.prob(x)

        identity_relevance = self._assess_identity_relevance(
            plot.text,
            plot.get_relationship_entity() or "user",
            plot.embedding,
        )

        knowledge_weight = self._compute_knowledge_type_weight(plot)
        combined_prob = (
            IDENTITY_RELEVANCE_WEIGHT * identity_relevance +
            VOI_DECISION_WEIGHT * voi_decision
        )

        knowledge_boost = (knowledge_weight - 0.5) * 0.4
        combined_prob = combined_prob + knowledge_boost

        combined_prob = max(combined_prob, MIN_STORE_PROB)
        combined_prob = min(combined_prob, 1.0)
        plot._storage_prob = combined_prob

        return self.rng.random() < combined_prob

    def _store_plot(self, plot: Plot) -> None:
        """使用关系优先组织和冲突检测存储 plot。"""
        conflicts = self._detect_conflicts(plot)
        if conflicts:
            self._handle_conflicts(plot, conflicts)

        story, _chosen_id = self._assign_plot_to_story(plot)
        self._update_story_with_plot(story, plot)
        self._weave_plot_edges(plot, story)
        self._add_to_temporal_index(plot)
        self._update_identity_dimensions(plot)

    def _detect_conflicts(self, new_plot: Plot) -> List[Conflict]:
        """检测新 plot 与现有记忆之间的潜在冲突。"""
        conflicts: List[Conflict] = []

        if not self.plots:
            return conflicts

        similar_plots = self.vindex.search(
            new_plot.embedding,
            k=CONFLICT_CHECK_K,
            kind="plot",
        )

        for pid, sim in similar_plots:
            if sim < CONFLICT_CHECK_SIMILARITY_THRESHOLD:
                continue

            old_plot = self.plots.get(pid)
            if old_plot is None or old_plot.status != "active":
                continue

            prob, explanation = self.coherence_guardian.detector.detect_contradiction(
                old_plot, new_plot
            )

            if prob > CONFLICT_PROBABILITY_THRESHOLD:
                conflict = Conflict(
                    type=ConflictType.FACTUAL,
                    node_a=old_plot.id,
                    node_b=new_plot.id,
                    severity=prob,
                    confidence=sim,
                    description=explanation,
                    evidence=[old_plot.text[:100], new_plot.text[:100]],
                )
                conflicts.append(conflict)

                logger.debug(
                    f"检测到冲突：{old_plot.id} <-> {new_plot.id}，"
                    f"prob={prob:.3f}，sim={sim:.3f}，reason={explanation}"
                )

        return conflicts[:MAX_CONFLICTS_PER_INGEST]

    def _handle_conflicts(self, new_plot: Plot, conflicts: List[Conflict]) -> None:
        """根据知识类型分类处理检测到的冲突。"""
        for conflict in conflicts:
            old_plot = self.plots.get(conflict.node_a)
            if old_plot is None:
                continue

            old_classification = self.knowledge_classifier.classify(old_plot.text)
            new_classification = self.knowledge_classifier.classify(new_plot.text)

            time_gap = abs(new_plot.ts - old_plot.ts)
            time_relation = "sequential" if time_gap > CONCURRENT_TIME_THRESHOLD else "concurrent"

            analysis = self.knowledge_classifier.resolve_conflict(
                old_classification.knowledge_type,
                new_classification.knowledge_type,
                time_relation,
                old_plot.text,
                new_plot.text,
                old_plot.embedding,
                new_plot.embedding,
            )

            self._apply_conflict_resolution(old_plot, new_plot, analysis, conflict)

    def _apply_conflict_resolution(
        self,
        old_plot: Plot,
        new_plot: Plot,
        analysis: ConflictAnalysis,
        conflict: Conflict,
    ) -> None:
        """应用冲突解决策略。"""
        resolution = analysis.resolution

        if resolution == ConflictResolution.UPDATE:
            new_plot.supersedes_id = old_plot.id
            old_plot.superseded_by_id = new_plot.id
            old_plot.status = "superseded"
            new_plot.update_type = "state_change"
            new_plot.redundancy_type = "update"

            logger.info(
                f"UPDATE 解决方案：{new_plot.id} 替代 {old_plot.id}。"
                f"原因：{analysis.rationale}"
            )

        elif resolution == ConflictResolution.PRESERVE_BOTH:
            self.graph.ensure_edge(old_plot.id, new_plot.id, "tension")
            self.graph.ensure_edge(new_plot.id, old_plot.id, "tension")

            if analysis.is_complementary:
                from aurora.core.tension import Tension, TensionType
                tension = Tension(
                    id=f"tension-{old_plot.id}-{new_plot.id}",
                    element_a_id=old_plot.id,
                    element_a_type="plot",
                    element_b_id=new_plot.id,
                    element_b_type="plot",
                    description=f"互补特征：{analysis.rationale}",
                    tension_type=TensionType.ADAPTIVE,
                    severity=conflict.severity,
                )
                self.coherence_guardian.tension_manager.tensions[tension.id] = tension

            logger.info(
                f"PRESERVE_BOTH 解决方案：{old_plot.id} 和 {new_plot.id} 都活跃。"
                f"原因：{analysis.rationale}"
            )

        elif resolution == ConflictResolution.CORRECT:
            new_plot.supersedes_id = old_plot.id
            old_plot.superseded_by_id = new_plot.id
            old_plot.status = "corrected"
            new_plot.update_type = "correction"
            new_plot.redundancy_type = "update"

            logger.info(
                f"CORRECT 解决方案：{old_plot.id} 由 {new_plot.id} 更正。"
                f"原因：{analysis.rationale}"
            )

        elif resolution == ConflictResolution.EVOLVE:
            new_plot.supersedes_id = old_plot.id
            new_plot.update_type = "refinement"
            new_plot.redundancy_type = "update"
            self.graph.ensure_edge(old_plot.id, new_plot.id, "evolved_to")

            logger.info(
                f"EVOLVE 解决方案：{old_plot.id} 演化为 {new_plot.id}。"
                f"原因：{analysis.rationale}"
            )

        else:
            logger.debug(
                f"对 {old_plot.id} 和 {new_plot.id} 之间的冲突无操作。"
                f"原因：{analysis.rationale}"
            )

    def _assign_plot_to_story(self, plot: Plot) -> Tuple[StoryArc, str]:
        """将 plot 分配给现有或新 story。"""
        relationship_entity = plot.get_relationship_entity()

        if relationship_entity:
            story = self._get_or_create_relationship_story(relationship_entity)
            chosen_id = story.id

            if story.centroid is None:
                self.vindex.add(story.id, plot.embedding, kind="story")
        else:
            logps: Dict[str, float] = {}
            for sid, story in self.stories.items():
                prior = math.log(len(story.plot_ids) + EPSILON_PRIOR)
                logps[sid] = prior + self.story_model.loglik(plot, story)

            chosen_id, _ = self.crp_story.sample(logps)
            if chosen_id is None:
                story = StoryArc(id=det_id("story", plot.id), created_ts=now_ts(), updated_ts=now_ts())
                self.stories[story.id] = story
                self.graph.add_node(story.id, "story", story)
                self.vindex.add(story.id, plot.embedding, kind="story")
                chosen_id = story.id
            else:
                story = self.stories[chosen_id]

        return story, chosen_id

    def _update_story_with_plot(self, story: StoryArc, plot: Plot) -> None:
        """使用新 plot 更新 story 统计信息和质心。"""
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            story._update_stats("dist", d2)
            gap = max(0.0, plot.ts - story.updated_ts)
            story._update_stats("gap", gap)

        story.plot_ids.append(plot.id)
        story.updated_ts = plot.ts
        story.actor_counts = {a: story.actor_counts.get(a, 0) + 1 for a in plot.actors}
        story.tension_curve.append(plot.tension)

        story.centroid = self._update_centroid_online(
            story.centroid, plot.embedding, len(story.plot_ids)
        )

        if plot.relational and story.is_relationship_story():
            self._update_relationship_trajectory(story, plot)

    def _update_relationship_trajectory(self, story: StoryArc, plot: Plot) -> None:
        """使用新 plot 更新关系轨迹。"""
        story.add_relationship_moment(
            event_summary=plot.text[:EVENT_SUMMARY_MAX_LENGTH] + "..." if len(plot.text) > EVENT_SUMMARY_MAX_LENGTH else plot.text,
            trust_level=TRUST_BASE + plot.relational.relationship_quality_delta,
            my_role=plot.relational.my_role_in_relation,
            quality_delta=plot.relational.relationship_quality_delta,
            ts=plot.ts,
        )

        if len(story.relationship_arc) >= 3:
            recent_roles = [m.my_role for m in story.relationship_arc[-10:]]
            role_counts = Counter(recent_roles)
            dominant_role = role_counts.most_common(1)[0][0] if role_counts else "助手"
            story.update_identity_in_relationship(dominant_role)

    def _weave_plot_edges(self, plot: Plot, story: StoryArc) -> None:
        """在图中存储 plot 和编织边。"""
        plot.story_id = story.id
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, plot.embedding, kind="plot")

        self._create_bidirectional_edge(plot.id, story.id, "belongs_to", "contains")

        if len(story.plot_ids) > 1:
            prev_id = story.plot_ids[-2]
            self.graph.ensure_edge(prev_id, plot.id, "temporal")

        for pid, _ in self.vindex.search(plot.embedding, k=SEMANTIC_NEIGHBORS_K, kind="plot"):
            if pid != plot.id:
                self.graph.ensure_edge(plot.id, pid, "semantic")
                self.graph.ensure_edge(pid, plot.id, "semantic")
