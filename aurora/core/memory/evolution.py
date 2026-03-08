"""
AURORA 演化模块
=======================

演化、反思和意义重构功能。

主要职责：
- 关系反思（评估关系健康状况、提取经验教训）
- 意义重构（更新对过去经历的理解）
- 故事状态更新（基于活动的概率性更新）
- 主题/身份维度的出现
- 身份矛盾分析
- 故事边界检测（高潮、解决、放弃）
- 图结构清理（弱边、相似节点、陈旧内容）

哲学："持续成为" - 身份通过经历而出现和演化。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from aurora.core.models.theme import Theme
from aurora.core.models.trace import EvolutionPatch, EvolutionSnapshot
from aurora.utils.id_utils import det_id
from aurora.utils.math_utils import sigmoid, softmax
from aurora.utils.time_utils import now_ts
from aurora.core.config.evolution import (
    CLIMAX_DECLINE_RATIO,
    CLIMAX_TENSION_WINDOW,
    PERIODIC_REFLECTION_ACCESS_COUNT,
    PERIODIC_REFLECTION_AGE_DAYS,
    REFRAME_ACCESS_COUNT_THRESHOLD,
    REFRAME_AGE_DAYS_THRESHOLD,
    RESOLUTION_MIN_ARC_LENGTH,
    RESOLUTION_TENSION_DROP_RATIO,
    STORY_ABANDONMENT_THRESHOLD_DAYS,
)
from aurora.core.config.identity import HARMONY_SIMILARITY_MIN, TENSION_SIMILARITY_MAX, TENSION_SIMILARITY_MIN
from aurora.core.config.numeric import EPSILON_PRIOR
from aurora.core.config.storage import (
    ARCHIVE_MIN_ACCESS_COUNT,
    ARCHIVE_STALE_DAYS_THRESHOLD,
    NODE_MERGE_SIMILARITY_THRESHOLD,
    WEAK_EDGE_MIN_SUCCESSES,
    WEAK_EDGE_MIN_WEIGHT,
)

if TYPE_CHECKING:
    from aurora.core.models.plot import Plot
    from aurora.core.models.story import StoryArc

logger = logging.getLogger(__name__)

class EvolutionMixin:
    """提供演化、反思和意义重构功能的混入类。"""

    # -------------------------------------------------------------------------
    # 关系反思
    # -------------------------------------------------------------------------

    def _reflect_on_relationships(self) -> None:
        """
        反思关系：这是"持续成为"的关键部分。

        需要回答的问题：
        1. 哪些关系在成长，哪些停滞不前？
        2. 我的身份在不同关系中是否一致？应该一致吗？
        3. 可以从每段关系中提取哪些经验教训？
        """
        for entity, story_id in self._relationship_story_index.items():
            story = self.stories.get(story_id)
            if story is None:
                continue

            # 如果没有有意义的历史，则跳过
            if len(story.relationship_arc) < 3:
                continue

            # 1. 评估关系健康趋势
            trust_trend = story.get_trust_trend(window=10)
            role_consistency = story.get_role_consistency(window=10)

            # 根据趋势更新关系健康状况
            health_delta = 0.1 * trust_trend + 0.05 * (role_consistency - 0.5)
            story.relationship_health = max(0.0, min(1.0,
                story.relationship_health + health_delta
            ))

            # 2. 如果关系已成熟，则提取经验教训
            if len(story.relationship_arc) >= 10 and len(story.lessons_from_relationship) < 3:
                lesson = self._extract_relationship_lesson(story)
                if lesson:
                    story.add_lesson(lesson)

            # 3. 根据一致的角色更新身份
            if role_consistency > HARMONY_SIMILARITY_MIN and story.relationship_arc:
                dominant_role = story.relationship_arc[-1].my_role
                if dominant_role != story.my_identity_in_this_relationship:
                    story.update_identity_in_relationship(dominant_role)
    
    def _extract_relationship_lesson(self, story: "StoryArc") -> Optional[str]:
        """
        从关系历史中提取经验教训。

        参数：
            story: 要提取经验教训的关系故事

        返回：
            经验教训字符串或无（如果没有明确的经验教训）
        """
        if not story.relationship_arc:
            return None

        # 简单启发式：查看信任演化
        if len(story.relationship_arc) >= 5:
            early_trust = sum(m.trust_level for m in story.relationship_arc[:3]) / 3
            recent_trust = sum(m.trust_level for m in story.relationship_arc[-3:]) / 3

            if recent_trust > early_trust + 0.2:
                return "耐心和持续的努力能建立信任"
            elif recent_trust < early_trust - 0.2:
                return "关系需要持续维护"

        # 检查角色一致性
        roles = [m.my_role for m in story.relationship_arc[-10:]]
        if len(set(roles)) == 1:
            return f"在这段关系中，保持{roles[0]}的角色是有效的"

        return None

    # -------------------------------------------------------------------------
    # 意义重构
    # -------------------------------------------------------------------------

    def _check_reframe_opportunities(self) -> None:
        """
        检查哪些记忆需要更新其意义。

        哲学：过去经历的意义可以演化。
        看起来像失败的事情后来可能被理解为转折点。

        触发条件：
        1. 身份维度强度发生了显著变化
        2. 关系质量改变（好→坏或坏→好）
        3. 积累了足够的反馈
        """
        for plot in self.plots.values():
            if plot.status != "active":
                continue
            if not plot.identity_impact:
                continue

            # 检查是否需要重构
            should_reframe, reason = self._check_should_reframe(plot)

            if should_reframe:
                new_meaning = self._generate_new_meaning(plot, reason)
                if new_meaning and new_meaning != plot.identity_impact.current_meaning:
                    plot.identity_impact.update_meaning(new_meaning)
    
    def _check_should_reframe(self, plot: "Plot") -> Tuple[bool, str]:
        """
        确定是否应该重构情节的意义。

        参数：
            plot: 要检查的情节

        返回：
            (should_reframe: bool, reason: str) 的元组
        """
        if not plot.identity_impact:
            return False, ""

        # 触发条件1：旧情节且访问次数多（重要但意义可能已演化）
        age_days = (now_ts() - plot.ts) / 86400
        if age_days > REFRAME_AGE_DAYS_THRESHOLD and plot.access_count > REFRAME_ACCESS_COUNT_THRESHOLD:
            # 检查身份维度是否已改变
            for dim in plot.identity_impact.identity_dimensions_affected:
                current_strength = self._identity_dimensions.get(dim, 0.5)
                # 如果我们在这个维度上已经成长，意义可能已加深
                if current_strength > 0.7:
                    return True, f"身份维度「{dim}」已增强"

        # 触发条件2：关系质量发生了显著变化
        if plot.relational:
            relationship_entity = plot.relational.with_whom
            story = self.get_relationship_story(relationship_entity)
            if story:
                # 比较当前信任与情节时的信任
                current_trust = story.get_current_trust()
                original_delta = plot.relational.relationship_quality_delta

                # 如果关系改善但情节是负面的，则重构
                if current_trust > 0.7 and original_delta < -0.1:
                    return True, "关系改善后重新理解"
                # 如果关系恶化但情节是正面的，则重构
                if current_trust < 0.3 and original_delta > 0.1:
                    return True, "关系变化后重新理解"

        # 触发条件3：对于重要情节，意义已有一段时间未更新
        if not plot.identity_impact.evolution_history:
            if age_days > PERIODIC_REFLECTION_AGE_DAYS and plot.access_count > PERIODIC_REFLECTION_ACCESS_COUNT:
                return True, "定期反思"

        return False, ""
    
    def _generate_new_meaning(self, plot: "Plot", reason: str) -> Optional[str]:
        """
        根据当前理解为情节生成新的意义。

        参数：
            plot: 要生成新意义的情节
            reason: 重构的原因

        返回：
            新的意义字符串或无
        """
        if not plot.identity_impact:
            return None

        # 根据原因生成新意义
        if "身份维度" in reason and "增强" in reason:
            # 这次经历促进了成长
            dims = plot.identity_impact.identity_dimensions_affected[:2]
            if dims:
                return f"这是我成为{dims[0]}的重要一步"

        elif "关系改善" in reason:
            return "虽然当时困难，但这帮助了关系的成长"

        elif "关系变化" in reason:
            return "这次经历让我对这段关系有了更深的理解"

        elif "定期反思" in reason:
            # 意义的通用深化
            if plot.relational:
                return f"在与{plot.relational.with_whom}的关系中，这是一个有意义的时刻"

        return None

    # -------------------------------------------------------------------------
    # 身份矛盾分析
    # -------------------------------------------------------------------------

    def _analyze_identity_tensions(self) -> None:
        """
        分析身份维度之间的矛盾。

        哲学：并非所有矛盾都是坏的。
        - 有些应该被解决（阻止行动）
        - 有些应该被保留（提供灵活性）
        - 有些应该被接受（成长的迹象）
        """
        theme_list = list(self.themes.values())

        for i, theme_a in enumerate(theme_list):
            if not theme_a.is_identity_dimension():
                continue

            for theme_b in theme_list[i+1:]:
                if not theme_b.is_identity_dimension():
                    continue

                # 检查矛盾
                if theme_a.prototype is not None and theme_b.prototype is not None:
                    sim = self.metric.sim(theme_a.prototype, theme_b.prototype)

                    # 中等相似度但不同维度 = 潜在的适应性矛盾
                    if TENSION_SIMILARITY_MIN < sim < TENSION_SIMILARITY_MAX:
                        # 这可能是一个适应性矛盾（在不同背景下两者都可以是真的）
                        # 例如："作为耐心解释者的我" vs "作为高效回应者的我"
                        theme_a.add_tension(theme_b.id)
                        theme_b.add_tension(theme_a.id)

                    elif sim > HARMONY_SIMILARITY_MIN:
                        # 高相似度 = 和谐，这些维度相互补充
                        theme_a.add_harmony(theme_b.id)
                        theme_b.add_harmony(theme_a.id)

    # -------------------------------------------------------------------------
    # 故事边界检测
    # -------------------------------------------------------------------------

    def _detect_story_boundaries(self) -> None:
        """
        检测故事边界：高潮、解决和放弃。

        哲学：故事有自然的生命周期。检测故事在其生命周期中的位置
        有助于记忆组织和检索。

        三种关键边界类型：
        - 高潮：故事达到张力峰值（变革时刻）
        - 解决：核心冲突得到解决（自然结束）
        - 放弃：故事进入休眠（可能稍后重新访问）
        """
        boundary_stats = {"climax": 0, "resolution": 0, "abandonment": 0}

        for story in self.stories.values():
            if story.status != "developing":
                continue

            # 检查高潮（达到张力峰值）
            if self._detect_climax(story):
                story.status = "resolved"
                boundary_stats["climax"] += 1
                logger.debug(f"故事 {story.id} 达到高潮，标记为已解决")
                continue

            # 检查解决（冲突已解决）
            if self._detect_resolution(story):
                story.status = "resolved"
                boundary_stats["resolution"] += 1
                logger.debug(f"故事 {story.id} 达到解决")
                continue

            # 检查放弃（长期不活动）
            if self._detect_abandonment(story):
                story.status = "abandoned"
                boundary_stats["abandonment"] += 1
                logger.debug(f"故事 {story.id} 检测为已放弃")

        if any(boundary_stats.values()):
            logger.info(f"故事边界检测：{boundary_stats}")

    def _detect_climax(self, story: "StoryArc") -> bool:
        """
        检测故事是否达到高潮（张力峰值后跟随下降）。

        哲学：高潮是一个变革时刻，张力在此达到峰值
        然后开始释放。这是概率性的，而非确定性的。

        参数：
            story: 要检查的故事

        返回：
            如果以概率确定性检测到高潮，则为真
        """
        curve = story.tension_curve

        # 需要足够的历史来检测高潮
        if len(curve) < CLIMAX_TENSION_WINDOW:
            return False

        # 在张力曲线中找到峰值
        window = curve[-CLIMAX_TENSION_WINDOW:]
        peak_idx = int(np.argmax(window))
        peak_val = window[peak_idx]

        # 峰值应至少比当前值高30%
        current_val = window[-1]
        if peak_val <= 0:
            return False

        decline_ratio = (peak_val - current_val) / peak_val

        # 使用sigmoid转换为概率而不是硬阈值
        # 下降越多 = 高潮概率越高
        p_climax = sigmoid((decline_ratio - CLIMAX_DECLINE_RATIO) * 10)

        # 概率性决策
        return self.rng.random() < p_climax

    def _detect_resolution(self, story: "StoryArc") -> bool:
        """
        检测故事是否达到解决（核心冲突已解决）。

        哲学：解决发生在张力从其峰值显著下降时，
        表明核心冲突已被解决。

        参数：
            story: 要检查的故事

        返回：
            如果以概率确定性检测到解决，则为真
        """
        curve = story.tension_curve

        # 需要足够的弧长以获得有意义的解决
        if len(curve) < RESOLUTION_MIN_ARC_LENGTH:
            return False

        # 检查张力是否从峰值显著下降
        peak_tension = max(curve)
        current_tension = curve[-1]

        if peak_tension <= 0:
            return False

        drop_ratio = (peak_tension - current_tension) / peak_tension

        # 附加信号：检查关系故事的关系健康状况
        health_factor = 1.0
        if story.is_relationship_story():
            # 高关系健康状况增加解决概率
            health_factor = 1.0 + (story.relationship_health - 0.5)

        # 使用sigmoid转换为概率
        p_resolution = sigmoid((drop_ratio - RESOLUTION_TENSION_DROP_RATIO) * 8) * health_factor
        p_resolution = min(1.0, p_resolution)  # 限制在有效概率范围内

        return self.rng.random() < p_resolution

    def _detect_abandonment(
        self, story: "StoryArc", threshold_days: float = STORY_ABANDONMENT_THRESHOLD_DAYS
    ) -> bool:
        """
        检测故事是否已被放弃（长期无活动）。

        哲学：故事可以在没有达到明确解决的情况下进入休眠。
        这不一定是负面的 - 它们可能稍后被重新访问。
        放弃是基于不活动持续时间的概率性的。

        参数：
            story: 要检查的故事
            threshold_days: 考虑放弃前的不活动天数

        返回：
            如果以概率确定性检测到放弃，则为真
        """
        ts = now_ts()
        idle_seconds = ts - story.updated_ts
        idle_days = idle_seconds / 86400.0

        # 使用故事自己的时间节奏作为背景
        # 通常有长间隙的故事不太可能被"放弃"
        gap_mean_days = story.gap_mean_safe() / 86400.0

        # 按故事的典型间隙规范化空闲时间
        # 如果 idle_days >> gap_mean_days，更可能被放弃
        if gap_mean_days > 0:
            normalized_idle = idle_days / gap_mean_days
        else:
            normalized_idle = idle_days / 7.0  # 默认1周节奏

        # 使用sigmoid计算放弃概率
        # 以threshold_days为中心，故事节奏作为调制
        p_abandon = sigmoid((idle_days - threshold_days) / 10.0)

        # 如果故事有活跃的关系背景，则降低概率
        if story.is_relationship_story() and story.relationship_health > 0.6:
            p_abandon *= 0.5  # 健康的关系不太可能被放弃

        return self.rng.random() < p_abandon

    # -------------------------------------------------------------------------
    # 图结构清理
    # -------------------------------------------------------------------------

    def _cleanup_graph_structure(self) -> None:
        """
        清理图结构以提高内存效率和一致性。

        哲学：记忆系统需要维护以保持健康。
        像人类遗忘一样，这是适应性的，而非病理性的。

        清理操作：
        1. 移除弱边（低置信度连接）
        2. 合并高度相似的节点（减少冗余）
        3. 归档陈旧内容（长期未使用的记忆）
        """
        logger.info("开始图结构清理")

        # 1. 移除弱边
        edges_removed = self._remove_weak_edges()

        # 2. 合并相似节点（仅限情节，以保留结构）
        nodes_merged = self._merge_similar_nodes()

        # 3. 归档陈旧内容
        archived = self._archive_stale_content()

        logger.info(
            f"图清理完成：edges_removed={edges_removed}，"
            f"nodes_merged={nodes_merged}，content_archived={archived}"
        )

    def _remove_weak_edges(self, min_weight: float = WEAK_EDGE_MIN_WEIGHT) -> int:
        """
        移除低置信度权重的边（低置信度连接）。

        哲学：未通过使用强化的边
        是移除的候选。使用概率决策以
        避免严格的截断。

        参数：
            min_weight: 保留的最小边权重（软阈值）

        返回：
            移除的边数
        """
        edges_to_remove: List[Tuple[str, str]] = []

        for src, dst, data in self.graph.g.edges(data=True):
            belief = data.get("belief")
            if belief is None:
                continue

            # 获取边强度（正结果的概率）
            edge_weight = belief.mean()

            # 使用sigmoid计算移除概率
            # 远低于阈值的边有高移除概率
            p_remove = sigmoid((min_weight - edge_weight) * 20)

            # 附加因素：试验次数很少的边被保留
            # （对新边的善意假设）
            # EdgeBelief使用Beta(a, b)，其中a-1 = 成功，b-1 = 失败
            # use_count跟踪总使用次数
            if belief.use_count < WEAK_EDGE_MIN_SUCCESSES:
                p_remove *= 0.1  # 强烈降低移除概率

            if self.rng.random() < p_remove:
                edges_to_remove.append((src, dst))

        # 移除边
        for src, dst in edges_to_remove:
            self.graph.g.remove_edge(src, dst)
            logger.debug(f"移除弱边：{src} -> {dst}")

        return len(edges_to_remove)

    def _merge_similar_nodes(
        self, similarity_threshold: float = NODE_MERGE_SIMILARITY_THRESHOLD
    ) -> int:
        """
        合并高度相似的情节节点以减少冗余。

        哲学：非常相似的记忆可以在不丧失
        基本信息的情况下进行整合。这类似于人类
        睡眠期间的记忆整合。

        参数：
            similarity_threshold: 节点可能被合并的相似度阈值

        返回：
            执行的合并操作数
        """
        merge_count = 0
        merged_ids: Set[str] = set()

        # 仅合并情节（故事和主题有语义结构）
        plot_ids = list(self.plots.keys())

        for i, pid_a in enumerate(plot_ids):
            if pid_a in merged_ids:
                continue

            plot_a = self.plots.get(pid_a)
            if plot_a is None or plot_a.status != "active":
                continue

            for pid_b in plot_ids[i + 1:]:
                if pid_b in merged_ids:
                    continue

                plot_b = self.plots.get(pid_b)
                if plot_b is None or plot_b.status != "active":
                    continue

                # 计算相似度
                sim = self.metric.sim(plot_a.embedding, plot_b.embedding)

                # 概率性合并决策
                p_merge = sigmoid((sim - similarity_threshold) * 50)

                if self.rng.random() < p_merge:
                    # 将B合并到A中（保留A作为幸存者）
                    self._merge_plots(plot_a, plot_b)
                    merged_ids.add(pid_b)
                    merge_count += 1
                    logger.debug(f"将情节 {pid_b} 合并到 {pid_a}（sim={sim:.3f}）")

        return merge_count

    def _merge_plots(self, survivor: "Plot", merged: "Plot") -> None:
        """
        将一个情节合并到另一个中。

        参数：
            survivor: 将保留的情节
            merged: 要合并到survivor中的情节
        """
        # 用合并的数据更新幸存者
        survivor.access_count += merged.access_count

        # 保留更近的last_access_ts
        if merged.last_access_ts and survivor.last_access_ts:
            survivor.last_access_ts = max(survivor.last_access_ts, merged.last_access_ts)
        elif merged.last_access_ts:
            survivor.last_access_ts = merged.last_access_ts

        # 标记合并的情节为已吸收（语义上：被吸收到另一个情节中）
        merged.status = "absorbed"

        # 如果需要，更新故事引用
        if merged.story_id and merged.story_id in self.stories:
            story = self.stories[merged.story_id]
            if merged.id in story.plot_ids:
                story.plot_ids.remove(merged.id)
                if survivor.id not in story.plot_ids:
                    story.plot_ids.append(survivor.id)

        # 从向量索引中移除合并的情节
        self.vindex.remove(merged.id)

        # 更新图边：将合并节点的边重定向到幸存者
        if self.graph.g.has_node(merged.id):
            # 获取涉及合并节点的所有边
            in_edges = list(self.graph.g.in_edges(merged.id))
            out_edges = list(self.graph.g.out_edges(merged.id))

            # 将边重定向到幸存者
            for src, _ in in_edges:
                if src != survivor.id:
                    edge_data = self.graph.g.edges[src, merged.id]
                    self.graph.g.add_edge(src, survivor.id, **edge_data)

            for _, dst in out_edges:
                if dst != survivor.id:
                    edge_data = self.graph.g.edges[merged.id, dst]
                    self.graph.g.add_edge(survivor.id, dst, **edge_data)

            # 移除合并的节点
            self.graph.g.remove_node(merged.id)

        # 从情节字典中移除
        del self.plots[merged.id]

    def _archive_stale_content(
        self, days_threshold: float = ARCHIVE_STALE_DAYS_THRESHOLD
    ) -> int:
        """
        归档长期未被访问的内容。

        哲学：长期未使用的记忆被归档（而非删除）以
        维持系统性能。如果需要，它们可以被检索
        但不参与活跃检索。

        参数：
            days_threshold: 归档前无访问的天数（软阈值）

        返回：
            归档的项目数
        """
        ts = now_ts()
        archived_count = 0

        for plot in self.plots.values():
            if plot.status != "active":
                continue

            # 计算陈旧度
            if plot.last_access_ts:
                idle_days = (ts - plot.last_access_ts) / 86400.0
            else:
                idle_days = (ts - plot.ts) / 86400.0

            # 访问次数影响归档概率
            # 频繁访问的内容不太可能被归档
            access_factor = 1.0 / (1.0 + plot.access_count)

            # 计算归档概率
            p_archive = sigmoid((idle_days - days_threshold) / 30.0) * access_factor

            # 访问次数很少会增加归档概率
            if plot.access_count <= ARCHIVE_MIN_ACCESS_COUNT:
                p_archive = min(1.0, p_archive * 1.5)

            if self.rng.random() < p_archive:
                plot.status = "archived"
                archived_count += 1

                # 从向量索引中移除以加快搜索
                self.vindex.remove(plot.id)

                logger.debug(
                    f"归档情节 {plot.id}（idle_days={idle_days:.1f}，"
                    f"access_count={plot.access_count}）"
                )

        return archived_count

    # -------------------------------------------------------------------------
    # 异步演化：写时复制以实现非阻塞处理
    # -------------------------------------------------------------------------

    def create_evolution_snapshot(self) -> EvolutionSnapshot:
        """
        为演化处理创建只读快照。

        这捕获当前状态而不持有锁，允许
        演化在后台线程中进行，同时新的摄入继续。

        返回：
            包含当前状态的EvolutionSnapshot
        """
        return EvolutionSnapshot(
            story_ids=list(self.stories.keys()),
            story_statuses={sid: s.status for sid, s in self.stories.items()},
            story_centroids={sid: s.centroid.copy() if s.centroid is not None else None
                           for sid, s in self.stories.items()},
            story_tension_curves={sid: list(s.tension_curve) for sid, s in self.stories.items()},
            story_updated_ts={sid: s.updated_ts for sid, s in self.stories.items()},
            story_gap_means={sid: s.gap_mean_safe() for sid, s in self.stories.items()},
            theme_ids=list(self.themes.keys()),
            theme_story_counts={tid: len(t.story_ids) for tid, t in self.themes.items()},
            theme_prototypes={tid: t.prototype.copy() if t.prototype is not None else None
                            for tid, t in self.themes.items()},
            crp_theme_alpha=self.crp_theme.alpha,
            rng_state=self.rng.bit_generator.state,
        )

    def compute_evolution_patch(self, snapshot: EvolutionSnapshot) -> EvolutionPatch:
        """从快照计算演化变化（纯函数，无副作用）。"""
        rng = np.random.default_rng()
        rng.bit_generator.state = snapshot.rng_state

        status_changes = self._compute_story_status_changes(snapshot, rng)
        theme_assignments, new_themes = self._compute_theme_assignments(snapshot, status_changes, rng)

        return EvolutionPatch(
            status_changes=status_changes,
            theme_assignments=theme_assignments,
            new_themes=new_themes,
        )

    def _compute_story_status_changes(
        self, snapshot: EvolutionSnapshot, rng: np.random.Generator
    ) -> Dict[str, str]:
        """计算哪些故事应该改变状态。"""
        status_changes: Dict[str, str] = {}
        ts = now_ts()

        for sid in snapshot.story_ids:
            if snapshot.story_statuses[sid] != "developing":
                continue

            # 计算活动概率
            updated = snapshot.story_updated_ts[sid]
            tau = snapshot.story_gap_means[sid]
            idle = max(0.0, ts - updated)
            p_active = math.exp(-idle / max(tau, EPSILON_PRIOR))

            if rng.random() < p_active:
                continue

            # 确定解决 vs 放弃
            curve = snapshot.story_tension_curves[sid]
            p_resolve = sigmoid(-(curve[-1] - curve[0])) if len(curve) >= 3 else 0.5
            status_changes[sid] = "resolved" if rng.random() < p_resolve else "abandoned"

        return status_changes

    def _compute_theme_assignments(
        self,
        snapshot: EvolutionSnapshot,
        status_changes: Dict[str, str],
        rng: np.random.Generator,
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, np.ndarray]]]:
        """计算新解决故事的主题分配。"""
        theme_assignments: List[Tuple[str, str]] = []
        new_themes: List[Tuple[str, np.ndarray]] = []
        current_theme_counts = dict(snapshot.theme_story_counts)

        for sid, new_status in status_changes.items():
            if new_status != "resolved":
                continue

            centroid = snapshot.story_centroids[sid]
            if centroid is None:
                continue

            choice = self._sample_theme_assignment(
                centroid, snapshot, current_theme_counts, rng
            )

            if choice == "__new__":
                new_theme_id = det_id("theme", sid)
                new_themes.append((new_theme_id, centroid.copy()))
                theme_assignments.append((sid, new_theme_id))
                current_theme_counts[new_theme_id] = 1
            else:
                theme_assignments.append((sid, choice))
                current_theme_counts[choice] = current_theme_counts.get(choice, 0) + 1

        return theme_assignments, new_themes

    def _sample_theme_assignment(
        self,
        centroid: np.ndarray,
        snapshot: EvolutionSnapshot,
        current_counts: Dict[str, int],
        rng: np.random.Generator,
    ) -> str:
        """使用CRP（中国餐厅过程）采样主题分配。"""
        logps: Dict[str, float] = {}

        for tid in snapshot.theme_ids:
            prior = math.log(current_counts.get(tid, 0) + EPSILON_PRIOR)
            prototype = snapshot.theme_prototypes.get(tid)
            if prototype is not None:
                d2 = float(np.dot(centroid - prototype, centroid - prototype))
                logps[tid] = prior - 0.5 * d2
            else:
                logps[tid] = prior

        logps["__new__"] = math.log(snapshot.crp_theme_alpha)

        keys = list(logps.keys())
        probs = softmax([logps[k] for k in keys])
        return rng.choice(keys, p=np.array(probs, dtype=np.float64))

    def apply_evolution_patch(self, patch: EvolutionPatch) -> None:
        """
        原子性地应用计算的演化变化。

        如果需要，应使用适当的锁定调用。

        参数：
            patch: 要应用的EvolutionPatch
        """
        # 1) 应用故事状态变化
        for sid, new_status in patch.status_changes.items():
            if sid in self.stories:
                self.stories[sid].status = new_status

        # 2) 创建新主题
        for theme_id, prototype in patch.new_themes:
            theme = Theme(id=theme_id, created_ts=now_ts(), updated_ts=now_ts())
            theme.prototype = prototype
            self.themes[theme_id] = theme
            self.graph.add_node(theme_id, "theme", theme)
            self.vindex.add(theme_id, prototype, kind="theme")

        # 3) 应用主题分配并编织边
        for sid, tid in patch.theme_assignments:
            if sid not in self.stories or tid not in self.themes:
                continue

            story = self.stories[sid]
            theme = self.themes[tid]

            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()

            # 更新原型（在线均值）
            if story.centroid is not None:
                theme.prototype = self._update_centroid_online(
                    theme.prototype, story.centroid, len(theme.story_ids)
                )

            # 编织边
            self._create_bidirectional_edge(sid, tid, "thematizes", "exemplified_by")

        # 4) 压力管理
        self._pressure_manage()
