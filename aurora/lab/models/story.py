"""
AURORA 故事弧模型
======================

中观叙事单元 - 现在围绕关系组织，而不仅仅是语义聚类。

关键洞察：身份在关系中定义。故事不再仅仅是
"语义相似的事件"，而是"关系的叙事"。

主要组织维度现在是 `relationship_with`，而不是语义相似性。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from aurora.utils.time_utils import now_ts


# ─────────────────────────────────────────────────────────────────────────────
# 关系轨迹数据结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RelationshipMoment:
    """
    关系轨迹中的一个时刻。

    捕捉关系在特定时间点的演变方式。
    """

    ts: float  # 时间戳
    event_summary: str  # 发生事件的简要总结
    trust_level: float  # 此时刻的信任水平 [0, 1]
    my_role: str  # 我在此时刻的角色
    quality_delta: float = 0.0  # 关系质量的变化

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "ts": self.ts,
            "event_summary": self.event_summary,
            "trust_level": self.trust_level,
            "my_role": self.my_role,
            "quality_delta": self.quality_delta,
        }

    # 向后兼容别名
    to_dict = to_state_dict

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "RelationshipMoment":
        """从状态字典重建。"""
        return cls(
            ts=d["ts"],
            event_summary=d["event_summary"],
            trust_level=d.get("trust_level", 0.5),
            my_role=d.get("my_role", "assistant"),
            quality_delta=d.get("quality_delta", 0.0),
        )

    # 向后兼容别名
    from_dict = from_state_dict


# -----------------------------------------------------------------------------
# StoryArc Model
# -----------------------------------------------------------------------------

# ─────────────────────────────────────────────────────────────────────────────
# StoryArc 模型
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StoryArc:
    """
    中观叙事单元：现在是关系叙事，而不仅仅是语义聚类。

    关键范式转变：
    - 旧：故事按语义相似性分组
    - 新：故事代表关系，身份在其中涌现

    主要组织维度是 `relationship_with`。
    "我是谁"通过 `my_identity_in_this_relationship` 在每个关系中回答。

    属性：
        id: 唯一标识符
        created_ts: 创建时间戳
        updated_ts: 最后更新时间戳
        plot_ids: 属于此故事的情节 ID 列表

    关系中心：
        relationship_with: 此故事涉及的实体 ID（主键）
        relationship_type: 关系类型（"user"、"system"、"concept"）
        relationship_arc: 关系时刻的轨迹
        my_identity_in_this_relationship: "我在这段关系中是谁"
        lessons_from_relationship: 这段关系教会我什么
        relationship_health: 当前关系的健康度/质量 [0, 1]

    叙事结构：
        setup: 开端 - 故事的起始情境
        rising_action: 发展 - 情节推进事件列表
        climax: 高潮 - 张力最高点
        falling_action: 收尾 - 高潮后的情节
        resolution: 结局 - 最终解决
        central_conflict: 核心冲突
        turning_points: 转折点列表 (时间戳, 描述)
        moral: 寓意 - 从故事中提取的意义

    生成参数：
        centroid: 此故事中情节的平均嵌入
        dist_mean, dist_m2, dist_n: 语义离散度的 Welford 统计
        gap_mean, gap_m2, gap_n: 时间间隔的 Welford 统计

    元数据：
        actor_counts: 每个角色出现次数的计数
        tension_curve: 张力值的历史
        status: 故事生命周期状态
        reference_count: 此故事被引用的次数
    """

    id: str
    created_ts: float
    updated_ts: float
    plot_ids: List[str] = field(default_factory=list)

    # === 关系中心字段（新 - 主要组织维度） ===
    relationship_with: Optional[str] = None  # 实体 ID（组织的主键）
    relationship_type: str = "user"  # "user"、"system"、"concept"
    relationship_arc: List[RelationshipMoment] = field(default_factory=list)
    my_identity_in_this_relationship: str = ""  # "我在这段关系中是谁"
    lessons_from_relationship: List[str] = field(default_factory=list)
    relationship_health: float = 0.5  # 当前关系质量 [0, 1]

    # === 叙事结构 ===
    setup: Optional[str] = None  # 开端 - 故事的起始情境
    rising_action: List[str] = field(default_factory=list)  # 发展 - 情节推进
    climax: Optional[str] = None  # 高潮 - 张力最高点
    falling_action: List[str] = field(default_factory=list)  # 收尾
    resolution: Optional[str] = None  # 结局 - 最终解决

    # === 叙事元素 ===
    central_conflict: Optional[str] = None  # 核心冲突
    turning_points: List[Tuple[float, str]] = field(default_factory=list)  # 转折点 (时间戳, 描述)
    moral: Optional[str] = None  # 寓意 - 从故事中提取的意义

    # 在线生成参数（为兼容性保留）
    centroid: Optional[np.ndarray] = None

    # 语义离散度统计（Welford 算法）
    dist_mean: float = 0.0
    dist_m2: float = 0.0
    dist_n: int = 0

    # 时间间隔统计（Welford 算法）
    gap_mean: float = 0.0
    gap_m2: float = 0.0
    gap_n: int = 0

    actor_counts: Dict[str, int] = field(default_factory=dict)
    tension_curve: List[float] = field(default_factory=list)

    status: Literal["developing", "resolved", "abandoned"] = "developing"
    reference_count: int = 0

    def _update_stats(self, name: str, x: float) -> None:
        """
        使用 Welford 算法更新运行统计。

        参数：
            name: "dist"（距离）或 "gap"（时间间隔）
            x: 新观测值
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
            raise ValueError(f"未知的统计名称: {name}")

    def dist_var(self) -> float:
        """获取距离统计的方差。"""
        return self.dist_m2 / (self.dist_n - 1) if self.dist_n > 1 else 1.0

    def gap_mean_safe(self, default: float = 3600.0) -> float:
        """获取间隔平均值，带安全默认值。"""
        return self.gap_mean if self.gap_n > 0 and self.gap_mean > 0 else default

    def activity_probability(self, ts: Optional[float] = None) -> float:
        """
        故事在学习的时间风险模型下仍然活跃的概率。

        如果故事通常每 ~gap_mean 秒获得一次更新，那么空闲
        >> gap_mean 应该平滑地降低活动概率（不通过固定
        阈值）。

        参数：
            ts: 当前时间戳（默认为现在）

        返回：
            (0, 1) 中的活动概率
        """
        ts = ts or now_ts()
        idle = max(0.0, ts - self.updated_ts)
        tau = self.gap_mean_safe()
        # 指数分布的生存函数：P(active) ~ exp(-idle/tau)
        return math.exp(-idle / max(tau, 1e-6))

    def mass(self) -> float:
        """
        故事级别的涌现重要性。

        返回：
            结合新鲜度、大小和引用的质量值
        """
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        size = math.log1p(len(self.plot_ids))
        return freshness * (size + math.log1p(self.reference_count + 1))

    # ─────────────────────────────────────────────────────────────────────────
    # 关系中心方法
    # ─────────────────────────────────────────────────────────────────────────

    def is_relationship_story(self) -> bool:
        """检查此故事是否围绕关系组织。"""
        return self.relationship_with is not None

    def add_relationship_moment(
        self,
        event_summary: str,
        trust_level: float,
        my_role: str,
        quality_delta: float = 0.0,
        ts: Optional[float] = None,
    ) -> None:
        """向关系轨迹添加一个时刻。"""
        moment = RelationshipMoment(
            ts=ts or now_ts(),
            event_summary=event_summary,
            trust_level=trust_level,
            my_role=my_role,
            quality_delta=quality_delta,
        )
        self.relationship_arc.append(moment)

        # 根据质量增量更新关系健康度
        self.relationship_health = max(
            0.0, min(1.0, self.relationship_health + quality_delta * 0.1)
        )

    def get_trust_trend(self, window: int = 10) -> float:
        """
        获取最近交互中信任水平的趋势。

        返回：
            如果信任增加则为正，如果减少则为负。
        """
        if len(self.relationship_arc) < 2:
            return 0.0

        recent = self.relationship_arc[-window:]
        if len(recent) < 2:
            return 0.0

        # 简单线性趋势
        first_half = recent[: len(recent) // 2]
        second_half = recent[len(recent) // 2 :]

        avg_first = sum(m.trust_level for m in first_half) / len(first_half)
        avg_second = sum(m.trust_level for m in second_half) / len(second_half)

        return avg_second - avg_first

    def get_role_consistency(self, window: int = 10) -> float:
        """
        获取我在这段关系中角色的一致性。

        返回：
            如果完全一致则为 1.0，如果角色变化则更低。
        """
        if len(self.relationship_arc) < 2:
            return 1.0

        recent = self.relationship_arc[-window:]
        roles = [m.my_role for m in recent]

        if not roles:
            return 1.0

        # 使用 Counter 计算最常见的角色
        role_counts = Counter(roles)
        max_count = role_counts.most_common(1)[0][1] if role_counts else 0
        return max_count / len(roles)

    # ─────────────────────────────────────────────────────────────────────────
    # 时间方法（时间作为一等公民）
    # ─────────────────────────────────────────────────────────────────────────

    def get_temporal_span(self, plots_dict: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """获取此故事的时间跨度。

        时间作为一等公民：故事的时间跨度对于
        理解其叙事弧和时间背景至关重要。

        参数：
            plots_dict: 可选的 plot_id -> Plot 字典，用于精确时间戳。
                        如果未提供，使用 created_ts 和 updated_ts。

        返回：
            (最早时间戳, 最晚时间戳) 的元组。
            如果没有情节，返回 (created_ts, updated_ts)。
        """
        if not self.plot_ids:
            return (self.created_ts, self.updated_ts)

        if plots_dict is None:
            # 回退到故事时间戳
            return (self.created_ts, self.updated_ts)

        # 从实际情节获取时间戳
        timestamps = []
        for pid in self.plot_ids:
            plot = plots_dict.get(pid)
            if plot is not None:
                ts = getattr(plot, "ts", None)
                if ts is not None:
                    timestamps.append(ts)

        if not timestamps:
            return (self.created_ts, self.updated_ts)

        return (min(timestamps), max(timestamps))

    def get_temporal_narrative(
        self, plots_dict: Optional[Dict[str, Any]] = None, locale: str = "zh"
    ) -> str:
        """生成此故事的时间叙事摘要。

        时间作为一等公民：生成故事时间维度的人类可读描述，
        帮助用户理解事件的时间线。

        示例输出：
        - 中文："从2024年1月开始，用户和我讨论了Python学习，历经3个月，共12次交互..."
        - 英文："Starting from January 2024, discussed Python learning over 3 months, 12 interactions..."

        参数：
            plots_dict: 可选的 plot_id -> Plot 字典，用于事件详情。
            locale: 叙事的语言（"zh" 为中文，"en" 为英文）。

        返回：
            故事时间弧的自然语言描述。
        """
        import datetime

        # 获取时间跨度
        start_ts, end_ts = self.get_temporal_span(plots_dict)

        # 转换为 datetime
        start_dt = datetime.datetime.fromtimestamp(start_ts)

        # 计算持续时间
        duration_days = (end_ts - start_ts) / (24 * 3600)
        interaction_count = len(self.plot_ids)

        # 格式化日期
        if locale == "zh":
            start_str = start_dt.strftime("%Y年%m月%d日")

            # 构建叙事
            parts = []

            if self.relationship_with:
                parts.append(f"与{self.relationship_with}的关系故事")
            else:
                parts.append(f"故事{self.id[:8]}")

            parts.append(f"从{start_str}开始")

            if duration_days < 1:
                parts.append("在同一天内")
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

            # 如果可用，添加关系背景
            if self.my_identity_in_this_relationship:
                parts.append(f"我作为{self.my_identity_in_this_relationship}")

            # 添加叙事阶段
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

        else:  # 英文
            start_str = start_dt.strftime("%B %d, %Y")

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
        """计算交互的时间密度。

        更高的密度意味着时间跨度内的交互更频繁。

        参数：
            plots_dict: 可选的 plot_id -> Plot 字典。

        返回：
            每天的交互数，如果没有时间跨度则为 0。
        """
        start_ts, end_ts = self.get_temporal_span(plots_dict)
        duration_days = (end_ts - start_ts) / (24 * 3600)

        if duration_days < 0.001:  # 少于一分钟
            return float(len(self.plot_ids))  # 所有交互在单一时刻

        return len(self.plot_ids) / duration_days

    def update_identity_in_relationship(self, new_identity: str) -> None:
        """更新我在这段关系中的身份。"""
        self.my_identity_in_this_relationship = new_identity

    def add_lesson(self, lesson: str) -> None:
        """添加从这段关系中学到的课程。"""
        if lesson not in self.lessons_from_relationship:
            self.lessons_from_relationship.append(lesson)

    def get_current_trust(self) -> float:
        """获取这段关系中的当前信任水平。"""
        if not self.relationship_arc:
            return 0.5  # 中立默认值
        return self.relationship_arc[-1].trust_level

    def to_relationship_narrative(self) -> str:
        """生成这段关系的自然语言叙事。"""
        if not self.is_relationship_story():
            return f"故事 {self.id}（不是关系故事）"

        parts = []

        # 关系身份
        if self.my_identity_in_this_relationship:
            parts.append(
                f"在与 {self.relationship_with} 的关系中，我是{self.my_identity_in_this_relationship}。"
            )

        # 信任和健康度
        trust = self.get_current_trust()
        if trust > 0.7:
            parts.append("我们建立了良好的信任关系。")
        elif trust > 0.4:
            parts.append("我们的关系正在发展中。")
        else:
            parts.append("我们的关系还需要培养。")

        # 课程
        if self.lessons_from_relationship:
            lessons = "、".join(self.lessons_from_relationship[:3])
            parts.append(f"这段关系教会我：{lessons}。")

        return "".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # 叙事结构方法
    # ─────────────────────────────────────────────────────────────────────────

    def detect_climax(self, threshold_percentile: float = 0.9) -> Optional[int]:
        """
        基于 tension_curve 检测高潮点。

        高潮被识别为超过所有张力值的阈值百分位数的最大张力点。

        参数：
            threshold_percentile: 考虑为高潮的百分位数阈值
                                  （默认 0.9 = 90 百分位数）

        返回：
            tension_curve 中高潮点的索引，如果没有明确的高潮则为 None。
        """
        if not self.tension_curve or len(self.tension_curve) < 3:
            return None

        tensions = np.array(self.tension_curve)
        max_idx = int(np.argmax(tensions))
        max_tension = tensions[max_idx]

        # 检查最大张力是否超过阈值百分位数
        threshold = np.percentile(tensions, threshold_percentile * 100)
        if max_tension >= threshold:
            return max_idx

        return None

    def extract_moral(self) -> Optional[str]:
        """
        从此故事中提取寓意/意义。

        这是一个占位符实现，基于故事的结构和学到的课程生成寓意。
        在生产环境中，这将由基于 LLM 的意义提取增强。

        返回：
            描述寓意/意义的字符串，如果数据不足则为 None。
        """
        # 如果已设置，返回它
        if self.moral:
            return self.moral

        # 从可用信息合成
        parts = []

        # 从关系课程
        if self.lessons_from_relationship:
            parts.append(f"从关系中学到：{self.lessons_from_relationship[0]}")

        # 从核心冲突
        if self.central_conflict and self.resolution:
            parts.append(f"面对「{self.central_conflict}」，最终{self.resolution}")

        # 从张力模式
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
        向叙事添加转折点。

        转折点是故事方向改变的重要时刻。
        它们与时间戳一起存储以便按时间顺序排列。

        参数：
            ts: 转折点的时间戳
            description: 描述发生了什么变化
        """
        self.turning_points.append((ts, description))
        # 按时间戳排序
        self.turning_points.sort(key=lambda x: x[0])

    def get_narrative_phase(
        self,
    ) -> Literal["setup", "rising", "climax", "falling", "resolution", "unknown"]:
        """
        根据故事结构确定当前叙事阶段。

        返回：
            叙事弧的当前阶段。
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
        计算叙事结构的完整程度。

        返回：
            从 0.0 到 1.0 的分数，表示叙事完整性。
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
        生成此故事的结构化叙事摘要。

        返回：
            总结叙事弧的格式化字符串。
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
            return f"故事 {self.id}（叙事结构待完善）"

        return "\n".join(parts)

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化为 JSON 兼容的字典。"""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "plot_ids": self.plot_ids,
            # 关系中心字段
            "relationship_with": self.relationship_with,
            "relationship_type": self.relationship_type,
            "relationship_arc": [m.to_state_dict() for m in self.relationship_arc],
            "my_identity_in_this_relationship": self.my_identity_in_this_relationship,
            "lessons_from_relationship": self.lessons_from_relationship,
            "relationship_health": self.relationship_health,
            # 叙事结构
            "setup": self.setup,
            "rising_action": self.rising_action,
            "climax": self.climax,
            "falling_action": self.falling_action,
            "resolution": self.resolution,
            # 叙事元素
            "central_conflict": self.central_conflict,
            "turning_points": self.turning_points,  # List[Tuple[float, str]] 是 JSON 可序列化的
            "moral": self.moral,
            # 生成参数
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
        """从状态字典重建。

        向后兼容：处理没有叙事结构字段的旧数据。
        """
        centroid = d.get("centroid")

        # 解析关系弧
        relationship_arc = []
        if "relationship_arc" in d:
            relationship_arc = [
                RelationshipMoment.from_state_dict(m) for m in d["relationship_arc"]
            ]

        # 解析转折点 - 将列表转换回元组以保持类型一致性
        raw_turning_points = d.get("turning_points", [])
        turning_points: List[Tuple[float, str]] = (
            [(float(tp[0]), str(tp[1])) for tp in raw_turning_points] if raw_turning_points else []
        )

        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            plot_ids=d.get("plot_ids", []),
            # 关系中心字段
            relationship_with=d.get("relationship_with"),
            relationship_type=d.get("relationship_type", "user"),
            relationship_arc=relationship_arc,
            my_identity_in_this_relationship=d.get("my_identity_in_this_relationship", ""),
            lessons_from_relationship=d.get("lessons_from_relationship", []),
            relationship_health=d.get("relationship_health", 0.5),
            # 叙事结构（向后兼容默认值）
            setup=d.get("setup"),
            rising_action=d.get("rising_action", []),
            climax=d.get("climax"),
            falling_action=d.get("falling_action", []),
            resolution=d.get("resolution"),
            # 叙事元素（向后兼容默认值）
            central_conflict=d.get("central_conflict"),
            turning_points=turning_points,
            moral=d.get("moral"),
            # 生成参数
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
