"""记忆片段模块。

定义记忆的基本单位 Fragment，包含：
- 表面文本（surface）
- 标签（tags）
- 生动度（vividness）、显著性（salience）、未解决度（unresolvedness）
- 关联的线程和记忆结
- 时间戳和激活计数
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from aurora.runtime.contracts import clamp


@dataclass(frozen=True, slots=True)
class Fragment:
    """记忆片段。

    记忆的基本存储单位，记录一次交互或事件的压缩表示。
    通过显著性、生动度、未解决度等指标动态管理记忆的活跃程度。

    Attributes:
        fragment_id: 片段唯一 ID。
        relation_id: 所属关系 ID。
        turn_id: 关联的转换 ID（可选）。
        surface: 片段表面文本（压缩后的记忆内容）。
        tags: 标签列表。
        vividness: 生动度（0.0–1.0），越高表示记忆越清晰。
        salience: 显著性（0.0–1.0），越高表示越容易被检索。
        unresolvedness: 未解决度（0.0–1.0），越高表示越需要整合。
        thread_ids: 关联的记忆线程 ID 列表。
        knot_ids: 关联的记忆结 ID 列表。
        created_at: 创建时间戳。
        last_touched_at: 最后触碰时间戳。
        activation_count: 激活次数。
    """

    fragment_id: str
    relation_id: str
    turn_id: str | None
    surface: str
    tags: tuple[str, ...]
    vividness: float
    salience: float
    unresolvedness: float
    thread_ids: tuple[str, ...]
    knot_ids: tuple[str, ...]
    created_at: float
    last_touched_at: float
    activation_count: int = 0

    def touched(
        self,
        at: float,
        delta_salience: float = 0.08,
        delta_unresolved: float = 0.0,
    ) -> "Fragment":
        """触碰片段，更新其状态。

        增加显著性和激活计数，可选调整未解决度。
        用于记录片段被激活（检索、关联等）的事件。

        Args:
            at: 触碰时间戳。
            delta_salience: 显著性增量，默认 0.08。
            delta_unresolved: 未解决度增量，默认 0.0。

        Returns:
            Fragment: 新片段实例（原实例不变）。
        """
        return replace(
            self,
            salience=clamp(self.salience + delta_salience),
            unresolvedness=clamp(self.unresolvedness + delta_unresolved),
            last_touched_at=at,
            activation_count=self.activation_count + 1,
        )
