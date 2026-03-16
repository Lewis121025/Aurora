"""本体代谢状态模块。

定义 Aurora 的代谢状态（MetabolicState），管理 awake/doze/sleep 三态转换
及相关生理需求（如睡眠需求）。
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora.runtime.contracts import Phase, clamp


@dataclass(slots=True)
class MetabolicState:
    """Aurora 代谢状态。

    维护当前相位、睡眠需求、活跃关系/记忆结等信息。
    """

    phase: Phase = Phase.AWAKE
    """当前相位（awake/doze/sleep）。"""

    sleep_need: float = 0.0
    """睡眠需求累积值，范围 0.0–1.0。"""

    active_relation_ids: tuple[str, ...] = ()
    """活跃关系 ID 列表（最多保留 4 个）。"""

    active_knot_ids: tuple[str, ...] = ()
    """活跃记忆结 ID 列表（最多保留 8 个）。"""

    pending_sleep_relation_ids: tuple[str, ...] = ()
    """等待进入 sleep 处理的关系 ID 列表。"""

    last_transition_at: float = 0.0
    """最后一次相位转换的时间戳。"""

    def enter_phase(self, next_phase: Phase, now_ts: float) -> None:
        """进入新相位。

        仅在相位实际变化时更新，并记录转换时间戳。
        """
        if self.phase != next_phase:
            self.phase = next_phase
            self.last_transition_at = now_ts

    def queue_relation_for_sleep(self, relation_id: str) -> None:
        """将关系加入 sleep 处理队列。

        用于在 sleep 阶段处理需要整合的关系。
        """
        if relation_id not in self.pending_sleep_relation_ids:
            self.pending_sleep_relation_ids = self.pending_sleep_relation_ids + (relation_id,)

    def set_active_relation(self, relation_id: str) -> None:
        """设置活跃关系。

        若关系已存在则移至末尾（LRU 行为），否则追加。
        最多保留 4 个活跃关系。
        """
        if relation_id in self.active_relation_ids:
            self.active_relation_ids = tuple(
                [item for item in self.active_relation_ids if item != relation_id] + [relation_id]
            )
            return
        self.active_relation_ids = tuple([*self.active_relation_ids, relation_id][-4:])

    def set_active_knots(self, knot_ids: tuple[str, ...]) -> None:
        """设置活跃记忆结。

        最多保留 8 个活跃记忆结。
        """
        self.active_knot_ids = tuple(knot_ids[-8:])

    def settle_after_sleep(self) -> None:
        """sleep 后的状态整理。

        清空待处理关系队列，减少睡眠需求（固定减少 0.45）。
        """
        self.pending_sleep_relation_ids = ()
        self.sleep_need = clamp(self.sleep_need - 0.45)

    def bump_sleep_need(self, amount: float) -> None:
        """增加睡眠需求。

        Args:
            amount: 增加量，最终值会被 clamp 到 0.0–1.0。
        """
        self.sleep_need = clamp(self.sleep_need + amount)
