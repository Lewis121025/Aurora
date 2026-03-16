"""运行时状态模块。

定义 Aurora 运行时状态（RuntimeState），管理：
- 活跃会话
- 轮数累积
- 最后交互时间
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RuntimeState:
    """Aurora 运行时状态。

    Attributes:
        session_turn_counts: 会话 -> 轮数映射。
        last_interaction_at: 最后交互时间戳。
        active_sessions: 活跃会话集合。
    """

    session_turn_counts: dict[str, int] = field(default_factory=dict)
    last_interaction_at: float = 0.0
    active_sessions: set[str] = field(default_factory=set)

    def record_turn(self, session_id: str, now_ts: float) -> int:
        """记录一次对话轮次。

        Args:
            session_id: 会话 ID。
            now_ts: 当前时间戳。

        Returns:
            累计轮数。
        """
        self.last_interaction_at = now_ts
        self.active_sessions.add(session_id)
        count = self.session_turn_counts.get(session_id, 0) + 1
        self.session_turn_counts[session_id] = count
        return count

    def get_turn_count(self, session_id: str) -> int:
        """获取会话轮数。

        Args:
            session_id: 会话 ID。

        Returns:
            轮数。
        """
        return self.session_turn_counts.get(session_id, 0)

    def is_idle(self, now_ts: float, timeout_seconds: float) -> bool:
        """判断是否空闲。

        Args:
            now_ts: 当前时间戳。
            timeout_seconds: 超时阈值（秒）。

        Returns:
            是否空闲。
        """
        if self.last_interaction_at == 0.0:
            return True
        return (now_ts - self.last_interaction_at) >= timeout_seconds

    def clear_session(self, session_id: str) -> None:
        """清除会话数据。

        Args:
            session_id: 会话 ID。
        """
        self.active_sessions.discard(session_id)
        self.session_turn_counts.pop(session_id, None)
