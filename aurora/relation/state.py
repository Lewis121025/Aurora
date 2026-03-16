"""关系状态模块。

定义关系状态（RelationalState），即 Aurora 的"灵魂"：
- intimacy_level: 亲密度
- current_vibe: 当前氛围
- interaction_rules: 交互规则
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RelationalState:
    """关系状态。

    存储关系的核心状态，在每次对话时全量挂载到 System Prompt。
    这就是 Aurora 拟人化的"主脑"。

    Attributes:
        intimacy_level: 亲密度（0-10）。
        current_vibe: 当前氛围描述。
        interaction_rules: 交互规则列表。
        last_distilled_at: 最后蒸馏时间戳。
    """

    intimacy_level: int = 5
    current_vibe: str = "中性"
    interaction_rules: list[str] = field(default_factory=list)
    last_distilled_at: float = 0.0

    def to_prompt_segment(self) -> str:
        """转换为 System Prompt 片段。

        Returns:
            格式化的状态字符串。
        """
        rules = "\n  ".join(f"- {r}" for r in self.interaction_rules) or "  （无特殊规则）"
        return f"""[RELATIONAL_STATE]
intimacy_level: {self.intimacy_level}
current_vibe: "{self.current_vibe}"
interaction_rules:
  {rules}"""

    def apply_patch(
        self,
        intimacy_delta: int | None = None,
        vibe: str | None = None,
        new_rules: list[str] | None = None,
    ) -> None:
        """应用状态补丁。

        Args:
            intimacy_delta: 亲密度变化。
            vibe: 新氛围。
            new_rules: 新规则列表（追加）。
        """
        if intimacy_delta is not None:
            self.intimacy_level = max(0, min(10, self.intimacy_level + intimacy_delta))
        if vibe is not None:
            self.current_vibe = vibe
        if new_rules:
            for rule in new_rules:
                if rule not in self.interaction_rules:
                    self.interaction_rules.append(rule)
