"""身份解析模块。

将会话（session）与长期关系（relation）解耦。
"""

from __future__ import annotations

from uuid import uuid4


class IdentityResolver:
    """身份解析器。

    管理 session_id → relation_id 的绑定。

    Attributes:
        _bindings: session_id → relation_id 映射。
    """

    def __init__(self) -> None:
        self._bindings: dict[str, str] = {}

    def resolve(self, session_id: str) -> str:
        """将 session_id 解析为 relation_id。

        Args:
            session_id: 会话 ID。

        Returns:
            relation_id: 关系 ID。
        """
        if session_id in self._bindings:
            return self._bindings[session_id]
        relation_id = f"rel_{uuid4().hex[:12]}"
        self._bindings[session_id] = relation_id
        return relation_id

    def bind(self, session_id: str, relation_id: str) -> None:
        """显式绑定 session 到已有 relation。"""
        self._bindings[session_id] = relation_id
