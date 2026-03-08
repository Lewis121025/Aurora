from __future__ import annotations

import collections
import threading
from typing import Dict, Optional

from aurora.runtime.settings import AuroraSettings
from aurora.integrations.llm.provider import LLMProvider
from aurora.runtime.tenant import AuroraTenant


class AuroraHub:
    """具有 LRU 驱逐的多租户路由器。

    使用场景：
      - 每个 user_id 一个内存实例
      - 限制加载的租户数量以避免无限制的 RAM 占用

    驱逐策略：
      - 超过上限时，驱逐最近最少使用的租户
      - 驱逐触发快照（租户已定期快照，但我们可以添加强制快照）
    """

    def __init__(self, settings: AuroraSettings, llm: Optional[LLMProvider] = None):
        self.settings = settings
        self.llm = llm
        self._lock = threading.RLock()
        self._tenants: Dict[str, AuroraTenant] = {}
        self._lru = collections.OrderedDict()  # user_id -> None

    def tenant(self, user_id: str) -> AuroraTenant:
        with self._lock:
            if user_id in self._tenants:
                self._touch(user_id)
                return self._tenants[user_id]

            t = AuroraTenant(user_id=user_id, settings=self.settings, llm=self.llm)
            self._tenants[user_id] = t
            self._touch(user_id)
            self._evict_if_needed()
            return t

    def _touch(self, user_id: str) -> None:
        if user_id in self._lru:
            self._lru.move_to_end(user_id)
        else:
            self._lru[user_id] = None

    def _evict_if_needed(self) -> None:
        cap = int(self.settings.tenant_max_loaded)
        while cap > 0 and len(self._tenants) > cap:
            old_user_id, _ = self._lru.popitem(last=False)
            self._tenants.pop(old_user_id, None)
