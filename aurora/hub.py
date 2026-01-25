from __future__ import annotations

import collections
import threading
from typing import Dict, Optional

from aurora.config import AuroraSettings
from aurora.llm.provider import LLMProvider
from aurora.service import AuroraTenant


class AuroraHub:
    """Multi-tenant router with LRU eviction.

    Use case:
      - one memory instance per user_id
      - cap number of loaded tenants to avoid unbounded RAM

    Eviction policy:
      - when exceeding cap, evict least-recently-used tenant
      - eviction triggers snapshot (tenant already snapshots periodically, but we can add forced snapshot)
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
