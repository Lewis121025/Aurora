from __future__ import annotations
from typing import Any
from urllib.parse import quote

from aurora.core_math.state import isoformat_utc, utc_now
from aurora.host_runtime.config import AuroraSettings
from aurora.host_runtime.graph_store import AuroraGraphStore
from aurora.substrate_core.engine import AuroraMemoryCore


class AuroraMemory:
    def __init__(
        self,
        user_id: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        self.user_id = user_id or "default"
        self.settings = AuroraSettings.from_env()

        safe_user_id = quote(self.user_id, safe="")
        graph_filename = f"{safe_user_id}.aurora"

        self.store = AuroraGraphStore(
            data_dir=data_dir or self.settings.data_dir,
            filename=graph_filename,
        )

        self.core = AuroraMemoryCore()

        if not self.store.exists():
            sealed = self.core.boot()
            graph = self._create_initial_graph(sealed)
            self.store.save(graph)

    def _create_initial_graph(self, sealed_state: bytes) -> Any:
        import networkx as nx

        g = nx.Graph()
        g.graph["sealed_state"] = sealed_state
        return g

    def _get_sealed_state(self, graph: Any) -> bytes:
        return graph.graph.get("sealed_state", self.core.boot())

    def _save_sealed_state(self, graph: Any, sealed_state: bytes) -> None:
        graph.graph["sealed_state"] = sealed_state
        self.store.save(graph)

    def add(self, text: str, user_id: str | None = None) -> dict[str, Any]:
        if user_id and user_id != self.user_id:
            raise ValueError("Cross-user add not supported in single-instance mode")

        graph = self.store.load()
        sealed = self._get_sealed_state(graph)

        new_sealed, memory_id = self.core.add_memory(sealed, text)

        self._save_sealed_state(graph, new_sealed)

        return {
            "memory_id": memory_id,
            "user_id": self.user_id,
            "text": text,
            "created_at": isoformat_utc(utc_now()),
        }

    def search(
        self, query: str, user_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        if user_id and user_id != self.user_id:
            raise ValueError("Cross-user search not supported in single-instance mode")

        graph = self.store.load()
        sealed = self._get_sealed_state(graph)

        results = self.core.resonance_search(sealed, query, limit)

        return [
            {
                "memory_id": r["memory_id"],
                "text": r["text"],
                "source": r["source"],
                "timestamp": r["timestamp"],
                "score": r["resonance"],
            }
            for r in results
        ]

    def get_all(self, user_id: str | None = None) -> list[dict[str, Any]]:
        if user_id and user_id != self.user_id:
            raise ValueError("Cross-user get_all not supported in single-instance mode")

        graph = self.store.load()
        sealed = self._get_sealed_state(graph)

        memories = self.core.get_all_memories(sealed)

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "source": m["source"],
                "timestamp": m["timestamp"],
                "type": m["type"],
            }
            for m in memories
        ]

    def update(self, memory_id: str, data: str, user_id: str | None = None) -> dict[str, Any]:
        if user_id and user_id != self.user_id:
            raise ValueError("Cross-user update not supported in single-instance mode")

        graph = self.store.load()
        sealed = self._get_sealed_state(graph)

        new_sealed = self.core.update_memory(sealed, memory_id, data)

        self._save_sealed_state(graph, new_sealed)

        return {
            "memory_id": memory_id,
            "text": data,
            "updated_at": isoformat_utc(utc_now()),
        }

    def delete(self, memory_id: str, user_id: str | None = None) -> None:
        if user_id and user_id != self.user_id:
            raise ValueError("Cross-user delete not supported in single-instance mode")

        graph = self.store.load()
        sealed = self._get_sealed_state(graph)

        new_sealed = self.core.delete_memory(sealed, memory_id)

        self._save_sealed_state(graph, new_sealed)

    def delete_all(self, user_id: str | None = None) -> None:
        if user_id and user_id != self.user_id:
            raise ValueError("Cross-user delete_all not supported in single-instance mode")

        sealed = self.core.boot()
        graph = self._create_initial_graph(sealed)
        self.store.save(graph)

    def health(self) -> dict[str, Any]:
        graph_health = self.store.health()

        if graph_health.get("exists"):
            graph = self.store.load()
            sealed = self._get_sealed_state(graph)
            core_health = self.core.health(sealed)
        else:
            core_health = {}

        return {
            "status": "healthy" if graph_health.get("exists") else "initialized",
            "user_id": self.user_id,
            "graph": graph_health,
            "core": core_health,
        }


class MemoryClient:
    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self._memory = AuroraMemory(**kwargs)

    def add(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return self._memory.add(text, **kwargs)

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        return self._memory.search(query, **kwargs)

    def get_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._memory.get_all(**kwargs)

    def update(self, memory_id: str, data: str, **kwargs: Any) -> dict[str, Any]:
        return self._memory.update(memory_id, data, **kwargs)

    def delete(self, memory_id: str, **kwargs: Any) -> None:
        self._memory.delete(memory_id, **kwargs)

    def delete_all(self, **kwargs: Any) -> None:
        self._memory.delete_all(**kwargs)

    def health(self) -> dict[str, Any]:
        return self._memory.health()
