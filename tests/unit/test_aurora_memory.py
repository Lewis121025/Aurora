from __future__ import annotations

from pathlib import Path

from aurora.memory import AuroraMemory, MemoryClient


class TestAuroraMemory:
    def test_add_memory(self, tmp_path: Path) -> None:
        memory = AuroraMemory(data_dir=str(tmp_path))
        result = memory.add("I love coffee in the morning")

        assert "memory_id" in result
        assert result["text"] == "I love coffee in the morning"
        assert result["user_id"] == "default"
        assert "created_at" in result

    def test_search_memory(self, tmp_path: Path) -> None:
        memory = AuroraMemory(data_dir=str(tmp_path))

        memory.add("I love coffee in the morning")
        memory.add("I prefer tea in the evening")

        results = memory.search("coffee")

        assert len(results) > 0
        assert any("coffee" in r["text"].lower() for r in results)

    def test_get_all_memories(self, tmp_path: Path) -> None:
        memory = AuroraMemory(data_dir=str(tmp_path))

        memory.add("First memory")
        memory.add("Second memory")

        all_memories = memory.get_all()

        assert len(all_memories) >= 2

    def test_update_memory(self, tmp_path: Path) -> None:
        memory = AuroraMemory(data_dir=str(tmp_path))

        result = memory.add("Original text")
        memory_id = result["memory_id"]

        updated = memory.update(memory_id, "Updated text")

        assert updated["text"] == "Updated text"

    def test_delete_memory(self, tmp_path: Path) -> None:
        memory = AuroraMemory(data_dir=str(tmp_path))

        result = memory.add("Memory to delete")
        memory_id = result["memory_id"]

        memory.delete(memory_id)

        all_memories = memory.get_all()
        assert not any(m["memory_id"] == memory_id for m in all_memories)

    def test_delete_all(self, tmp_path: Path) -> None:
        memory = AuroraMemory(data_dir=str(tmp_path))

        memory.add("Memory 1")
        memory.add("Memory 2")

        memory.delete_all()

        all_memories = memory.get_all()
        assert len(all_memories) == 0

    def test_health(self, tmp_path: Path) -> None:
        memory = AuroraMemory(data_dir=str(tmp_path))

        health = memory.health()

        assert "status" in health
        assert "graph" in health
        assert "core" in health


class TestMemoryClient:
    def test_memory_client_add(self, tmp_path: Path) -> None:
        client = MemoryClient(data_dir=str(tmp_path))

        result = client.add("Test memory")

        assert "memory_id" in result

    def test_memory_client_search(self, tmp_path: Path) -> None:
        client = MemoryClient(data_dir=str(tmp_path))

        client.add("Searchable memory")
        results = client.search("Searchable")

        assert len(results) > 0

    def test_memory_client_delete_all(self, tmp_path: Path) -> None:
        client = MemoryClient(data_dir=str(tmp_path))

        client.add("Memory one")
        client.add("Memory two")
        client.delete_all()

        all_mem = client.get_all()
        assert len(all_mem) == 0
