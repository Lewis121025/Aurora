from __future__ import annotations
import hashlib
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx


MAGIC = b"AURG"


class AuroraGraphStore:
    def __init__(self, data_dir: str | Path | None = None, filename: str = "memory.aurora") -> None:
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / ".aurora_seed_v1"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.data_dir / filename

    def exists(self) -> bool:
        return self.filepath.exists() and self.filepath.stat().st_size > 0

    def _load_graph_nosig(self) -> nx.Graph:
        if not self.exists():
            return nx.Graph()
        with open(self.filepath, "rb") as f:
            data = f.read()
        if not data.startswith(MAGIC):
            raise ValueError("Invalid graph file prefix.")
        checksum = data[len(MAGIC) : len(MAGIC) + 64]
        payload = data[len(MAGIC) + 64 :]
        if hashlib.sha256(payload).hexdigest().encode("ascii") != checksum:
            raise ValueError("Graph checksum mismatch.")
        return pickle.loads(payload)

    def _save_graph_nosig(self, graph: nx.Graph) -> None:
        payload = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
        checksum = hashlib.sha256(payload).hexdigest().encode("ascii")
        data = MAGIC + checksum + payload
        with tempfile.NamedTemporaryFile(dir=self.data_dir, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        os.replace(tmp_path, self.filepath)

    def load(self) -> nx.Graph:
        return self._load_graph_nosig()

    def save(self, graph: nx.Graph) -> None:
        self._save_graph_nosig(graph)

    def delete(self) -> None:
        if self.filepath.exists():
            self.filepath.unlink()

    def health(self) -> dict[str, Any]:
        if not self.exists():
            return {
                "exists": False,
                "node_count": 0,
                "edge_count": 0,
            }
        g = self.load()
        return {
            "exists": True,
            "node_count": g.number_of_nodes(),
            "edge_count": g.number_of_edges(),
            "filepath": str(self.filepath),
        }
