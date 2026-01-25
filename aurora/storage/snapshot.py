from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class Snapshot:
    last_seq: int
    state: Any


class SnapshotStore:
    """Simple snapshot store using pickle.

    Security note:
      - Only load snapshots you trust (pickle is unsafe for untrusted blobs).
      - For untrusted environments, serialize to JSON and reconstruct.
    """

    def __init__(self, dirpath: str):
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)

    def _path(self, last_seq: int) -> str:
        return os.path.join(self.dirpath, f"snapshot_{last_seq}.pkl")

    def save(self, snap: Snapshot) -> str:
        path = self._path(snap.last_seq)
        with open(path, "wb") as f:
            pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def latest(self) -> Optional[Tuple[int, Snapshot]]:
        # pick highest seq
        best_seq = None
        best_path = None
        for fn in os.listdir(self.dirpath):
            if not fn.startswith("snapshot_") or not fn.endswith(".pkl"):
                continue
            try:
                seq = int(fn[len("snapshot_") : -len(".pkl")])
            except ValueError:
                continue
            if best_seq is None or seq > best_seq:
                best_seq, best_path = seq, os.path.join(self.dirpath, fn)
        if best_seq is None or best_path is None:
            return None
        with open(best_path, "rb") as f:
            snap: Snapshot = pickle.load(f)
        return best_seq, snap
