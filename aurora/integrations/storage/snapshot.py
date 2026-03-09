from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class Snapshot:
    last_seq: int
    state: Any


class SnapshotStore:
    """Aurora V2 结构化快照存储。"""

    def __init__(self, dirpath: str):
        self.dirpath = dirpath
        os.makedirs(self.dirpath, exist_ok=True)

    def _path(self, last_seq: int) -> str:
        return os.path.join(self.dirpath, f"snapshot_{last_seq}.json")

    def save(self, snap: Snapshot) -> str:
        path = self._path(snap.last_seq)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "last_seq": snap.last_seq,
                    "state": snap.state,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return path

    def latest(self) -> Optional[Tuple[int, Snapshot]]:
        self._ensure_no_legacy_snapshots()

        best_seq = None
        best_path = None
        for fn in os.listdir(self.dirpath):
            if not fn.startswith("snapshot_") or not fn.endswith(".json"):
                continue
            try:
                seq = int(fn[len("snapshot_") : -len(".json")])
            except ValueError:
                continue
            if best_seq is None or seq > best_seq:
                best_seq, best_path = seq, os.path.join(self.dirpath, fn)
        if best_seq is None or best_path is None:
            return None

        with open(best_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        snap = Snapshot(last_seq=int(payload["last_seq"]), state=payload["state"])
        return best_seq, snap

    def _ensure_no_legacy_snapshots(self) -> None:
        for fn in os.listdir(self.dirpath):
            if fn.endswith(".pkl"):
                raise ValueError(
                    "Detected legacy pickle snapshots. Aurora V2 requires a fresh data directory."
                )
