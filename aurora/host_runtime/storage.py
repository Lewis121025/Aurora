from __future__ import annotations

import os
import tempfile
from pathlib import Path


class SealedBlobStore:
    def __init__(self, root: str, sealed_state_filename: str, alarm_filename: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / sealed_state_filename
        self.alarm_path = self.root / alarm_filename

    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> bytes:
        return self.state_path.read_bytes()

    def save(self, blob: bytes) -> None:
        self._atomic_write_bytes(self.state_path, blob)

    def read_alarm(self) -> str | None:
        if not self.alarm_path.exists():
            return None
        raw = self.alarm_path.read_text(encoding="utf-8").strip()
        return raw or None

    def write_alarm(self, when: str | None) -> None:
        payload = "" if when is None else when
        self._atomic_write_text(self.alarm_path, payload)

    def _atomic_write_bytes(self, target: Path, payload: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.root, prefix=f"{target.stem}.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _atomic_write_text(self, target: Path, payload: str) -> None:
        self._atomic_write_bytes(target, payload.encode("utf-8"))
