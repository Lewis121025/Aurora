"""Local blob store for raw payloads and residuals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BlobStore:
    """Filesystem-backed blob store.

    Aurora stores raw packet payloads as plain text files and uses stable blob
    references on the metadata side.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_text(self, blob_id: str, text: str) -> str:
        path = self._text_path(blob_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def put_json(self, blob_id: str, payload: Any) -> str:
        path = self._json_path(blob_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return str(path)

    def read_text(self, blob_ref: str) -> str:
        return Path(blob_ref).read_text(encoding="utf-8")

    def read_json(self, blob_ref: str) -> Any:
        return json.loads(self.read_text(blob_ref))

    def put(self, blob_ref: str, data: bytes) -> Path:
        path = self._binary_path(blob_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def get(self, blob_ref: str) -> bytes:
        return self._binary_path(blob_ref).read_bytes()

    def get_json(self, blob_ref: str) -> Any:
        return json.loads(self.get(blob_ref).decode("utf-8"))

    def exists(self, blob_ref: str) -> bool:
        return self._binary_path(blob_ref).exists() or Path(blob_ref).exists()

    def _text_path(self, blob_id: str) -> Path:
        return self.root / f"{self._sanitize(blob_id)}.txt"

    def _json_path(self, blob_id: str) -> Path:
        return self.root / f"{self._sanitize(blob_id)}.json"

    def _binary_path(self, blob_ref: str) -> Path:
        candidate = Path(blob_ref)
        if candidate.exists():
            return candidate
        return self.root / self._sanitize(blob_ref)

    def _sanitize(self, blob_ref: str) -> str:
        return blob_ref.replace(":", "_").replace("/", "_")


__all__ = ["BlobStore"]
