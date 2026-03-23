"""Local text blob store for raw payloads and residuals."""

from __future__ import annotations

from pathlib import Path


class BlobStore:
    """Filesystem-backed text blob store.

    Aurora stores raw packet payloads as plain text files and keeps the
    resulting file paths in packet metadata.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_text(self, blob_id: str, text: str) -> str:
        path = self._text_path(blob_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def read_text(self, blob_ref: str) -> str:
        return Path(blob_ref).read_text(encoding="utf-8")

    def _text_path(self, blob_id: str) -> Path:
        return self.root / f"{self._sanitize(blob_id)}.txt"

    def _sanitize(self, blob_ref: str) -> str:
        return blob_ref.replace(":", "_").replace("/", "_")


__all__ = ["BlobStore"]
