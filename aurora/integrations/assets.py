from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from urllib.parse import urlparse, unquote


class AssetResolver:
    """Resolve public image URIs into remote-safe values without persisting raw bytes."""

    ALLOWED_SCHEMES = {"http", "https", "data", "file"}

    def validate_uri(self, uri: str) -> str:
        value = str(uri).strip()
        if not value:
            raise ValueError("Image URI must not be empty")
        parsed = urlparse(value)
        if parsed.scheme not in self.ALLOWED_SCHEMES:
            raise ValueError(
                f"Unsupported image URI scheme: {parsed.scheme or '<missing>'}. "
                "Supported schemes are http, https, data, and file."
            )
        if parsed.scheme == "file" and not parsed.path:
            raise ValueError("file: image URI must include a path")
        if parsed.scheme in {"http", "https"} and not parsed.netloc:
            raise ValueError("http(s) image URI must include a host")
        if parsed.scheme == "data" and "," not in value:
            raise ValueError("data: image URI must contain an inline payload")
        return value

    def resolve_for_remote(self, uri: str, *, mime_type: str | None = None) -> str:
        validated = self.validate_uri(uri)
        if validated.startswith(("http://", "https://", "data:")):
            return validated
        return self._file_uri_to_data_uri(validated, mime_type=mime_type)

    def _file_uri_to_data_uri(self, uri: str, *, mime_type: str | None = None) -> str:
        parsed = urlparse(uri)
        path = Path(unquote(parsed.path))
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"Image file URI does not point to a file: {path}")
        content_type = mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
