from __future__ import annotations

import gzip
import hashlib
import json

from aurora.core_math.state import SealedState

_MAGIC = b"AUR1"


def seal_state(state: SealedState) -> bytes:
    payload = json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(payload).hexdigest().encode("ascii")
    return _MAGIC + checksum + gzip.compress(payload)


def unseal_state(blob: bytes) -> SealedState:
    if not blob.startswith(_MAGIC):
        raise ValueError("Invalid sealed blob prefix.")
    checksum = blob[len(_MAGIC) : len(_MAGIC) + 64]
    payload = gzip.decompress(blob[len(_MAGIC) + 64 :])
    if hashlib.sha256(payload).hexdigest().encode("ascii") != checksum:
        raise ValueError("Sealed blob checksum mismatch.")
    return SealedState.from_dict(json.loads(payload.decode("utf-8")))
