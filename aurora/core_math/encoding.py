from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


@dataclass(frozen=True)
class HashingEncoder:
    dim: int
    seed: int = 7

    def tokenize(self, text: str) -> list[str]:
        raw = text.strip().lower()
        if not raw:
            return []
        tokens = [tok for tok in _TOKEN_RE.findall(raw) if not tok.isspace()]
        chars = [ch for ch in raw if not ch.isspace()]
        for n in (2, 3):
            for idx in range(max(0, len(chars) - n + 1)):
                tokens.append("".join(chars[idx : idx + n]))
        return tokens

    def _signed_bucket(self, token: str) -> tuple[int, int]:
        digest = hashlib.blake2b(
            (token + f"::{self.seed}").encode("utf-8"), digest_size=16
        ).digest()
        index = int.from_bytes(digest[:8], "little") % self.dim
        sign = 1 if (digest[8] & 1) == 0 else -1
        return index, sign

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in self.tokenize(text):
            index, sign = self._signed_bucket(token)
            vec[index] += sign
        return _normalize(vec)
