from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


class Encoder(Protocol):
    dim: int
    def encode(self, text: str) -> np.ndarray: ...


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


class SemanticEncoder:
    def __init__(self, model_id: str = "Xenova/bge-small-zh-v1.5", use_quantized: bool = True):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        
        self._model_id = model_id
        path = snapshot_download(repo_id=model_id, allow_patterns=["onnx/*", "tokenizer*"])
        
        self.tokenizer = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        
        model_name = "model_quantized.onnx" if use_quantized else "model.onnx"
        model_path = os.path.join(path, "onnx", model_name)
        
        # CPU Execution Provider is fine for this micro model
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        # BAAI/bge models typically produce 512, 768 or 1024 dimensional embeddings depending on version
        # bge-small series uses 512 dimensions
        self.dim = 512
        
        # Max sequence length for typical BERT-style models
        self.max_length = 512

    def encode(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(self.dim, dtype=np.float64)
            
        # Add CLS and SEP tokens manually if the tokenizer config doesn't do it automatically,
        # but the huggingface tokenizer.json usually handles this via templates.
        encoding = self.tokenizer.encode(text)
        
        # Truncate
        input_ids = encoding.ids[:self.max_length]
        attention_mask = encoding.attention_mask[:self.max_length]
        
        # Padding (though batch size is 1, ONNX might expect dynamic or static shape)
        input_ids_arr = np.array([input_ids], dtype=np.int64)
        attention_mask_arr = np.array([attention_mask], dtype=np.int64)
        token_type_ids_arr = np.zeros_like(input_ids_arr, dtype=np.int64)
        
        inputs = {
            "input_ids": input_ids_arr,
            "attention_mask": attention_mask_arr,
            "token_type_ids": token_type_ids_arr
        }
        
        outputs = self.session.run(None, inputs)
        
        # Mean Pooling over token embeddings
        token_embeddings = outputs[0] # shape: (1, seq_len, hidden_size)
        input_mask_expanded = np.expand_dims(attention_mask_arr, -1)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        pooled_embedding = sum_embeddings / sum_mask
        
        # Return normalized 1D vector
        return _normalize(pooled_embedding[0])

