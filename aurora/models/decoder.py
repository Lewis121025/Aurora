"""Local decoder adapters for Aurora workspace-conditioned generation."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from aurora.core.types import DecoderOutput, DecoderRequest


class LocalDecoder(Protocol):
    def decode(self, request: DecoderRequest) -> DecoderOutput:
        """Generate text from a workspace-conditioned request."""


@dataclass(slots=True)
class TransformersLocalDecoder:
    """Lazy Hugging Face decoder adapter.

    It only imports heavy dependencies when instantiated or used, so the package
    stays importable in environments without the optional runtime stack.
    """

    model_id: str
    tokenizer_id: str | None = None
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError("transformers / torch are required for TransformersLocalDecoder") from exc

        tokenizer_id = self.tokenizer_id or self.model_id
        tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_id)
        model: Any = AutoModelForCausalLM.from_pretrained(self.model_id)
        if self.device != "auto":
            model = model.to(cast(Any, self.device))
        elif hasattr(model, "to"):
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        return model, tokenizer

    def decode(self, request: DecoderRequest) -> DecoderOutput:
        model, tokenizer = self._ensure_loaded()
        torch: Any = self._torch
        prompt = request.prompt or ""
        input_text = f"{prompt}\n\n{request.workspace.summary_vector}\n{request.cue}".strip()
        device = self._model_device(model)
        inputs: Any = tokenizer(input_text, return_tensors="pt").to(device)
        guard = torch.no_grad() if torch is not None else nullcontext()
        with guard:
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return DecoderOutput(
            text=text,
            token_count=len(text.split()),
            model_name=self.model_id,
            metadata={"mode": "transformers", **self.metadata},
        )

    def _model_device(self, model: Any) -> Any:
        if getattr(model, "device", None) is not None:
            return model.device
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            first = next(iter(parameters()), None)
            if first is not None:
                return first.device
        if self._torch is not None:
            return self._torch.device(self.device if self.device != "auto" else "cpu")
        return self.device if self.device != "auto" else "cpu"


def build_local_decoder(kind: str = "transformers", **kwargs: Any) -> LocalDecoder:
    kind = kind.lower().strip()
    if kind == "transformers":
        return TransformersLocalDecoder(**kwargs)
    raise ValueError(f"unsupported decoder kind: {kind}")


__all__ = [
    "LocalDecoder",
    "TransformersLocalDecoder",
    "build_local_decoder",
]
