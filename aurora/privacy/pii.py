from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


# 非常简单的 PII 模式（生产环境应使用真实检测器）
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"\b(\+?\d[\d\s\-().]{7,}\d)\b")
_CREDIT = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


@dataclass
class RedactionResult:
    redacted_text: str
    findings: List[Tuple[str, str]]  # (type, match)


def redact(text: str) -> RedactionResult:
    findings: List[Tuple[str, str]] = []

    def _sub(pattern: re.Pattern, label: str, s: str, token: str) -> str:
        nonlocal findings
        for m in pattern.finditer(s):
            findings.append((label, m.group(0)))
        return pattern.sub(token, s)

    out = text
    out = _sub(_EMAIL, "email", out, "[REDACTED_EMAIL]")
    out = _sub(_PHONE, "phone", out, "[REDACTED_PHONE]")
    out = _sub(_CREDIT, "credit_card", out, "[REDACTED_CC]")
    return RedactionResult(redacted_text=out, findings=findings)
