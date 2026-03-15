from __future__ import annotations

import re
from collections import Counter


def extract_tags(text: str) -> tuple[str, ...]:
    latin = re.findall(r"[A-Za-z]{3,}", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]{1,4}", text)
    merged = [token.strip() for token in latin + cjk if token.strip()]
    if not merged:
        merged = [text[:12].strip() or "moment"]
    counter = Counter(merged)
    return tuple(sorted(token for token, _ in counter.most_common(12)))
