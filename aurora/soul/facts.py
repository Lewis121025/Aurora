from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ExtractedFact:
    fact_text: str
    fact_type: str
    entities: List[str]
    plot_id: Optional[str] = None

    def to_state_dict(self) -> dict[str, object]:
        return {
            "fact_text": self.fact_text,
            "fact_type": self.fact_type,
            "entities": list(self.entities),
            "plot_id": self.plot_id,
        }


class FactExtractor:
    FACT_PATTERNS = {
        "quantity": [
            r"(\d+)\s*(items?|books?|clothes?|things?|件|本|个|条)",
            r"(几个|多少|几本|几件)\s*(\w+)",
        ],
        "action": [
            r"\b(bought|returned|picked up|ordered|received|purchased)\b",
            r"\b(买|退|订|收|购买|退货)\b",
        ],
        "location": [
            r"\b(at|from|to)\s+(store|shop|restaurant|place|library|bookstore)\s+(\w+)",
            r"\b(在|从|到)\s*(商店|书店|餐厅|图书馆|地方)\s*(\w+)",
        ],
        "time": [
            r"\b(yesterday|last week|tomorrow|next week|today|recently)\b",
            r"\b(昨天|上周|明天|下周|今天|最近|之前|以后)\b",
        ],
        "preference": [
            r"\b(like|love|prefer|hate|enjoy|dislike)\s+(\w+)",
            r"\b(喜欢|爱|偏好|讨厌|享受|不喜欢)\s+(\w+)",
        ],
    }

    def __init__(self, min_fact_length: int = 3) -> None:
        self.min_fact_length = min_fact_length
        self._compiled_patterns = {
            fact_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for fact_type, patterns in self.FACT_PATTERNS.items()
        }

    def extract(self, text: str) -> List[ExtractedFact]:
        if not text or not text.strip():
            return []

        facts: List[ExtractedFact] = []
        text_lower = text.lower()

        for pattern in self._compiled_patterns.get("quantity", []):
            for match in pattern.finditer(text_lower):
                quantity = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
                item = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                fact_text = f"quantity:{quantity} {item}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(fact_text=fact_text, fact_type="quantity", entities=[item] if item else []))

        for pattern in self._compiled_patterns.get("action", []):
            for match in pattern.finditer(text_lower):
                action = match.group(0)
                start = max(0, match.start() - 20)
                end = min(len(text_lower), match.end() + 30)
                context = text_lower[start:end].strip()
                fact_text = f"action:{action} {context}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(fact_text=fact_text, fact_type="action", entities=[action]))

        for pattern in self._compiled_patterns.get("location", []):
            for match in pattern.finditer(text_lower):
                location_parts = [group for group in match.groups() if group]
                fact_text = f"location:{' '.join(location_parts)}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(fact_text=fact_text, fact_type="location", entities=location_parts))

        for pattern in self._compiled_patterns.get("time", []):
            for match in pattern.finditer(text_lower):
                time_expr = match.group(0)
                fact_text = f"time:{time_expr}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(fact_text=fact_text, fact_type="time", entities=[time_expr]))

        for pattern in self._compiled_patterns.get("preference", []):
            for match in pattern.finditer(text_lower):
                preference = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                target = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
                fact_text = f"preference:{preference} {target}".strip()
                if len(fact_text) >= self.min_fact_length:
                    facts.append(ExtractedFact(fact_text=fact_text, fact_type="preference", entities=[target] if target else []))

        seen: set[str] = set()
        unique_facts: List[ExtractedFact] = []
        for fact in facts:
            if fact.fact_text in seen:
                continue
            seen.add(fact.fact_text)
            unique_facts.append(fact)
        return unique_facts
