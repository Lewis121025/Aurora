from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemClock:
    def now(self) -> float:
        return time.time()
