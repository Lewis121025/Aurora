from __future__ import annotations

from enum import Enum


class Phase(str, Enum):
    AWAKE = "awake"
    DOZE = "doze"
    SLEEP = "sleep"
