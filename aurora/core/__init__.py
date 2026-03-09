"""Aurora core domain layer.

这里放纯领域与算法逻辑：
- ``soul_memory``: Soul-Memory 主引擎
- ``components`` / ``graph`` / ``retrieval``: 底层算法结构
- 其余模块: tension、coherence、causal 等基础能力
"""

# Main system
from aurora.core.soul_memory import AuroraSoulMemory
from aurora.core.tension import TensionManager, Tension, TensionType, TensionResolution
from aurora.core.narrator import (
    NarratorEngine,
    NarrativePerspective,
    NarrativeTrace,
    NarrativeElement,
    NarrativeRole,
)

__all__ = [
    "AuroraSoulMemory",
    "TensionManager",
    "Tension",
    "TensionType",
    "TensionResolution",
    "NarratorEngine",
    "NarrativePerspective",
    "NarrativeTrace",
    "NarrativeElement",
    "NarrativeRole",
]
