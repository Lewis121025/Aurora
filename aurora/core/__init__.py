"""Aurora core domain layer.

这里放纯领域与算法逻辑：
- ``memory``: 记忆引擎与 mixin
- ``models``: 数据模型
- ``components`` / ``graph`` / ``retrieval``: 底层算法结构
- 其余模块: coherence、causal、tension、self_narrative 等核心能力
"""

# Main system
from aurora.core.memory import AuroraMemory
from aurora.core.tension import TensionManager, Tension, TensionType, TensionResolution
from aurora.core.narrator import (
    NarratorEngine,
    NarrativePerspective,
    NarrativeTrace,
    NarrativeElement,
    NarrativeRole,
)

__all__ = [
    "AuroraMemory",
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
