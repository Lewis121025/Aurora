"""
aurora/soul
Aurora 记忆与身份引擎核心包。
它实现了 V4 版本的生成式灵魂算法，包括动态心理轴、非参数聚类检索以及认知失调驱动的自我演化。
"""

from aurora.soul.engine import AuroraSoul, SoulConfig
from aurora.soul.models import IdentityState, Plot, StoryArc, Theme

__all__ = ["AuroraSoul", "SoulConfig", "Plot", "StoryArc", "Theme", "IdentityState"]
