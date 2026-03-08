"""
AURORA 配置常量模块
====================

按功能模块组织的配置常量。

模块结构：
- retrieval.py: 检索相关常量
- storage.py: 存储和容量相关常量
- coherence.py: 一致性检查相关常量
- evolution.py: 演化和反思相关常量
- identity.py: 身份和关系相关常量
- knowledge.py: 知识分类相关常量
- query_types.py: 查询类型检测相关常量
- numeric.py: 数值稳定性和基础常量

用法：
    from aurora.core.config.retrieval import INITIAL_SEARCH_K
    from aurora.core.config.identity import IDENTITY_RELEVANCE_WEIGHT
"""

from . import coherence as _coherence
from . import evolution as _evolution
from . import identity as _identity
from . import knowledge as _knowledge
from . import numeric as _numeric
from . import query_types as _query_types
from . import retrieval as _retrieval
from . import storage as _storage
from .coherence import *  # noqa: F401,F403
from .evolution import *  # noqa: F401,F403
from .identity import *  # noqa: F401,F403
from .knowledge import *  # noqa: F401,F403
from .numeric import *  # noqa: F401,F403
from .query_types import *  # noqa: F401,F403
from .retrieval import *  # noqa: F401,F403
from .storage import *  # noqa: F401,F403


def _module_exports(module: object) -> list[str]:
    return [name for name in vars(module) if name.isupper()]


__all__ = sorted(
    {
        *_module_exports(_coherence),
        *_module_exports(_evolution),
        *_module_exports(_identity),
        *_module_exports(_knowledge),
        *_module_exports(_numeric),
        *_module_exports(_query_types),
        *_module_exports(_retrieval),
        *_module_exports(_storage),
    }
)
