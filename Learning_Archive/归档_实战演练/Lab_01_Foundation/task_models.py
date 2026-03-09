from typing import Tuple, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass, field
import time

# =====================================================================
# LAB 01: 坚不可摧的底层基石 (Foundation Models)
# 
# 任务档案：
# 你的初级系统需要处理外部配置，和内部的流传数据块(Plot)。
# 作为一个严苛的架构师，你必须利用 Pydantic 和 Dataclass 建立防线。
# 
# 提示词：如果卡住，请查看原版 My_Aurora/aurora/runtime/settings.py
# 和 My_Aurora/aurora/core/models/plot.py。
# =====================================================================

# ---------------------------------------------------------
# 【阶段 A】: 外部雷达配置中心
# 目标：实现一个能够从环境变量或 .env 读取覆盖项的配置类。
# ---------------------------------------------------------

class MicroSettings(BaseSettings):
    """系统的绝对控制面板"""
    
    # TODO 1: 补全 SettingsConfigDict！
    # 要求 1: 规定它必须只去捕获前缀为 "MICRO_" 的系统环境变量。
    # 要求 2: 当有不认识的额外变量混进来时，选择 "ignore" (无视)。
    model_config = SettingsConfigDict(
        # >>> 在这里写下代码
        # env_prefix=...
        # extra=...
    )

    # TODO 2: 设置一个极其严格的字面量约束！
    # 要求：系统只允许 `llm_router` 是 "openai" 或 "mock" 这两个词的其中之一。
    # 并且，必须默认给它赋初始值为 "mock"。
    # >>> 在这里写下代码


# ---------------------------------------------------------
# 【阶段 B】: 内部不可篡改的数据神经元
# 目标：利用 Dataclass 构建纯净的、每次都能取到最新时间、且防止历史窜改的类。
# ---------------------------------------------------------

# 这个工厂函数会返回被调用的那一瞬间的精确时间
def get_fresh_time() -> float:
    return time.time()

# TODO 3: 请给下面的类加上那顶魔法的 "装饰器帽子"！
# >>> 在这里写下代码
class MicroPlot:
    """内部的最小记忆承载单元"""
    
    # 属性 1: 必须是一段字符串文字
    content: str
    
    # TODO 4: 参与这段记忆的人。
    # 架构要求：这绝对不能是一个普通的 List（比如 ["User", "AI"]）！一旦存入，这波人的名字绝对不准被篡改！
    # 请选择一个在 Python 中原生具备 “不可变 (Immutable)” 特性的类型来约束它。
    # >>> 在这里写下代码
    # actors: ...
    

    # TODO 5: 记忆的生成时间。
    # 致命陷阱提示：如果你直接写 = get_fresh_time()，这辈子造出来的 Plot 时间全是重样的！
    # 你必须利用 field(...) 工厂模式，确保它在每次大批量被造出来的时候才临时打表。
    # >>> 在这里写下代码
    # created_ts: float = ...

