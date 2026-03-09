# Lab 01 架构设计内参 (Architecture Design Review)

在你动手编写 `task_models.py` 里的代码之前，作为架构师，你必须明白我们**为什么**要这样规定，而不是仅仅知道**怎么写**。
这个文档将为你复盘这些设计的反面教材（Anti-Patterns）、当前选择的理由，以及业界可能的更好方案。

---

## 探讨 1: 外部雷达配置中心 (MicroSettings)

### ❌ 反面教材 (Anti-Pattern)
如果是一个外包团队或初级工程师，他们通常会这样写配置：
```python
# 灾难级别写法 -> config.py
LLM_ROUTER = "openai"
DATA_DIR = "/data/default"

# 或者稍微好一点的
import os
LLM_ROUTER = os.getenv("LLM_ROUTER", "openai")
```
**为什么灾难？**
1. **类型灾难**：`os.getenv` 读出来的永远是字符串，即使你想设 `TIMEOUT = 30` (整数)，它读出来的也会变成 `"30"` (字符串)。这会引发深层代码的类型错乱。
2. **缺乏防呆机制**：如果你把 `"openai"` 拼写成了 `"opneai"`，原生的 Python 不会在启动时管你，而是等跑到调用接口的那一行突然崩溃。如果你在跑一个 10 小时的数据清洗任务，在第 9 个小时调接口时才报错闪退，这是毁灭性的。

### ✅ 我们的设计（为什么使用 `Pydantic BaseSettings`）
1. **启动即决断 (Fail Fast)**：利用 `SettingsConfigDict` 的 `Literal` 定义。当项目启动、初始化这个配置类的第 **0.01 秒**，只要环境变量有拼写错误，立刻原地报错并拒绝启动系统。这帮我们规避了潜伏的隐患。
2. **前缀隔离屏障 (Prefix)**：强制加 `MICRO_` 这样的前缀，防止跟宿主机操作系统的其他环境变量（比如 `PATH` 或者 `JAVA_HOME`）发生变量名撞车。

### 🚀 有没有更好的方式？
在当前级别的规模下，`Pydantic` 是无敌的王者。但如果有一天，我们的 Aurora 变成了有 1000 台服务器的集群分布式服务呢？
**更好的架构（进阶方案）**：引入 **Apollo / Nacos (分布式配置中心)** 或者 **Kubernetes ConfigMap**。
当你修改了云端的一个配置项后，不需要重启代码进程，1000 台机器会自动在内存里热更新配置。在那时，仅靠本地扫 `.env` 文件和环境变量的 Pydantic 就不够用了，我们需要接外部的网络监听组件。

---

## 探讨 2: 内部数据的不可篡改性 (MicroPlot)

### ❌ 反面教材 (Anti-Pattern)
普通的面向对象写法：
```python
class BadPlot:
    def __init__(self, content: str, actors: list):
        self.content = content
        # 使用了 list 列表作为属性
        self.actors = actors
        self.created_at = time.time()
```
**为什么灾难？**
1. **时间戳克隆魔咒**：如上所述，`time.time()` 在类被加载进内存时只执行一次！这就导致你的系统开了多久，之后所有实例化的 `plot` 都是同一个时间戳。
2. **后门篡改风险**：你使用了 `list` 存储 `actors`。如果系统里的某只菜鸟工程师在某一层后续的流水线代码里手贱写了一句 `plot.actors.append("Hacker")`，历史记录就被修改了！这是一个没有保护机制的数据体。

### ✅ 我们的设计（使用 `Dataclass` + `Tuple` + `工厂函数`）
1. **`Tuple` 的架构哲学**：元组 (Tuple) 在 Python 中是绝对不可变（Immutable）的。如果你尝试修改一个 Tuple，底层内存会直接爆出错误拒绝执行。**在数据进入业务流水线后，“绝对不允许修改原始记录”是事件溯源 (Event Sourcing) 的核心信仰。**
2. **`default_factory` 的架构哲学**：把对时间的调用，推迟到每一次这个类想要“诞生”新对象的那一毫秒！这是一个极度经典的单例模式与工厂模式结合的点子。

### 🚀 有没有更好的方式？
对于高频吞吐对象，`Dataclass` 有一个隐性痛点——它太像一个字典，且比普通的类消耗稍微多一点点内存。
**更好的架构（进阶方案）**：在对性能有极致变态要求的量化交易或底层框架里，我们会换一种写法：**`__slots__` 机制配合 `NamedTuple`** 或者更激进的 **Rust `PyO3` 扩展**。
在 Python 加上 `__slots__ = ("content", "actors", "created_ts")` 可以关闭掉每个对象内部用来保存属性的字典 `__dict__`，直接节约一半的内存！如果你要在一秒内生成一百万个 Plot 对象，`slots` 才是真正的解药。

---
当你理解了为什么要这么写（Fail Fast, Immutability, Memory Isolation），现在去打开 `task_models.py`，你敲下的代码就再也不是无名字符，而是有灵魂的防护墙。
