# 🟡 Stage 2: 静态蓝图 —— Pydantic、Dataclass 与数据模型 (强化实战版)

恭喜你跨越了异常捕获与类型提示的门槛。现在，我们要给这座名为 Aurora 的摩天大楼**打钢筋**。AI 的记忆并非虚无缥缈，它们是一条条有着特定字段的数据格式。

通过这一关，你将掌握当今 Python 工业界构建“数据承载体”和“配置开关”的两把无上神兵：**Pydantic** 与 **Dataclass**。

---

## 🏗️ 序章：为什么不能用普通的 `class`？
在远古的 Python 代码里，程序员写一个包含大量字段的数据模型是极度痛苦的：
```python
# 古板的写法，为了存一个 Plot 类的四个属性，光写 __init__ 就要敲几十下键盘
class OldPlot:
    def __init__(self, id: str, text: str, surprise: float, ts: float):
        self.id = id
        self.text = text
        self.surprise = surprise
        self.ts = ts
```
如果有20个字段呢？如果还要写如何打印它们、如何比较它们呢？代码将会变得又臭又长。
在现在的 Aurora 中，我们淘汰了这种写法。我们分别用 `Pydantic` 管理外部配置，用内置的 `@dataclass` 管理内部频繁生成的结构体。

## 🎯 任务 1：现代化系统的总开关 —— Pydantic
如果你要在代码里配一堆密码和开关，你怎么做？写在普通的字典里？太容易敲错字母了。
**请打开文件：`aurora/runtime/settings.py`**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional

class AuroraSettings(BaseSettings):
    """Runtime configuration."""
    model_config = SettingsConfigDict(
        env_prefix="AURORA_",
        extra="ignore",
        env_file=".env",
    )

    data_dir: str = "./data"
    llm_provider: Literal["bailian", "ark", "mock"] = "mock"
```

**【高级配置哲学剖析】：**
- **Pydantic 神库 (`BaseSettings`)**：这是目前 Python 界公认最强的数据验证库。当你继承 `BaseSettings` 后，这个类就不再是一个普通类了。它变成了一个**长着眼睛的环境雷达**。
- **自动环境变量覆盖 (`env_file=".env"`)**：这表示，无论你在代码里给 `data_dir` 写了什么默认值（比如 `"./data"`），只要项目根目录下有一个名为 `.env` 的隐藏纯文本文件，并且里面写着 `AURORA_DATA_DIR=./my_secret_data`，那么程序启动的瞬间，Pydantic 就会自动抓住 `.env` 里的值，**覆盖并抛弃掉代码里写死的默认值**！这是工业界管理敏感密钥（不能传到通用代码库里的 API Key）的唯一正确姿势。
- **字面量防呆 (`Literal`)**：`llm_provider` 的类型被锁死在 `["bailian", "ark", "mock"]` 这三个词里。如果你在 `.env` 里手滑打成了 `AURORA_LLM_PROVIDER=openai`，在脚本启动的 0.1 秒，程序会直接报错闪退：“你填入的值不在合法候选项中”，从而杜绝了在深层代码网络里报出诡异的问题。

### 💻 实操练习 1：亲眼见证 Pydantic 的覆盖魔法
我们写一段真实运作的代码来看看 Pydantic 是怎么无缝替换参数的。在你项目根目录（`My_Aurora`）下，新建一个文件叫 `test_settings.py`：

```python
# 文件名: test_settings.py (实操演示)

# 我们直接导入 Aurora 的总配置类
from aurora.runtime.settings import AuroraSettings
import os

print("=== 1. 默认状态下的配置 ===")
# 实例化一个配置对象（不提供任何覆盖，完全随缘读取）
default_settings = AuroraSettings()
print(f"默认 LLM 供应商是: {default_settings.llm_provider}")
print(f"默认数据存储文件夹是: {default_settings.data_dir}")

print("\n=== 2. 用系统环境变量强制接管 ===")
# 模拟我们在外部终端里临时设置了环境变量 (注意加上了我们在 SettingsConfigDict 规定的前缀 AURORA_)
os.environ["AURORA_LLM_PROVIDER"] = "ark"
os.environ["AURORA_DATA_DIR"] = "/tmp/my_test_aurora_data"

# 再次实例化，见证奇迹的时刻
overridden_settings = AuroraSettings()
print(f"环境变量入侵后的 LLM 供应商是: {overridden_settings.llm_provider}")
print(f"环境变量入侵后的数据文件夹是: {overridden_settings.data_dir}")

print("\n=== 3. 类型防呆测试 (故意写错) ===")
os.environ["AURORA_LLM_PROVIDER"] = "deepseek"  # Literal 没允许这个词
try:
    bad_settings = AuroraSettings()
except Exception as e:
    # 你会看到 Pydantic 给出了极其详尽的数据校验(Validation)错误提示
    print(f"Pydantic 拦截了非法配置启动！原因: {e}")
```
打开你的终端，利用我们第一环配置好的大环境执行：`uv run python test_settings.py`。
仔细阅读输出，看 Pydantic 是如何聪明地从 `os.environ`（或者 `.env` 文件）里吸入变量的组合规则，并且在遇到不认识的 `deepseek` 单词时无情打回的。

---

## 🎯 任务 2：AI 记忆的神经元形态 —— Dataclass 魔法

看完了外部配置的保险柜，我们来看系统内部频繁流转业务主体：Plot（情节，也就是一句话形成的基础记忆神经元）。
**请打开文件：`aurora/core/models/plot.py`**（找到 `class Plot:` 定义）

```python
from dataclasses import dataclass, field
import numpy as np
import time

def now_ts() -> float:
    """返回当前精确时间戳."""
    return time.time()

@dataclass
class Plot:
    id: str
    ts: float
    text: str
    actors: Tuple[str, ...]
    embedding: np.ndarray

    access_count: int = 0
    # 工业级深坑：不要直接写 = now_ts() !
    last_access_ts: float = field(default_factory=now_ts)
```

**【极简美学拆解】：**
- **魔法装饰器 (`@dataclass`)**：只要在类名脑袋上“戴上这顶帽子”，Python 便会在你看不见的低层，默默为你生成所有繁杂的内部构造方法。你只需要像列表一样把所有的变量名 and 他们的 `Type Hint`（类型提示）一字排开，整个大模型所需的数据块就干净利落地定义好了！极其易读。
- **神级避灾点 (`default_factory=now_ts`)**：
  - **初级小白的致死坑**：如果你的时间戳字段直接写 `last_access_ts: float = time.time()`，那么这段代码在程序被导入的一瞬间就会把时间取出来**钉死定格**。之后的一整天里，每当这个系统接收新消息造出一个新 `Plot`，所有 Plot 身上的这一秒时间全是同一个过去的克隆时间！
  - **工厂模式解法**：`field(default_factory=now_ts)` 意为“每次实例化这套模具时，请临时调用一下 `now_ts` 工厂函数重新取一次表”。它保证了砸出来的每一个 Plot 都是带着最新鲜的热乎时间。

### 💻 实操练习 2：纯净的 Dataclass 数据块结构
为了让你更直观地看到 Dataclass 和 Tuple（元组）起到的保护效果。我们在项目根目录下新建 `test_dataclass.py`：

```python
# 文件名: test_dataclass.py (实操演示)
from dataclasses import dataclass, field
import time
from typing import Tuple

# 1. 模拟一个缩小版的 Plot 类
@dataclass
class MiniPlot:
    text: str
    # 元组一旦装载好，就不能再增删改；这是为了防止后期的 AI 脑子进水篡改历史
    actors: Tuple[str, ...] 
    # 正确使用工厂获取最新的生成时间
    created_at: float = field(default_factory=time.time)

# 2. 砸出两个前后不同时期的记忆块
print("正在生成第一条记忆...")
plot_1 = MiniPlot(text="你好 Aurora！", actors=("User", "Agent"))
time.sleep(1) # 刻意让程序睡1秒，模拟相隔一段时间才说第二句话

print("正在生成第二条记忆...")
plot_2 = MiniPlot(text="我们今天学习什么？", actors=("User", "Agent"))

# 3. 打印对比 
print("\n--- 查看对象自带的完美打印显示 ---")
# 你没写任何打印规则，但 Dataclass 送给你了极其漂亮的打印输出结构！
print(plot_1) 
print(plot_2)

print("\n--- 验证时间戳 ---")
print(f"时间间距: {plot_2.created_at - plot_1.created_at:.2f} 秒 (这证明 default_factory 每次都取了最新的表！)")

print("\n--- 验证铁盒子 (Tuple) 的不可篡改性 ---")
try:
    # 尝试粗暴地改写参与这句对话的演员名单
    plot_1.actors[0] = "Hacker"
except TypeError as e:
    print(f"篡改历史失败（类型保护生效）: {e}")
```
打开终端，运行 `uv run python test_dataclass.py`。
观察屏幕上完美整洁的对象输出，并确认由于我们使用了 `Tuple`（元组/铁盒子），即便外部强行篡改说这句话的人的名字，也必定拦截失败。这就是静态数据模型设计的最终意义！

---
**🎉 Stage 2 结业检视：**
地基与钢筋已全部就位！
通过实战，你掌握了让 Pydantic 从空气中（环境变量）贪婪地吸取消化运行配置的方法。你也亲自上手操作了 Dataclass，制造出了永远拥有鲜活时间戳并坚挺不可篡改的事件实体 `Plot`。

大军已列阵，粮草已备齐。接下来，我们要让这个系统动起来。准备好在 Stage 3 前往中央司令部，检阅千军万马奔流通过的“大动脉”吧！
