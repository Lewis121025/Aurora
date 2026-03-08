# Aurora 项目完整学习路线

> 为 Python 小白设计的从零到精通的学习路径
>
> 基于认知科学原理：间隔重复、主动学习、渐进式复杂度

---

## 📋 学习路线总览

```
第0阶段: Python 基础 (2-3周)
    ↓
第1阶段: Aurora 核心概念 (3天) 
    ↓
第2阶段: Python 进阶 (2周)
    ↓
第3阶段: Aurora 数据模型 (1周)
    ↓
第4阶段: Aurora 运行时 (1周)
    ↓
第5阶段: Aurora 核心算法 (2周)
    ↓
第6阶段: 实战项目 (2周)
```

**总时长**: 约 8-10 周（每天 2-3 小时）

**学习原则**:
- ✅ 每天学习 2-3 小时，不要超过 4 小时（避免认知过载）
- ✅ 每学习 25 分钟休息 5 分钟（番茄工作法）
- ✅ 每完成一个阶段休息 1-2 天（巩固记忆）
- ✅ 动手实践 > 阅读理论（70% 实践，30% 理论）
- ✅ 每天写学习日志（元认知训练）

---

## 第0阶段: Python 基础 (2-3周)

### 🎯 学习目标
- 掌握 Python 基本语法
- 能够编写简单的 Python 程序
- 理解函数、类、模块的概念

### 📚 学习内容

#### Week 1: Python 入门
**Day 1-2: 环境搭建与基础语法**
- [ ] 安装 Python 3.10+
- [ ] 安装 VS Code 或 PyCharm
- [ ] 学习变量、数据类型（int, float, str, bool）
- [ ] 学习运算符和表达式

**实践任务**:
```python
# 练习1: 计算器
def calculator(a, b, operation):
    if operation == '+':
        return a + b
    elif operation == '-':
        return a - b
    # 继续完成 *, /
```

**Day 3-4: 控制流**
- [ ] if/elif/else 条件语句
- [ ] for 循环和 while 循环
- [ ] break, continue, pass

**实践任务**:
```python
# 练习2: 猜数字游戏
import random
secret = random.randint(1, 100)
# 让用户猜数字，给出提示
```

**Day 5-7: 数据结构**
- [ ] 列表（list）
- [ ] 字典（dict）
- [ ] 元组（tuple）
- [ ] 集合（set）

**实践任务**:
```python
# 练习3: 学生成绩管理
students = {
    "Alice": [85, 90, 88],
    "Bob": [78, 82, 80]
}
# 计算每个学生的平均分
```

#### Week 2: 函数与模块
**Day 8-10: 函数**
- [ ] 函数定义和调用
- [ ] 参数（位置参数、关键字参数、默认参数）
- [ ] 返回值
- [ ] 作用域

**实践任务**:
```python
# 练习4: 文本分析工具
def word_count(text):
    """统计文本中的单词数量"""
    pass

def most_common_word(text):
    """找出最常见的单词"""
    pass
```

**Day 11-14: 类与对象**
- [ ] 类的定义
- [ ] 实例化对象
- [ ] 方法和属性
- [ ] `__init__` 构造函数

**实践任务**:
```python
# 练习5: 银行账户类
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        """存款"""
        pass

    def withdraw(self, amount):
        """取款"""
        pass
```

#### Week 3: 文件与异常
**Day 15-17: 文件操作**
- [ ] 读取文件（read, readline, readlines）
- [ ] 写入文件（write）
- [ ] with 语句

**实践任务**:
```python
# 练习6: 日志记录器
def log_message(message, filename="app.log"):
    """将消息写入日志文件"""
    pass
```

**Day 18-21: 异常处理**
- [ ] try/except
- [ ] 常见异常类型
- [ ] 自定义异常

**实践任务**:
```python
# 练习7: 安全的除法函数
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "不能除以零"
```

### ✅ 自我检测
完成以下任务，确认你已掌握 Python 基础：

```python
# 综合练习: 简单的待办事项管理器
class TodoList:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        """添加任务"""
        pass

    def remove_task(self, task_id):
        """删除任务"""
        pass

    def list_tasks(self):
        """列出所有任务"""
        pass

    def save_to_file(self, filename):
        """保存到文件"""
        pass

    def load_from_file(self, filename):
        """从文件加载"""
        pass

# 使用示例
todo = TodoList()
todo.add_task("学习 Python")
todo.add_task("阅读 Aurora 代码")
todo.list_tasks()
todo.save_to_file("tasks.txt")
```

**通过标准**: 能够独立完成上述代码，并运行成功。

---

## 第1阶段: Aurora 核心概念 (3天)

### 🎯 学习目标
- 理解 Aurora 的设计哲学
- 理解叙事记忆的概念
- 不看代码，先建立心智模型

### 📚 学习内容

#### Day 1: 什么是叙事记忆？
**阅读材料**:
- [ ] 阅读 `README.md` 的"核心特性"部分
- [ ] 理解 Plot、Story、Theme 的概念

**思考练习**:
```
用自己的话解释：
1. 什么是 Plot？举一个生活中的例子。
2. 什么是 Story？它和 Plot 有什么关系？
3. 什么是 Theme？为什么叫"主题"而不是"模式"？
```

**类比理解**:
```
把 Aurora 想象成一个人的记忆系统：

Plot = 你记得的一件具体的事
例如："昨天我和朋友去咖啡馆聊天"

Story = 你和某个人的关系故事
例如："我和这个朋友的友谊故事"

Theme = 你从这些故事中理解的自己
例如："我是一个重视友谊的人"
```

#### Day 2: Aurora 的架构
**阅读材料**:
- [ ] 阅读 `README.md` 的"架构设计"部分
- [ ] 理解分层结构

**画图练习**:
```
在纸上画出 Aurora 的架构图：
1. core 层做什么？
2. runtime 层做什么？
3. integrations 层做什么？
4. interfaces 层做什么？

用箭头标出数据流向。
```

#### Day 3: Aurora 的工作流程
**阅读材料**:
- [ ] 阅读 `README.md` 的"核心运行链路"部分

**流程图练习**:
```
画出一次对话的完整流程：
1. 用户说了一句话
2. Aurora 如何处理？
3. 数据存储在哪里？
4. 如何检索记忆？

用流程图表示出来。
```

### ✅ 自我检测
不看文档，回答以下问题：
1. Aurora 的三层记忆结构是什么？
2. Aurora 的数据存储在哪里？
3. Aurora 如何检索记忆？
4. Aurora 的核心设计原则是什么？

**通过标准**: 能够用自己的话清晰解释上述概念。

---

## 第2阶段: Python 进阶 (2周)

### 🎯 学习目标
- 掌握 Aurora 代码中使用的 Python 高级特性
- 理解类型注解、装饰器、上下文管理器等

### 📚 学习内容

#### Week 1: 类型注解与数据类
**Day 1-3: 类型注解**
- [ ] 基本类型注解（int, str, float, bool）
- [ ] 容器类型注解（List, Dict, Tuple, Optional）
- [ ] 函数类型注解

**学习资源**:
```python
# 类型注解示例
from typing import List, Dict, Optional

def greet(name: str) -> str:
    return f"Hello, {name}!"

def process_scores(scores: List[int]) -> float:
    return sum(scores) / len(scores)

def find_user(user_id: int) -> Optional[Dict[str, str]]:
    # 可能返回 None
    return None
```

**实践任务**:
```python
# 练习8: 为之前的 BankAccount 类添加类型注解
from typing import Optional

class BankAccount:
    def __init__(self, owner: str, balance: float = 0.0) -> None:
        self.owner: str = owner
        self.balance: float = balance

    def deposit(self, amount: float) -> None:
        """存款"""
        pass
```

**Day 4-7: dataclass**
- [ ] 什么是 dataclass
- [ ] @dataclass 装饰器
- [ ] field() 函数
- [ ] 序列化和反序列化

**学习资源**:
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    age: int
    scores: List[int] = field(default_factory=list)

    def average_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

# 使用
student = Student(name="Alice", age=20, scores=[85, 90, 88])
print(student.average_score())
```

**实践任务**:
```python
# 练习9: 用 dataclass 重写 BankAccount
from dataclasses import dataclass
from typing import List

@dataclass
class Transaction:
    amount: float
    type: str  # "deposit" or "withdraw"
    timestamp: float

@dataclass
class BankAccount:
    owner: str
    balance: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)

    def deposit(self, amount: float) -> None:
        """存款"""
        pass
```

#### Week 2: 高级特性
**Day 8-10: 抽象基类与协议**
- [ ] ABC (Abstract Base Class)
- [ ] @abstractmethod
- [ ] Protocol

**学习资源**:
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self) -> str:
        """动物叫声"""
        pass

class Dog(Animal):
    def make_sound(self) -> str:
        return "Woof!"

class Cat(Animal):
    def make_sound(self) -> str:
        return "Meow!"
```

**实践任务**:
```python
# 练习10: 创建一个存储接口
from abc import ABC, abstractmethod
from typing import Any, Optional

class Storage(ABC):
    @abstractmethod
    def save(self, key: str, value: Any) -> None:
        """保存数据"""
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """加载数据"""
        pass

class FileStorage(Storage):
    def save(self, key: str, value: Any) -> None:
        # 实现文件存储
        pass

    def load(self, key: str) -> Optional[Any]:
        # 实现文件加载
        pass
```

**Day 11-14: 上下文管理器与生成器**
- [ ] with 语句的原理
- [ ] `__enter__` 和 `__exit__`
- [ ] yield 关键字

**学习资源**:
```python
class FileManager:
    def __init__(self, filename: str):
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, 'r')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# 使用
with FileManager("data.txt") as f:
    content = f.read()
```

### ✅ 自我检测
完成以下综合练习：

```python
# 综合练习: 简单的记忆系统
from dataclasses import dataclass, field
from typing import List, Optional, Protocol
from abc import ABC, abstractmethod
import time

@dataclass
class Memory:
    """记忆单元"""
    id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

class MemoryStore(ABC):
    """记忆存储接口"""
    @abstractmethod
    def add(self, memory: Memory) -> None:
        pass

    @abstractmethod
    def search(self, query: str) -> List[Memory]:
        pass

class SimpleMemoryStore(MemoryStore):
    """简单的内存存储实现"""
    def __init__(self):
        self.memories: List[Memory] = []

    def add(self, memory: Memory) -> None:
        # 实现添加逻辑
        pass

    def search(self, query: str) -> List[Memory]:
        # 实现搜索逻辑（简单的关键词匹配）
        pass

# 测试
store = SimpleMemoryStore()
store.add(Memory(id="m1", content="我喜欢 Python", tags=["编程"]))
store.add(Memory(id="m2", content="我喜欢机器学习", tags=["AI"]))
results = store.search("Python")
```

**通过标准**: 能够独立完成上述代码，理解每个概念的作用。

---

## 第3阶段: Aurora 数据模型 (1周)

### 🎯 学习目标
- 理解 Plot、Story、Theme 的代码实现
- 掌握 Aurora 的数据结构
- 能够创建和操作这些数据模型

### 📚 学习内容

#### Day 1-2: Plot 模型
**阅读文件**: `aurora/core/models/plot.py`

**学习重点**:
- [ ] Plot 的三层结构：事实层、关系层、身份层
- [ ] RelationalContext 数据类
- [ ] IdentityImpact 数据类
- [ ] Plot 的序列化方法

**实践任务**:
```python
# 练习11: 创建一个 Plot
from aurora.core.models.plot import Plot, RelationalContext, IdentityImpact
import time

plot = Plot(
    id="plot_001",
    event_id="evt_001",
    user_msg="我喜欢写 Python 代码",
    agent_msg="很好！Python 是一门优雅的语言。",
    embedding=[0.1, 0.2, 0.3],  # 简化的嵌入向量
    ts=time.time()
)

# 添加关系层
plot.relational = RelationalContext(
    with_whom="user",
    my_role_in_relation="技术导师",
    relationship_quality_delta=0.1,
    what_this_says_about_us="我们有共同的技术兴趣"
)

print(f"Plot ID: {plot.id}")
print(f"关系对象: {plot.relational.with_whom}")
```

**思考问题**:
1. 为什么 Plot 需要三层结构？
2. RelationalContext 捕获了什么信息？
3. embedding 向量的作用是什么？

#### Day 3-4: Story 模型
**阅读文件**: `aurora/core/models/story.py`

**学习重点**:
- [ ] StoryArc 数据类
- [ ] RelationshipMoment 数据类
- [ ] Story 如何围绕关系组织
- [ ] Story 的演化机制

**实践任务**:
```python
# 练习12: 创建一个 Story
from aurora.core.models.story import StoryArc, RelationshipMoment

story = StoryArc(
    id="story_001",
    relationship_with="user",
    plot_ids=["plot_001", "plot_002"],
    prototype=[0.15, 0.25, 0.35],  # 平均嵌入
    created_ts=time.time()
)

# 添加关系时刻
moment = RelationshipMoment(
    ts=time.time(),
    event_summary="讨论 Python 编程",
    trust_level=0.8,
    my_role="技术导师"
)
story.trajectory.append(moment)

print(f"Story 关系对象: {story.relationship_with}")
print(f"包含 {len(story.plot_ids)} 个 Plot")
```

**思考问题**:
1. Story 和 Plot 的关系是什么？
2. 为什么 Story 围绕"关系"组织？
3. trajectory（轨迹）记录了什么？

#### Day 5-7: Theme 模型
**阅读文件**: `aurora/core/models/theme.py`

**学习重点**:
- [ ] Theme 数据类
- [ ] identity_dimension（身份维度）
- [ ] supporting_relationships（支持关系）
- [ ] tensions_with（张力关系）

**实践任务**:
```python
# 练习13: 创建一个 Theme
from aurora.core.models.theme import Theme

theme = Theme(
    id="theme_001",
    story_ids=["story_001", "story_002"],
    prototype=[0.2, 0.3, 0.4],
    created_ts=time.time(),
    identity_dimension="作为技术导师的我",
    supporting_relationships=["user", "developer_community"],
    strength=0.85
)

print(f"身份维度: {theme.identity_dimension}")
print(f"支持关系: {theme.supporting_relationships}")
print(f"强度: {theme.strength}")
```

**思考问题**:
1. Theme 如何从 Story 中涌现？
2. identity_dimension 回答了什么问题？
3. strength（强度）代表什么？

### ✅ 自我检测
完成以下综合练习：

```python
# 综合练习: 构建完整的记忆层次
from aurora.core.models.plot import Plot
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
import time

# 1. 创建多个 Plot
plots = [
    Plot(id=f"plot_{i}", event_id=f"evt_{i}",
         user_msg=f"消息{i}", agent_msg=f"回复{i}",
         embedding=[i*0.1, i*0.2, i*0.3], ts=time.time())
    for i in range(5)
]

# 2. 将 Plot 聚合成 Story
story = StoryArc(
    id="story_001",
    relationship_with="user",
    plot_ids=[p.id for p in plots],
    prototype=[0.2, 0.4, 0.6],
    created_ts=time.time()
)

# 3. 从 Story 提炼 Theme
theme = Theme(
    id="theme_001",
    story_ids=[story.id],
    prototype=[0.25, 0.45, 0.65],
    created_ts=time.time(),
    identity_dimension="作为助手的我",
    supporting_relationships=["user"],
    strength=0.9
)

# 4. 打印层次结构
print("记忆层次:")
print(f"  Theme: {theme.identity_dimension}")
print(f"    └─ Story: 与 {story.relationship_with} 的关系")
print(f"        └─ {len(plots)} 个 Plots")
```

**通过标准**: 理解三层模型的关系，能够创建和操作这些数据结构。

---

## 第4阶段: Aurora 运行时 (1周)

### 🎯 学习目标
- 理解 AuroraRuntime 的工作原理
- 掌握 ingest 和 query 流程
- 理解持久化机制

### 📚 学习内容

#### Day 1-2: 配置与初始化
**阅读文件**:
- `aurora/runtime/settings.py`
- `aurora/runtime/bootstrap.py`

**学习重点**:
- [ ] AuroraSettings 配置项
- [ ] Provider 的概念（LLM、Embedding）
- [ ] create_memory() 如何创建记忆引擎

**实践任务**:
```python
# 练习14: 配置 Aurora
from aurora.runtime.settings import AuroraSettings

settings = AuroraSettings(
    data_dir="./my_aurora_data",
    llm_provider="mock",
    embedding_provider="local",
)

print(f"数据目录: {settings.data_dir}")
print(f"LLM Provider: {settings.llm_provider}")
print(f"Embedding Provider: {settings.embedding_provider}")
```

**思考问题**:
1. 为什么需要 Provider 抽象？
2. data_dir 存储了什么？
3. mock provider 的作用是什么？

#### Day 3-4: Ingest 流程
**阅读文件**: `aurora/runtime/runtime.py`（重点看 `ingest_interaction` 方法）

**学习重点**:
- [ ] ingest_interaction() 的参数
- [ ] _apply_interaction() 的实现
- [ ] 如何写入 event log
- [ ] 如何调用 AuroraMemory.ingest()

**实践任务**:
```python
# 练习15: 摄入交互
from aurora import AuroraRuntime, AuroraSettings

runtime = AuroraRuntime(
    settings=AuroraSettings(
        data_dir="./test_data",
        llm_provider="mock",
        embedding_provider="local",
    )
)

# 摄入第一次交互
result = runtime.ingest_interaction(
    event_id="evt_001",
    session_id="test_session",
    user_message="你好，我是新用户",
    agent_message="你好！很高兴认识你。",
)

print(f"摄入结果: {result}")
print(f"当前 Plots 数量: {len(runtime.mem.plots)}")
```

**调试练习**:
在 `runtime.ingest_interaction()` 处设置断点，跟踪执行流程：
1. 参数如何传递？
2. event log 何时写入？
3. AuroraMemory 何时被调用？

#### Day 5-7: Query 流程
**阅读文件**: `aurora/runtime/runtime.py`（重点看 `query` 方法）

**学习重点**:
- [ ] query() 的参数
- [ ] 如何调用 AuroraMemory.retrieve()
- [ ] QueryResult 的结构
- [ ] 如何从 doc store 补充信息

**实践任务**:
```python
# 练习16: 查询记忆
# 继续上面的 runtime

# 摄入更多交互
runtime.ingest_interaction(
    event_id="evt_002",
    session_id="test_session",
    user_message="我想学习 Python",
    agent_message="Python 是一门很好的入门语言！",
)

# 查询记忆
results = runtime.query(text="Python", k=5)

print(f"找到 {len(results.hits)} 条记忆:")
for hit in results.hits:
    print(f"  - {hit.plot_id}: {hit.summary[:50]}...")
    print(f"    相似度: {hit.score:.3f}")
```

**思考问题**:
1. query 如何找到相关记忆？
2. k 参数的作用是什么？
3. score（相似度）是如何计算的？

### ✅ 自我检测
完成以下综合练习：

```python
# 综合练习: 完整的对话记忆系统
from aurora import AuroraRuntime, AuroraSettings
import time

# 1. 初始化运行时
runtime = AuroraRuntime(
    settings=AuroraSettings(
        data_dir="./chat_memory",
        llm_provider="mock",
        embedding_provider="local",
    )
)

# 2. 模拟一段对话
conversations = [
    ("我喜欢编程", "编程是一项很有创造力的活动！"),
    ("我最近在学 Python", "Python 是一门优雅的语言。"),
    ("我想做一个聊天机器人", "可以用 Python 的 NLP 库来实现。"),
    ("我对机器学习感兴趣", "机器学习和 Python 是绝配！"),
]

for i, (user_msg, agent_msg) in enumerate(conversations):
    runtime.ingest_interaction(
        event_id=f"evt_{i:03d}",
        session_id="learning_session",
        user_message=user_msg,
        agent_message=agent_msg,
    )
    time.sleep(0.1)  # 模拟时间间隔

# 3. 查询记忆
queries = ["Python", "机器学习", "编程"]
for query in queries:
    results = runtime.query(text=query, k=3)
    print(f"\n查询: {query}")
    print(f"找到 {len(results.hits)} 条记忆")
    for hit in results.hits[:2]:  # 只显示前2条
        print(f"  - {hit.summary[:40]}... (score: {hit.score:.3f})")

# 4. 查看内存状态
print(f"\n当前状态:")
print(f"  Plots: {len(runtime.mem.plots)}")
print(f"  Stories: {len(runtime.mem.stories)}")
print(f"  Themes: {len(runtime.mem.themes)}")
```

**通过标准**: 能够独立运行完整的对话记忆系统，理解 ingest 和 query 的流程。

---

## 第5阶段: Aurora 核心算法 (2周)

### 🎯 学习目标
- 深入理解 AuroraMemory 引擎
- 掌握向量检索和关系图谱
- 理解记忆演化机制

### 📚 学习内容

#### Week 1: 记忆引擎核心

**Day 1-3: AuroraMemory 主引擎**
**阅读文件**: `aurora/core/memory/engine.py`（前 200 行）

**学习重点**:
- [ ] AuroraMemory 类的结构
- [ ] 核心数据结构：plots, stories, themes
- [ ] VectorIndex 和 MemoryGraph
- [ ] ingest() 方法的主流程

**代码阅读练习**:
```python
# 找到 AuroraMemory.ingest() 方法，回答：
# 1. 它接收什么参数？
# 2. 它调用了哪些子方法？
# 3. 它返回什么结果？

# 在代码中找到这些关键步骤：
# - 创建 Plot
# - 更新向量索引
# - 分配到 Story
# - 更新关系图
```

**实践任务**:
```python
# 练习17: 直接使用 AuroraMemory
from aurora.core import AuroraMemory
from aurora.core.models import MemoryConfig
from aurora.integrations.embeddings.local_semantic import LocalSemanticEmbedding

# 创建记忆引擎
config = MemoryConfig()
embedder = LocalSemanticEmbedding()
memory = AuroraMemory(config=config, embedder=embedder)

# 摄入记忆
result = memory.ingest(
    event_id="evt_001",
    user_msg="我喜欢 Python",
    agent_msg="Python 很棒！",
    actors=["user", "agent"],
)

print(f"创建的 Plot ID: {result.plot_id}")
print(f"分配到的 Story ID: {result.story_id}")
```

**Day 4-7: 向量检索与关系图谱**
**阅读文件**:
- `aurora/core/graph/vector_index.py`
- `aurora/core/graph/memory_graph.py`

**学习重点**:
- [ ] VectorIndex 如何存储和检索向量
- [ ] 余弦相似度的计算
- [ ] MemoryGraph 如何存储关系
- [ ] 图的邻居查询

**数学基础**:
```python
# 练习18: 理解余弦相似度
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 测试
v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([2.0, 3.0, 4.0])
similarity = cosine_similarity(v1, v2)
print(f"相似度: {similarity:.3f}")

# 思考：为什么 Aurora 使用余弦相似度而不是欧氏距离？
```

**实践任务**:
```python
# 练习19: 使用 VectorIndex
from aurora.core.graph.vector_index import VectorIndex
import numpy as np

# 创建索引
index = VectorIndex(dim=3)

# 添加向量
index.add("vec1", np.array([1.0, 0.0, 0.0]))
index.add("vec2", np.array([0.9, 0.1, 0.0]))
index.add("vec3", np.array([0.0, 1.0, 0.0]))

# 查询最近邻
query = np.array([0.95, 0.05, 0.0])
neighbors = index.search(query, k=2)

print("最近邻:")
for id, score in neighbors:
    print(f"  {id}: {score:.3f}")
```

#### Week 2: 高级特性

**Day 8-10: 记忆演化**
**阅读文件**: `aurora/core/memory/evolution.py`

**学习重点**:
- [ ] 记忆如何随时间演化
- [ ] 压力衰减机制
- [ ] 记忆的强化和遗忘

**概念理解**:
```
记忆演化的三个机制：

1. 强化（Reinforcement）
   - 相似的新记忆会强化旧记忆
   - 增加记忆的"强度"

2. 衰减（Decay）
   - 长时间未访问的记忆会衰减
   - 降低记忆的"强度"

3. 整合（Consolidation）
   - 多个相似记忆整合成更抽象的记忆
   - Plot → Story → Theme
```

**Day 11-14: 一致性检查**
**阅读文件**: `aurora/core/coherence.py`

**学习重点**:
- [ ] CoherenceGuardian 的作用
- [ ] 如何检测记忆冲突
- [ ] 冲突类型：事实冲突、时间冲突、逻辑冲突

**实践任务**:
```python
# 练习20: 理解记忆冲突
from aurora.core.coherence import CoherenceGuardian, ConflictType

# 思考以下场景：
# 记忆1: "用户说他喜欢 Python"
# 记忆2: "用户说他不喜欢编程"
#
# 这是什么类型的冲突？
# Aurora 如何处理这种冲突？

# 阅读 CoherenceGuardian.check_conflicts() 方法
# 理解冲突检测的逻辑
```

### ✅ 自我检测
完成以下综合练习：

```python
# 综合练习: 实现一个简化版的记忆引擎
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import time

@dataclass
class SimplePlot:
    id: str
    content: str
    embedding: np.ndarray
    timestamp: float

class SimpleMemoryEngine:
    """简化版的记忆引擎"""

    def __init__(self):
        self.plots: Dict[str, SimplePlot] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    def ingest(self, plot_id: str, content: str, embedding: np.ndarray) -> None:
        """摄入一个记忆"""
        plot = SimplePlot(
            id=plot_id,
            content=content,
            embedding=embedding,
            timestamp=time.time()
        )
        self.plots[plot_id] = plot
        self.embeddings[plot_id] = embedding

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """检索最相似的记忆"""
        results = []
        for plot_id, emb in self.embeddings.items():
            # 计算余弦相似度
            similarity = np.dot(query_embedding, emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(emb)
            )
            results.append((plot_id, similarity))

        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

# 测试
engine = SimpleMemoryEngine()
engine.ingest("p1", "我喜欢 Python", np.array([1.0, 0.0, 0.0]))
engine.ingest("p2", "我喜欢编程", np.array([0.9, 0.1, 0.0]))
engine.ingest("p3", "我喜欢音乐", np.array([0.0, 1.0, 0.0]))

query = np.array([0.95, 0.05, 0.0])
results = engine.retrieve(query, k=2)
print("检索结果:")
for plot_id, score in results:
    print(f"  {plot_id}: {engine.plots[plot_id].content} (score: {score:.3f})")
```

**通过标准**: 理解向量检索的原理，能够实现简化版的记忆引擎。

---

## 第6阶段: 实战项目 (2周)

### 🎯 学习目标
- 独立完成一个基于 Aurora 的项目
- 深化对 Aurora 的理解
- 培养解决实际问题的能力

### 📚 项目选择

选择以下项目之一（或自己设计）：

#### 项目1: 个人学习助手
**功能**:
- 记录每天的学习内容
- 根据学习历史推荐复习内容
- 生成学习报告

**技术要点**:
- 使用 Aurora 存储学习记录
- 实现基于时间的记忆检索
- 可视化学习进度

#### 项目2: 智能笔记系统
**功能**:
- 记录笔记和想法
- 自动关联相关笔记
- 发现知识之间的联系

**技术要点**:
- 使用 Aurora 的关系图谱
- 实现笔记的语义搜索
- 可视化知识网络

#### 项目3: 对话机器人
**功能**:
- 记住与用户的对话历史
- 根据历史调整回复风格
- 理解用户的偏好

**技术要点**:
- 使用 Aurora 的 Story 和 Theme
- 实现个性化回复
- 集成 LLM API

### 📝 项目实施步骤

#### Week 1: 设计与实现

**Day 1-2: 需求分析与设计**
- [ ] 明确项目目标
- [ ] 设计数据模型
- [ ] 画出系统架构图
- [ ] 列出功能清单

**Day 3-5: 核心功能实现**
- [ ] 搭建项目框架
- [ ] 集成 Aurora
- [ ] 实现核心功能
- [ ] 编写单元测试

**Day 6-7: 功能完善**
- [ ] 添加错误处理
- [ ] 优化用户体验
- [ ] 编写文档

#### Week 2: 测试与优化

**Day 8-10: 测试与调试**
- [ ] 功能测试
- [ ] 性能测试
- [ ] 修复 bug

**Day 11-12: 优化与重构**
- [ ] 代码重构
- [ ] 性能优化
- [ ] 添加日志

**Day 13-14: 总结与展示**
- [ ] 编写项目文档
- [ ] 准备演示
- [ ] 总结学习心得

### 📋 项目模板

```python
# 项目结构示例
my_aurora_project/
├── README.md           # 项目说明
├── requirements.txt    # 依赖列表
├── config.py          # 配置文件
├── main.py            # 主程序
├── models.py          # 数据模型
├── aurora_wrapper.py  # Aurora 封装
└── tests/             # 测试文件
    └── test_main.py

# main.py 示例
from aurora import AuroraRuntime, AuroraSettings
from typing import List, Dict

class MyAuroraApp:
    def __init__(self, data_dir: str):
        self.runtime = AuroraRuntime(
            settings=AuroraSettings(
                data_dir=data_dir,
                llm_provider="mock",
                embedding_provider="local",
            )
        )

    def add_memory(self, content: str) -> str:
        """添加一条记忆"""
        # 实现你的逻辑
        pass

    def search_memory(self, query: str, k: int = 5) -> List[Dict]:
        """搜索记忆"""
        # 实现你的逻辑
        pass

    def get_summary(self) -> Dict:
        """获取记忆摘要"""
        # 实现你的逻辑
        pass

if __name__ == "__main__":
    app = MyAuroraApp(data_dir="./my_data")
    # 运行你的应用
```

### ✅ 项目评估标准

**功能完整性** (30分):
- [ ] 核心功能完整实现
- [ ] 错误处理完善
- [ ] 用户体验良好

**代码质量** (30分):
- [ ] 代码结构清晰
- [ ] 命名规范
- [ ] 有适当的注释
- [ ] 有单元测试

**Aurora 集成** (20分):
- [ ] 正确使用 Aurora API
- [ ] 充分利用 Aurora 特性
- [ ] 理解 Aurora 的设计理念

**文档与展示** (20分):
- [ ] README 清晰完整
- [ ] 有使用说明
- [ ] 能够演示项目

**通过标准**: 总分 >= 70 分

---

## 📚 学习资源推荐

### Python 基础
- **书籍**: 《Python编程：从入门到实践》
- **在线课程**: Python 官方教程 (docs.python.org)
- **练习平台**: LeetCode Python 题库

### Python 进阶
- **书籍**: 《流畅的Python》
- **文档**: Python typing 模块文档
- **文章**: Real Python 网站

### 数学基础
- **线性代数**: 3Blue1Brown 的线性代数系列视频
- **概率论**: Khan Academy 概率论课程
- **向量检索**: 理解余弦相似度和欧氏距离

### 软件工程
- **设计模式**: 《Head First 设计模式》
- **代码整洁**: 《代码整洁之道》
- **测试**: pytest 官方文档

---

## 🎓 学习日志模板

每天学习后，填写学习日志（建立元认知）：

```markdown
# 学习日志 - YYYY-MM-DD

## 今天学习的内容
-

## 遇到的问题
-

## 解决方案
-

## 新的理解
-

## 明天的计划
-

## 自我评估
- 理解程度: ☆☆☆☆☆ (1-5星)
- 实践程度: ☆☆☆☆☆ (1-5星)
- 需要复习: 是/否
```

---

## 🏆 学习里程碑

完成以下里程碑，你就掌握了 Aurora：

- [ ] **里程碑1**: 完成 Python 基础学习，能够编写简单程序
- [ ] **里程碑2**: 理解 Aurora 的核心概念和设计哲学
- [ ] **里程碑3**: 掌握 Python 高级特性，能够阅读 Aurora 代码
- [ ] **里程碑4**: 理解 Aurora 的数据模型，能够操作 Plot/Story/Theme
- [ ] **里程碑5**: 理解 Aurora 运行时，能够使用 Aurora API
- [ ] **里程碑6**: 理解 Aurora 核心算法，能够修改和扩展
- [ ] **里程碑7**: 完成实战项目，能够独立开发基于 Aurora 的应用

---

## 💡 学习建议

### 认知科学原理应用

**1. 间隔重复**
- 每学完一个阶段，隔天复习
- 每周末复习本周内容
- 每月复习整个月的内容

**2. 主动学习**
- 不要只看代码，要动手写
- 不要只做练习，要思考为什么
- 不要只学习，要教别人（费曼学习法）

**3. 具体到抽象**
- 先看例子，再理解原理
- 先用 API，再看实现
- 先模仿，再创新

**4. 多模态学习**
- 阅读代码
- 画图理解
- 动手实践
- 讲给别人听

**5. 认知负荷管理**
- 每天学习 2-3 小时，不要贪多
- 遇到困难时，休息一下
- 感到疲劳时，停止学习

### 遇到困难时

**如果代码看不懂**:
1. 先看函数签名和文档字符串
2. 画出数据流图
3. 用调试器单步执行
4. 简化问题，写个小例子

**如果概念不理解**:
1. 找类比（用生活中的例子）
2. 画图可视化
3. 讲给别人听（或讲给橡皮鸭）
4. 查找相关资料

**如果进度太慢**:
1. 不要着急，学习需要时间
2. 回顾学习日志，看看进步
3. 调整学习计划，降低难度
4. 寻求帮助（社区、导师）

---

## 🎉 结语

恭喜你选择学习 Aurora！这是一个充满挑战但非常有价值的旅程。

记住：
- **学习是马拉松，不是短跑**
- **理解比记忆更重要**
- **实践比理论更重要**
- **坚持比天赋更重要**

祝你学习愉快！🚀

---

## 📞 获取帮助

如果在学习过程中遇到问题：
1. 查看 Aurora 的 README 和文档
2. 阅读代码注释和文档字符串
3. 在 GitHub Issues 提问
4. 加入开发者社区

记住：**提问是学习的一部分，不要害怕问问题！**
