# 文件名: 练习_数据类.py (实操演示)
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
    plot_1.actors[0] = "Hacker" # type: ignore
except TypeError as e:
    print(f"篡改历史失败（类型保护生效）: {e}")
