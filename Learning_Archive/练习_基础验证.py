# 文件名: 练习_基础验证.py (实操演示)

from aurora.exceptions import AuroraError

# 1. 创建自定义的异常类，继承自 Aurora 的基础报错
class EmptyInputError(AuroraError):
    def __init__(self, provided_user_name: str):
        self.provided_user_name = provided_user_name
        # 调用父类的初始化，并传入最终要在屏幕上打印的消息
        super().__init__(f"{provided_user_name}，你不能传入空的对话文本！")

# 2. 模拟系统核心处理前的数据检查
def process_message(user_name: str, message: str):
    print(f"正在分析 {user_name} 输入的内容: '{message}'")
    # 工业代码最常见的防御性编程（Defensive Programming）
    # strip() 会去除字符串首尾的空格，用来防范一排空白的情况
    if not message.strip():
        # 主动中断，并抛出我们刚刚编写的带参类对象
        raise EmptyInputError(provided_user_name=user_name)
    print("分析成功！")

# 3. 稳妥地运行代码（不用 try 的话，程序遇到错误会强行崩溃并退出终端）
print("\n--- 第一轮正常测试 ---")
process_message("小白", "今天天气真好")

print("\n--- 第二轮异常测试 ---")
try:
    process_message("小白", "   ") # 传入了全空格字符串
except EmptyInputError as e:
    # 捕获到了，程序没崩溃！并顺利打印出了对象的附带信息
    print(f"安全拦截到预期内错误: '{e}'")
