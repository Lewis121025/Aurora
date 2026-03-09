# 文件名: 练习_配置管理.py (实操演示)
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
