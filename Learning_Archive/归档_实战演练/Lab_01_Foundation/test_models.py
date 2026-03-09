import os
import sys
import time
import pytest
from pydantic import ValidationError

# 把当前目录临时加进环境变量防查错
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 强行导入学员即将补全的模块
try:
    from task_models import MicroSettings, MicroPlot, get_fresh_time
except ImportError as e:
    pytest.fail(f"!!! 无法导入 task_models.py，请确保文件存在且没有致命语法错误: {e} !!!")

# =====================================================================
# Bootcamp Lab 01: 自动化护城河测试机
# =====================================================================

def test_pydantic_env_prefix_and_ignore():
    """测试 TODO 1: 能够正确抓取 MICRO_ 前缀和无视陌生变量"""
    os.environ["MICRO_LLM_ROUTER"] = "openai"
    os.environ["MICRO_UNKNOWN_GARBAGE"] = "123" # 测试额外字段是否被忽略
    
    try:
        settings = MicroSettings()
    except ValidationError as e:
        pytest.fail(f"你的 Pydantic 配置拦截了不该拦截的变量！错误: {e}")
    finally:
        os.environ.pop("MICRO_LLM_ROUTER", None)
        os.environ.pop("MICRO_UNKNOWN_GARBAGE", None)
        
    assert settings.model_config.get("env_prefix") == "MICRO_", "[FAIL] TODO 1: 你没有正确设置 env_prefix='MICRO_'"
    assert settings.model_config.get("extra") == "ignore", "[FAIL] TODO 1: 你没有正确设置 extra='ignore'"

def test_pydantic_literal_constraint():
    """测试 TODO 2: 字面量强校验与默认值"""
    s1 = MicroSettings()
    assert s1.llm_router == "mock", "[FAIL] TODO 2: 默认值没有被设置为 'mock'"
    
    # 测试恶意环境入侵
    os.environ["MICRO_LLM_ROUTER"] = "hacker_model"
    with pytest.raises(ValidationError) as exc:
        s2 = MicroSettings()
    os.environ.pop("MICRO_LLM_ROUTER", None)
    
    assert "Input should be" in str(exc.value), "[FAIL] TODO 2: 没有成功拦截非法的 'hacker_model'！"


def test_dataclass_decorator_existence():
    """测试 TODO 3: 是否正确加上了 @dataclass 装饰器"""
    assert hasattr(MicroPlot, "__dataclass_fields__"), "[FAIL] TODO 3: 你忘记给 MicroPlot 戴上魔法装饰器帽子了！"


def test_tuple_immutability():
    """测试 TODO 4: 必须使用 Tuple 保证不可篡改性"""
    plot = MicroPlot(content="hello", actors=("Alice", "Bob"))
    
    # 如果它是列表，这就成功了，但这在我们的架构里是灾难。
    assert isinstance(plot.actors, tuple), "[FAIL] TODO 4: 演员名单必须使用 tuple (元组) 类型，防止后期被人暗中修改！"


def test_fresh_timestamp():
    """测试 TODO 5: 致命的 default_factory 陷阱！时间戳必须是绝对应激鲜活的！"""
    plot1 = MicroPlot(content="1", actors=("Me",))
    time.sleep(0.1)
    plot2 = MicroPlot(content="2", actors=("You",))
    
    # 要求在没有手动传入参数时，两个自带生成的时间戳不能相等，也不能差太远
    assert plot1.created_ts != plot2.created_ts, "[FAIL] TODO 5: 你掉进了深坑！所有的 Plot 都共用了一个过去的时间。请使用 field(default_factory=...)"
    assert plot2.created_ts > plot1.created_ts, "[FAIL] TODO 5: 时光倒流？第二个生成的时间戳居然比第一个早？"

