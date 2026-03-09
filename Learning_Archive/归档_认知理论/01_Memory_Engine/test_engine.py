import sys
import os
import numpy as np
import pytest

# 把当前目录临时加进环境变量防查错
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from task_engine import MicroMemoryEngine
except ImportError as e:
    pytest.fail(f"!!! 无法导入 task_engine.py: {e} !!!")


def test_cosine_distance():
    """测试 TODO 1: 原生余弦相似度算法推导"""
    engine = MicroMemoryEngine()
    
    # 两个完全一样的方向
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([2.0, 0.0])
    # 两个完全垂直的方向
    vec3 = np.array([0.0, 1.0])
    
    res1 = engine.compute_cosine_distance(vec1, vec2)
    assert res1 is not None, "[FAIL] TODO 1: compute_cosine_distance 没有返回值！"
    assert abs(res1 - 1.0) < 1e-5, f"[FAIL] TODO 1: 方向一样的情况下应该为 1.0，但你算出来的是 {res1}"
    
    res2 = engine.compute_cosine_distance(vec1, vec3)
    assert abs(res2 - 0.0) < 1e-5, f"[FAIL] TODO 1: 垂直向量余弦应为 0.0，但你算出来的是 {res2}"


def test_calculate_voi():
    """测试 TODO 2: VoI 前沿新颖度打分网络"""
    engine = MicroMemoryEngine(novelty_weight=0.8)  # 权重设置
    
    current_emb = np.array([1.0, 0.0])
    
    # 历史记录里有一个极其相似的 [0.9, 0.1]
    past_history_stale = [
        np.array([0.0, 1.0]), 
        np.array([0.9, 0.1])
    ]
    
    # 因为很相似，它的 Maximum Cosine 是 0.99388。那它的 novelty 就是 1 - 0.99388 = 0.006。
    # 最终 VoI = 0.006 * 0.8 ≈ 0.0048 (极低，被视为废话)
    score_stale = engine.calculate_voi(current_emb, past_history_stale)
    assert score_stale is not None, "[FAIL] TODO 2: 没有返回 VoI 分数"
    assert score_stale < 0.1, f"[FAIL] TODO 2: 当遇到一句烂梗废话时，VoI 应该压到极低，但你的得分异常高: {score_stale}"
    
    # 历史记录里全是不搭噶的东西
    past_history_fresh = [
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.0])
    ]
    # 最高相似度是 0.0 (甚至是负的)，novelty > 1.0。
    score_fresh = engine.calculate_voi(current_emb, past_history_fresh)
    assert score_fresh > 0.6, f"[FAIL] TODO 2: 遇到千百年罕见的新梗时，VoI 应该极度膨胀，但你算出来太低了: {score_fresh}"


def test_probabilistic_pruning():
    """测试 TODO 3: 机缘概率截断算子"""
    engine = MicroMemoryEngine()
    
    # 测试极端高分，应该极大概率被保存
    high_score = 0.99
    results = [engine.should_store_memory(high_score) for _ in range(10)]
    assert True in results, "[FAIL] TODO 3: 对于 0.99 分的高价值情报，你的系统居然全当垃圾扔了？？你是不是没调用 random() 或者逻辑反了？"

    # 测试极低分，这很重要，必须允许存在"极为小概率"通过的现象！
    low_score = 0.01 
    # 为了保证能撞上这个小概率，我测 1000 次（固定种子下结果是确定的）
    engine.rng.seed(42)
    kept_count = sum(1 for _ in range(1000) if engine.should_store_memory(low_score))
    
    assert kept_count > 0, "[FAIL] TODO 3: 你依然写成了硬截断！如果是 0.01 这种废话废话，跑 1000 次怎么也得能偶尔机缘巧合被留下几次吧？你的过滤网太死了！"
    assert kept_count < 100, f"[FAIL] TODO 3: 你的系统不对劲，虽然是机缘巧合放行，但对于 0.01 分的垃圾，放行率不能高达 10% 以上！目前的留存量是: {kept_count}"

