import numpy as np
import random

# =====================================================================
# Agent Algorithm 特训 01: 感知与遗忘机制
# 锚定原版源码：aurora/core/memory/engine.py
# =====================================================================

class MicroMemoryEngine:
    def __init__(self, novelty_weight: float = 0.6, challenge_weight: float = 0.4):
        self.novelty_weight = novelty_weight
        self.challenge_weight = challenge_weight
        self.rng = random.Random(42) # 固定随机种子以便单元测试通过

    def compute_cosine_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        TODO 1: 算法核心构建 - 余弦相似度
        要求：不要调用外部高级库。请使用 numpy 的基础运算，基于两个高维数组，计算它们的余弦相似度。
        注意底部的除法保护，防止分母为 0！
        提示：相似度 = (A 点乘 B) / (A 的总长度 * B 的总长度)
        """
        # >>> 请在此处补全代码
        pass


    def calculate_voi(self, memory_emb: np.ndarray, past_embs: list[np.ndarray]) -> float:
        """
        TODO 2: 信息价值评分 (Value of Information)
        输入一句话的向量，和过往系统里的历史向量列表。
        要求：
        1. 遍历 past_embs，测算当前语句和过去每句话的 cosine_distance。
        2. "新颖度 (novelty)" = 1.0 - (过去所有历史的最大相似度)。也就是说，如果它和过去极其不像（最高相似度都很低），那这波新鲜感就拉满！
        3. 最终的 VoI = novelty * self.novelty_weight
        （在此极简模型中，忽略 challenge 维度，只算新颖度基础盘）
        """
        # >>> 请在此处补全代码
        pass


    def should_store_memory(self, voi_score: float) -> bool:
        """
        TODO 3: 概率化柔性截断 (AI 的机缘潜意识遗忘)
        切记：绝对不能写成 if voi_score > 0.5 返回 True！
        要求：
        使用 self.rng.random() 生成 0 到 1 的随机数。如果随机数小于等于这个 voi_score，就判定为应当存储 (True) 并在脑海里存入。否则丢弃 (False)。
        """
        # >>> 请在此处补全代码
        pass

