# 文件名: 练习_向量计算.py (实操演示)
import numpy as np

# 我们假设大模型把单词翻译成了二维坐标（X代表"毛绒度", Y代表"体型"）
# 比如："猫" 是毛茸茸的且体型小 -> [0.9, 0.2]
#       "狗" 是毛茸茸的且体型中 -> [0.8, 0.5]
#       "汽车" 是光秃秃的且体型大 -> [0.1, 0.9]

word_vectors = {
    "猫": np.array([0.9, 0.2]),
    "狗": np.array([0.8, 0.5]),
    "汽车": np.array([0.1, 0.9])
}

# 这是一个极其核心的大厂算法公式：余弦相似度 (Cosine Similarity)
# 它的作用就是：就算两个词向量在宇宙里相隔很远，但只要它们指着同一个方向，它们就是相似的！
def cosine_similarity(vec_a, vec_b):
    # 点乘：两个向量内积
    dot_product = np.dot(vec_a, vec_b)
    # 算长度（也就是几何的勾股定理）
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    # 方向一致性 = 点乘 / 两者长度的乘积
    return dot_product / (norm_a * norm_b)

print("--- AI 脑颅内部的几何测距 ---")
sim_cat_dog = cosine_similarity(word_vectors["猫"], word_vectors["狗"])
print(f"【猫】和【狗】的语义相似度 (余弦夹角): {sim_cat_dog:.3f} (满分1.0，非常高！)")

sim_cat_car = cosine_similarity(word_vectors["猫"], word_vectors["汽车"])
print(f"【猫】和【汽车】的语义相似度 (余弦夹角): {sim_cat_car:.3f} (两者是不同物种！)")
