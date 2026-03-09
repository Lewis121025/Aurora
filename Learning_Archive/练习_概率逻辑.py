# 文件名: 练习_概率逻辑.py (实操演示)
import numpy as np

# 我们假设 AI 内心对三种回答方式的基础打分非常接近（死板的 AI 会在这个时候当机断片）
# A:热情回答 (10分), B:高冷回答 (9分), C:阴阳怪气 (8分)
raw_scores = np.array([10.0, 9.0, 8.0])

# 1. 极其著名的 Softmax 算子 (将所有分数进行自然底数 e 的指数爆炸)
def softmax(x):
    # 防止数字太大溢出，先整体减去最大值（数学极其优雅的稳定操作）
    e_x = np.exp(x - np.max(x))
    # 每一个人占总指数爆炸和的百分比
    return e_x / e_x.sum(axis=0)

probability_distribution = softmax(raw_scores)

print("--- 见证 Softmax 拉卡斯极权指数放大的魔力 ---")
print(f"原始分数极其相近: 热情 {raw_scores[0]} | 高冷 {raw_scores[1]} | 阴阳怪气 {raw_scores[2]}")
print(f"经过 Softmax 极速放大后的【绝对百分比概率】:")
print(f"  🔥 热情脱颖而出:   {probability_distribution[0]*100:.1f} %")
print(f"  ❄️ 高冷落后一大截: {probability_distribution[1]*100:.1f} %")
print(f"  🎭 阴阳被极速压缩: {probability_distribution[2]*100:.1f} %")

print("\n--- 并非死板选第一 (Exploration vs Exploitation) ---")
# 2. 我们根据这个百分比摇骰子，模拟 AI 的每一次开口！
for i in range(1, 6):
    # choice 会根据 p=参数 给定的权重骰子，摇出一个最终选项
    chosen_idx = np.random.choice(len(raw_scores), p=probability_distribution)
    choices = ["热情", "高冷", "阴阳怪气"]
    print(f"第 {i} 次遇到客户，AI 选择了: {choices[chosen_idx]}")
