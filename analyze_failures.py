#!/usr/bin/env python3
"""
LongMemEval 失败案例深度分析
============================

从第一性原理分析每种失败类型的根本原因：
1. knowledge-update: 20% - 检测/存储/检索链路问题
2. temporal-reasoning: 60% - 时序处理问题
3. single-session-preference: 60% - 偏好存储问题
"""

import sys
import os
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.algorithms.constants import (
    UPDATE_HIGH_SIMILARITY_THRESHOLD,
    UPDATE_MODERATE_SIMILARITY_THRESHOLD,
    CONFLICT_CHECK_SIMILARITY_THRESHOLD,
)
from aurora.embeddings.bailian import BailianEmbedding
from aurora.utils.math_utils import cosine_sim


def create_test_cases():
    """创建模拟 LongMemEval 的测试用例"""
    return {
        "knowledge-update": [
            {
                "session": [
                    "User: I live in Beijing.",
                    "Assistant: Got it, you're based in Beijing.",
                    "User: I've moved to Shanghai now.",
                    "Assistant: Noted, you're now in Shanghai.",
                ],
                "question": "Where do I live?",
                "expected": "shanghai",
                "old_value": "beijing",
            },
            {
                "session": [
                    "User: My phone number is 123-456-7890.",
                    "Assistant: I've saved your number.",
                    "User: Actually, I changed my number to 098-765-4321.",
                    "Assistant: Updated to the new number.",
                ],
                "question": "What's my phone number?",
                "expected": "098-765-4321",
                "old_value": "123-456-7890",
            },
            {
                "session": [
                    "User: I work at Google.",
                    "Assistant: That's a great company!",
                    "User: I quit Google and joined Microsoft last month.",
                    "Assistant: Congratulations on the new job!",
                ],
                "question": "Where do I work?",
                "expected": "microsoft",
                "old_value": "google",
            },
        ],
        "temporal-reasoning": [
            {
                "session": [
                    "User: I learned Python in 2020.",
                    "User: Then I learned JavaScript in 2021.",
                    "User: Most recently, I started learning Rust in 2023.",
                ],
                "question": "What programming language did I learn most recently?",
                "expected": "rust",
            },
            {
                "session": [
                    "User: First, I visited Paris.",
                    "User: Then I went to London.",
                    "User: Finally, I traveled to Tokyo.",
                ],
                "question": "Which city did I visit first?",
                "expected": "paris",
            },
        ],
        "single-session-preference": [
            {
                "session": [
                    "User: I prefer coffee over tea.",
                    "User: I love reading science fiction books.",
                    "User: I'm into hiking and outdoor activities.",
                ],
                "question": "Do I prefer coffee or tea?",
                "expected": "coffee",
            },
            {
                "session": [
                    "User: I like working from home.",
                    "User: I enjoy cooking Italian food.",
                    "User: I'm a morning person, I wake up early.",
                ],
                "question": "Am I a morning or evening person?",
                "expected": "morning",
            },
        ],
    }


def analyze_knowledge_update(embedder, config):
    """深度分析 knowledge-update 失败原因"""
    print("\n" + "=" * 70)
    print("KNOWLEDGE-UPDATE 深度分析")
    print("=" * 70)
    
    test_cases = create_test_cases()["knowledge-update"]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'─' * 60}")
        print(f"测试案例 {i+1}")
        print(f"{'─' * 60}")
        
        memory = AuroraMemory(cfg=config, seed=42, embedder=embedder)
        
        # 记录详细信息
        old_value_plot_id = None
        new_value_plot_id = None
        
        # Ingest 所有对话
        for turn in case["session"]:
            # Extract actors from the message prefix
            actors = ("user", "agent")  # default
            if turn.lower().startswith("user:"):
                actors = ("user",)
            elif turn.lower().startswith("assistant:"):
                actors = ("assistant",)
            
            plot = memory.ingest(turn, actors=actors)
            print(f"\n[INGEST] {turn[:50]}...")
            print(f"  - plot.id: {plot.id[:16]}...")
            print(f"  - plot.actors: {plot.actors}")
            print(f"  - plot.redundancy_type: {plot.redundancy_type}")
            print(f"  - plot.supersedes_id: {plot.supersedes_id}")
            print(f"  - stored: {plot.id in memory.plots}")
            
            # 记录旧值和新值的 plot
            if case["old_value"].lower() in turn.lower():
                old_value_plot_id = plot.id
            if case["expected"].lower() in turn.lower():
                new_value_plot_id = plot.id
        
        # 分析 1：语义相似度检查
        print(f"\n[分析 1] 语义相似度检查")
        old_turn = [t for t in case["session"] if case["old_value"].lower() in t.lower()][0]
        new_turn = [t for t in case["session"] if case["expected"].lower() in t.lower()][0]
        
        old_emb = embedder.embed(old_turn)
        new_emb = embedder.embed(new_turn)
        similarity = cosine_sim(old_emb, new_emb)
        
        print(f"  旧信息: {old_turn[:40]}...")
        print(f"  新信息: {new_turn[:40]}...")
        print(f"  语义相似度: {similarity:.3f}")
        print(f"  UPDATE_HIGH_THRESHOLD: {UPDATE_HIGH_SIMILARITY_THRESHOLD}")
        print(f"  UPDATE_MODERATE_THRESHOLD: {UPDATE_MODERATE_SIMILARITY_THRESHOLD}")
        
        if similarity < UPDATE_MODERATE_SIMILARITY_THRESHOLD:
            print(f"  ⚠️ 问题：相似度 < {UPDATE_MODERATE_SIMILARITY_THRESHOLD}，更新检测被跳过!")
        elif similarity < UPDATE_HIGH_SIMILARITY_THRESHOLD:
            print(f"  ⚠️ 问题：相似度 < {UPDATE_HIGH_SIMILARITY_THRESHOLD}，仅进入reinforcement检测")
        else:
            print(f"  ✓ 相似度足够高，应该触发更新检测")
        
        # 分析 2：存储状态检查
        print(f"\n[分析 2] 存储状态检查")
        print(f"  总 plots 数量: {len(memory.plots)}")
        
        if old_value_plot_id and old_value_plot_id in memory.plots:
            old_plot = memory.plots[old_value_plot_id]
            print(f"  旧值 plot 状态: {old_plot.status}")
            print(f"  旧值被更新标记: {old_plot.superseded_by_id is not None}")
        else:
            print(f"  旧值 plot 未存储或ID丢失")
        
        if new_value_plot_id and new_value_plot_id in memory.plots:
            new_plot = memory.plots[new_value_plot_id]
            print(f"  新值 plot 状态: {new_plot.status}")
            print(f"  新值 supersedes_id: {new_plot.supersedes_id}")
        else:
            print(f"  新值 plot 未存储或ID丢失")
        
        # 分析 3：检索结果
        print(f"\n[分析 3] 检索结果")
        trace = memory.query(case["question"], k=5)
        
        print(f"  Query: {case['question']}")
        print(f"  Expected: {case['expected']}")
        
        found_new = False
        found_old = False
        
        for rank, (pid, score, kind) in enumerate(trace.ranked):
            plot = memory.plots.get(pid)
            if plot:
                is_new = case["expected"].lower() in plot.text.lower()
                is_old = case["old_value"].lower() in plot.text.lower()
                
                marker = ""
                if is_new:
                    found_new = True
                    marker = " [NEW VALUE ✓]"
                elif is_old:
                    found_old = True
                    marker = " [OLD VALUE ✗]"
                
                print(f"  #{rank+1} (score={score:.3f}, status={plot.status}): {plot.text[:40]}...{marker}")
        
        # 根本原因判定
        print(f"\n[根本原因判定]")
        if not found_new and not found_old:
            print("  ROOT CAUSE: 相关信息未被存储 (encoding gate 问题)")
        elif found_old and not found_new:
            print("  ROOT CAUSE: 只检索到旧值，新值未存储或排名过低")
        elif found_new and found_old:
            for rank, (pid, score, kind) in enumerate(trace.ranked):
                plot = memory.plots.get(pid)
                if plot and case["old_value"].lower() in plot.text.lower():
                    old_rank = rank + 1
                    break
            for rank, (pid, score, kind) in enumerate(trace.ranked):
                plot = memory.plots.get(pid)
                if plot and case["expected"].lower() in plot.text.lower():
                    new_rank = rank + 1
                    break
            if old_rank < new_rank:
                print(f"  ROOT CAUSE: 旧值(#{old_rank})排名高于新值(#{new_rank})，旧值未被supersede或未被过滤")
            else:
                print(f"  ✓ 新值(#{new_rank})排名高于旧值(#{old_rank})，应该能正确回答")
        elif found_new:
            print("  ✓ 只检索到新值，应该能正确回答")


def analyze_temporal_reasoning(embedder, config):
    """深度分析 temporal-reasoning 失败原因"""
    print("\n" + "=" * 70)
    print("TEMPORAL-REASONING 深度分析")
    print("=" * 70)
    
    test_cases = create_test_cases()["temporal-reasoning"]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'─' * 60}")
        print(f"测试案例 {i+1}")
        print(f"{'─' * 60}")
        
        memory = AuroraMemory(cfg=config, seed=42, embedder=embedder)
        
        # Ingest with artificial time gaps
        import time
        timestamps = []
        for j, turn in enumerate(case["session"]):
            plot = memory.ingest(turn)
            timestamps.append(plot.ts)
            print(f"\n[INGEST] {turn[:50]}...")
            print(f"  - plot.id: {plot.id[:16]}...")
            print(f"  - plot.ts: {plot.ts}")
            time.sleep(0.01)  # 小延迟确保时间戳不同
        
        # 分析 1：时间戳分布
        print(f"\n[分析 1] 时间戳分布")
        print(f"  所有时间戳相差: {max(timestamps) - min(timestamps):.6f} 秒")
        print(f"  问题: 所有对话使用 now_ts()，无法区分历史时间")
        
        # 分析 2：查询类型检测
        from aurora.algorithms.retrieval.field_retriever import QueryType
        detected_type = memory.retriever._classify_query(case["question"])
        print(f"\n[分析 2] 查询类型检测")
        print(f"  Query: {case['question']}")
        print(f"  Detected type: {detected_type}")
        
        if detected_type != QueryType.TEMPORAL:
            print(f"  ⚠️ 问题：未检测为 TEMPORAL 查询!")
        
        # 分析 3：检索结果
        trace = memory.query(case["question"], k=5)
        print(f"\n[分析 3] 检索结果")
        
        for rank, (pid, score, kind) in enumerate(trace.ranked):
            plot = memory.plots.get(pid)
            if plot:
                is_answer = case["expected"].lower() in plot.text.lower()
                marker = " [EXPECTED ✓]" if is_answer else ""
                print(f"  #{rank+1} (ts={plot.ts:.2f}, score={score:.3f}): {plot.text[:40]}...{marker}")


def analyze_preference(embedder, config):
    """深度分析 single-session-preference 失败原因"""
    print("\n" + "=" * 70)
    print("SINGLE-SESSION-PREFERENCE 深度分析")
    print("=" * 70)
    
    test_cases = create_test_cases()["single-session-preference"]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'─' * 60}")
        print(f"测试案例 {i+1}")
        print(f"{'─' * 60}")
        
        memory = AuroraMemory(cfg=config, seed=42, embedder=embedder)
        
        # Ingest 所有对话
        stored_count = 0
        for turn in case["session"]:
            plot = memory.ingest(turn)
            is_stored = plot.id in memory.plots
            if is_stored:
                stored_count += 1
            
            print(f"\n[INGEST] {turn[:50]}...")
            print(f"  - stored: {is_stored}")
            print(f"  - knowledge_type: {plot.knowledge_type}")
            print(f"  - knowledge_confidence: {plot.knowledge_confidence:.2f}")
            print(f"  - _storage_prob: {getattr(plot, '_storage_prob', 'N/A')}")
        
        print(f"\n[存储统计] {stored_count}/{len(case['session'])} plots 被存储")
        
        # 检索结果
        trace = memory.query(case["question"], k=5)
        print(f"\n[检索结果]")
        print(f"  Query: {case['question']}")
        print(f"  Expected: {case['expected']}")
        
        for rank, (pid, score, kind) in enumerate(trace.ranked):
            plot = memory.plots.get(pid)
            if plot:
                is_answer = case["expected"].lower() in plot.text.lower()
                marker = " [EXPECTED ✓]" if is_answer else ""
                print(f"  #{rank+1}: {plot.text[:40]}...{marker}")


def main():
    """主函数"""
    print("=" * 70)
    print("LongMemEval 失败案例第一性原理分析")
    print("=" * 70)
    
    # 创建 embedder
    api_key = os.getenv('AURORA_BAILIAN_API_KEY')
    if not api_key:
        print("Error: AURORA_BAILIAN_API_KEY not set")
        return
    
    embedder = BailianEmbedding(
        api_key=api_key,
        model='text-embedding-v4',
        dimension=1024,
    )
    
    config = MemoryConfig(dim=1024, max_plots=5000)
    
    # 运行分析
    analyze_knowledge_update(embedder, config)
    analyze_temporal_reasoning(embedder, config)
    analyze_preference(embedder, config)
    
    # 总结
    print("\n" + "=" * 70)
    print("根本原因总结")
    print("=" * 70)
    print("""
    1. KNOWLEDGE-UPDATE 失败根因:
       - 更新检测依赖高语义相似度 (>0.75)，但知识更新通常语义不相似
       - 检索时未过滤已被 supersede 的 plots
       - 冲突检测门槛过高 (>0.65)
    
    2. TEMPORAL-REASONING 失败根因:
       - 所有对话使用 now_ts()，无法记录历史时间
       - 时序查询类型检测可能不准确
       - 缺乏显式时间戳支持
    
    3. SINGLE-SESSION-PREFERENCE 失败根因:
       - 偏好存储权重较低 (0.6)
       - 偏好关键词检测不全
       - VOI 决策可能跳过偏好信息
    """)


if __name__ == "__main__":
    main()
