#!/usr/bin/env python3
"""
分析 single-session-assistant 准确率下降的原因

对比 baseline (73.2%) vs phase1 (60%) 的失败案例
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.insert(0, '.')

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.embeddings.bailian import BailianEmbedding
from dotenv import load_dotenv
load_dotenv()


def load_results():
    """加载结果文件"""
    with open('longmemeval_baseline.json', 'r') as f:
        baseline = json.load(f)
    
    with open('longmemeval_phase1_results.json', 'r') as f:
        phase1 = json.load(f)
    
    with open('data/longmemeval/longmemeval_oracle_new.json', 'r') as f:
        data = json.load(f)
    
    return baseline, phase1, data


def analyze_assistant_questions():
    """分析 assistant 类型问题的特点"""
    baseline, phase1, data = load_results()
    
    # 获取所有 assistant 问题
    assistant_items = [item for item in data if item['question_type'] == 'single-session-assistant']
    
    # Baseline 结果映射
    baseline_results = {}
    for r in baseline['detailed_results']:
        if r['question_type'] == 'single-session-assistant':
            baseline_results[r['question_id']] = r['is_correct']
    
    print("=" * 80)
    print("ASSISTANT QUESTION REGRESSION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal assistant questions: {len(assistant_items)}")
    print(f"Baseline accuracy: {baseline['by_type']['single-session-assistant']['accuracy']:.1f}%")
    print(f"Phase1 accuracy: {phase1['results']['single-session-assistant']['accuracy']:.1f}%")
    print(f"Regression: {baseline['by_type']['single-session-assistant']['accuracy'] - phase1['results']['single-session-assistant']['accuracy']:.1f}%")
    
    # 分析 baseline 中正确和错误的案例
    baseline_correct = [qid for qid, correct in baseline_results.items() if correct]
    baseline_incorrect = [qid for qid, correct in baseline_results.items() if not correct]
    
    print(f"\nBaseline: {len(baseline_correct)} correct, {len(baseline_incorrect)} incorrect")
    
    # 分析失败案例的特征
    print("\n" + "=" * 80)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 80)
    
    # 检查答案在哪个 turn (assistant vs user)
    print("\n1. Answer location analysis:")
    assistant_answer_count = 0
    user_answer_count = 0
    
    for item in assistant_items:
        qid = item['question_id']
        if qid in baseline_incorrect:
            # 找到答案所在的 turn
            for session in item['haystack_sessions']:
                for turn in session:
                    if turn.get('has_answer'):
                        if turn['role'] == 'assistant':
                            assistant_answer_count += 1
                        else:
                            user_answer_count += 1
                        break
    
    print(f"  Incorrect questions with answer in assistant turn: {assistant_answer_count}")
    print(f"  Incorrect questions with answer in user turn: {user_answer_count}")
    
    # 分析问题关键词
    print("\n2. Question keyword analysis:")
    keywords_to_check = {
        'remind': [],
        'recommend': [],
        'suggest': [],
        'mentioned': [],
        'said': [],
        'told': [],
        'what did you': [],
    }
    
    for item in assistant_items:
        qid = item['question_id']
        question_lower = item['question'].lower()
        is_correct = baseline_results.get(qid, None)
        
        for keyword in keywords_to_check.keys():
            if keyword in question_lower:
                keywords_to_check[keyword].append((qid, is_correct))
    
    for keyword, items in keywords_to_check.items():
        if items:
            correct_count = sum(1 for _, correct in items if correct)
            total_count = len(items)
            acc = correct_count / total_count * 100 if total_count > 0 else 0
            print(f"  '{keyword}': {correct_count}/{total_count} correct ({acc:.1f}%)")
    
    # 分析答案长度和复杂度
    print("\n3. Answer complexity analysis:")
    short_answers = []  # < 5 words
    medium_answers = []  # 5-15 words
    long_answers = []  # > 15 words
    
    for item in assistant_items:
        qid = item['question_id']
        answer = str(item['answer'])
        word_count = len(answer.split())
        is_correct = baseline_results.get(qid, None)
        
        if word_count < 5:
            short_answers.append((qid, is_correct))
        elif word_count < 15:
            medium_answers.append((qid, is_correct))
        else:
            long_answers.append((qid, is_correct))
    
    for category, items in [('Short (<5 words)', short_answers), 
                            ('Medium (5-15 words)', medium_answers),
                            ('Long (>15 words)', long_answers)]:
        if items:
            correct_count = sum(1 for _, correct in items if correct)
            total_count = len(items)
            acc = correct_count / total_count * 100 if total_count > 0 else 0
            print(f"  {category}: {correct_count}/{total_count} correct ({acc:.1f}%)")
    
    # 显示一些失败案例的详细信息
    print("\n" + "=" * 80)
    print("SAMPLE FAILURE CASES")
    print("=" * 80)
    
    failure_samples = []
    for item in assistant_items[:10]:  # Phase1 测试的前10个
        qid = item['question_id']
        is_correct = baseline_results.get(qid, None)
        if is_correct is False:
            failure_samples.append(item)
    
    for i, item in enumerate(failure_samples[:5], 1):
        print(f"\n{i}. Question ID: {item['question_id']}")
        print(f"   Q: {item['question']}")
        print(f"   Expected A: {item['answer']}")
        
        # 找到答案所在的 turn
        for session in item['haystack_sessions']:
            for turn in session:
                if turn.get('has_answer'):
                    print(f"   Answer in: {turn['role']} turn")
                    print(f"   Answer content: {turn['content'][:150]}...")
                    break
    
    return assistant_items, baseline_results


def test_retrieval_comparison():
    """测试 benchmark_mode 对检索的影响"""
    print("\n" + "=" * 80)
    print("RETRIEVAL COMPARISON TEST")
    print("=" * 80)
    
    baseline, phase1, data = load_results()
    
    # 选择一个失败的 assistant 问题
    assistant_items = [item for item in data if item['question_type'] == 'single-session-assistant']
    baseline_results = {}
    for r in baseline['detailed_results']:
        if r['question_type'] == 'single-session-assistant':
            baseline_results[r['question_id']] = r['is_correct']
    
    # 找一个 baseline 正确但可能在 phase1 失败的问题
    test_item = None
    for item in assistant_items[:10]:
        qid = item['question_id']
        if baseline_results.get(qid) == True:  # Baseline 正确
            test_item = item
            break
    
    if not test_item:
        print("No suitable test case found")
        return
    
    print(f"\nTest case: {test_item['question_id']}")
    print(f"Question: {test_item['question']}")
    print(f"Expected answer: {test_item['answer']}")
    
    # 初始化
    embedder = BailianEmbedding(
        api_key=os.getenv('AURORA_BAILIAN_API_KEY'),
        model='text-embedding-v4',
        dimension=1024
    )
    
    config_baseline = MemoryConfig(dim=1024, max_plots=5000, benchmark_mode=False)
    config_phase1 = MemoryConfig(dim=1024, max_plots=5000, benchmark_mode=True)
    
    # 测试 baseline 模式 (benchmark_mode=False)
    print("\n--- Baseline mode (benchmark_mode=False) ---")
    memory_baseline = AuroraMemory(cfg=config_baseline, seed=42, embedder=embedder, benchmark_mode=False)
    
    for session in test_item['haystack_sessions']:
        for turn in session:
            text = f"{turn['role'].capitalize()}: {turn['content']}"
            memory_baseline.ingest(text)
    
    trace_baseline = memory_baseline.query(test_item['question'], k=10)
    print(f"Stored plots: {len(memory_baseline.plots)}")
    print(f"Retrieved: {len(trace_baseline.ranked)} results")
    print("Top 3 retrieved:")
    for i, (plot_id, score, kind) in enumerate(trace_baseline.ranked[:3], 1):
        plot = memory_baseline.plots.get(plot_id)
        if plot:
            print(f"  {i}. [{kind}] {plot.text[:100]}...")
    
    # 测试 phase1 模式 (benchmark_mode=True)
    print("\n--- Phase1 mode (benchmark_mode=True) ---")
    memory_phase1 = AuroraMemory(cfg=config_phase1, seed=42, embedder=embedder, benchmark_mode=True)
    
    for session in test_item['haystack_sessions']:
        for turn in session:
            text = f"{turn['role'].capitalize()}: {turn['content']}"
            memory_phase1.ingest(text)
    
    trace_phase1 = memory_phase1.query(test_item['question'], k=10)
    print(f"Stored plots: {len(memory_phase1.plots)}")
    print(f"Retrieved: {len(trace_phase1.ranked)} results")
    print("Top 3 retrieved:")
    for i, (plot_id, score, kind) in enumerate(trace_phase1.ranked[:3], 1):
        plot = memory_phase1.plots.get(plot_id)
        if plot:
            print(f"  {i}. [{kind}] {plot.text[:100]}...")


if __name__ == '__main__':
    import os
    analyze_assistant_questions()
    # test_retrieval_comparison()  # 需要 API key，暂时注释
