#!/usr/bin/env python3
"""
详细分析 single-session-assistant 准确率下降的根本原因

对比 baseline (73.2%) vs phase1 (60%) 的失败案例
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.insert(0, '.')


def load_results():
    """加载结果文件"""
    with open('longmemeval_baseline.json', 'r') as f:
        baseline = json.load(f)
    
    with open('longmemeval_phase1_results.json', 'r') as f:
        phase1 = json.load(f)
    
    with open('data/longmemeval/longmemeval_oracle_new.json', 'r') as f:
        data = json.load(f)
    
    return baseline, phase1, data


def get_phase1_detailed_results():
    """尝试从 phase1 测试中获取详细结果"""
    # Phase1 只测试了前10个，我们需要找出是哪些
    baseline, phase1, data = load_results()
    
    assistant_items = [item for item in data if item['question_type'] == 'single-session-assistant']
    
    # Phase1 测试的是前10个
    phase1_tested = assistant_items[:10]
    phase1_qids = {item['question_id'] for item in phase1_tested}
    
    # 从 baseline 中获取这些问题的结果
    baseline_results = {}
    for r in baseline['detailed_results']:
        if r['question_type'] == 'single-session-assistant' and r['question_id'] in phase1_qids:
            baseline_results[r['question_id']] = r['is_correct']
    
    # Phase1 准确率是 60%，意味着 10 个中 6 个正确，4 个错误
    # 我们需要找出哪些在 baseline 中正确，但在 phase1 中可能失败
    baseline_correct_in_phase1 = [qid for qid, correct in baseline_results.items() if correct]
    baseline_incorrect_in_phase1 = [qid for qid, correct in baseline_results.items() if not correct]
    
    print("=" * 80)
    print("PHASE1 TEST SET ANALYSIS")
    print("=" * 80)
    print(f"\nPhase1 tested: {len(phase1_tested)} questions")
    print(f"Baseline correct in phase1 set: {len(baseline_correct_in_phase1)}")
    print(f"Baseline incorrect in phase1 set: {len(baseline_incorrect_in_phase1)}")
    print(f"Phase1 accuracy: 60% (6/10 correct)")
    
    # 如果 baseline 在 phase1 测试集中有 7+ 个正确，但 phase1 只有 6 个正确
    # 说明有至少 1 个从正确变成了错误
    if len(baseline_correct_in_phase1) >= 7:
        print(f"\n⚠️  WARNING: Baseline had {len(baseline_correct_in_phase1)} correct in phase1 set,")
        print(f"   but phase1 only got 6 correct. At least 1 regression!")
    
    return phase1_tested, baseline_results


def analyze_failure_patterns():
    """分析失败模式"""
    baseline, phase1, data = load_results()
    phase1_tested, baseline_results = get_phase1_detailed_results()
    
    print("\n" + "=" * 80)
    print("FAILURE PATTERN ANALYSIS")
    print("=" * 80)
    
    # 分析 phase1 测试集中的问题
    for item in phase1_tested:
        qid = item['question_id']
        baseline_correct = baseline_results.get(qid, None)
        
        # 找出答案位置
        answer_role = None
        answer_content = None
        for session in item['haystack_sessions']:
            for turn in session:
                if turn.get('has_answer'):
                    answer_role = turn['role']
                    answer_content = turn['content']
                    break
            if answer_role:
                break
        
        # 分析问题特征
        question_lower = item['question'].lower()
        has_remind = 'remind' in question_lower
        has_told = 'told' in question_lower or 'said' in question_lower
        has_suggest = 'suggest' in question_lower or 'recommend' in question_lower
        
        # 答案长度
        answer_words = len(str(item['answer']).split())
        
        print(f"\nQID: {qid}")
        print(f"  Question: {item['question'][:80]}...")
        print(f"  Expected: {item['answer'][:60]}...")
        print(f"  Answer in: {answer_role} turn")
        print(f"  Answer length: {answer_words} words")
        print(f"  Baseline correct: {baseline_correct}")
        print(f"  Keywords: remind={has_remind}, told/said={has_told}, suggest={has_suggest}")
        if answer_content:
            print(f"  Answer content preview: {answer_content[:100]}...")


def compare_retrieval_strategies():
    """对比 baseline 和 phase1 的检索策略差异"""
    print("\n" + "=" * 80)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("=" * 80)
    
    baseline, phase1, data = load_results()
    
    # 检查代码差异
    print("\n1. Benchmark Mode Difference:")
    print("   Baseline: benchmark_mode=False")
    print("   Phase1:   benchmark_mode=True")
    print("\n   Impact: benchmark_mode=True forces storage of all interactions,")
    print("           which may increase retrieval noise for assistant questions")
    
    print("\n2. Retrieval Parameters:")
    print("   Baseline: k=10 (from run_longmemeval_baseline.py)")
    print("   Phase1:   k=15 (from run_longmemeval_phase1.py)")
    print("\n   Impact: Larger k may retrieve more irrelevant results")
    
    print("\n3. Abstention Mechanism:")
    print("   Baseline: No abstention")
    print("   Phase1:   Has abstention check")
    print("\n   Impact: May incorrectly abstain on valid assistant questions")
    
    print("\n4. Time Filter:")
    print("   Baseline: No time filtering")
    print("   Phase1:   TimeRangeExtractor for pre-filtering")
    print("\n   Impact: May filter out relevant assistant responses")


def analyze_answer_extraction():
    """分析答案提取逻辑"""
    print("\n" + "=" * 80)
    print("ANSWER EXTRACTION ANALYSIS")
    print("=" * 80)
    
    baseline, phase1, data = load_results()
    
    # 检查 LLM prompt 差异
    print("\n1. LLM Prompt Comparison:")
    print("\n   Baseline prompt:")
    print("   'Based on the conversation history below, answer the question concisely.'")
    print("   'Focus on extracting the specific information requested.'")
    
    print("\n   Phase1 prompt:")
    print("   'Answer the question based on the conversation history.'")
    print("   'Answer (be brief and specific):'")
    
    print("\n   Difference: Baseline emphasizes 'extracting specific information'")
    print("              Phase1 is more generic")
    
    print("\n2. Context Length:")
    print("   Baseline: context[:3500]")
    print("   Phase1:   context[:3000]")
    print("\n   Impact: Phase1 uses shorter context, may miss relevant info")
    
    print("\n3. Answer Evaluation:")
    print("   Both use similar evaluation logic")
    print("   But phase1 has abstention handling which may affect scoring")


def main():
    """主函数"""
    get_phase1_detailed_results()
    analyze_failure_patterns()
    compare_retrieval_strategies()
    analyze_answer_extraction()
    
    print("\n" + "=" * 80)
    print("ROOT CAUSE HYPOTHESIS")
    print("=" * 80)
    print("""
Based on the analysis, the likely root causes are:

1. **Benchmark Mode Noise** (Most Likely)
   - benchmark_mode=True stores ALL interactions, including user turns
   - For assistant questions asking "what did you say/recommend",
     the system may retrieve user turns instead of assistant turns
   - This increases retrieval noise

2. **Context Truncation**
   - Phase1 uses context[:3000] vs baseline context[:3500]
   - May truncate relevant assistant responses

3. **Prompt Weakening**
   - Phase1 prompt is less specific about "extracting specific information"
   - May generate less precise answers

4. **Abstention False Positives**
   - Abstention mechanism may incorrectly flag valid assistant questions
   - Leading to "I don't know" responses

RECOMMENDATIONS:
1. For assistant questions, prioritize assistant turns in retrieval
2. Increase context length back to 3500
3. Strengthen prompt for assistant questions
4. Review abstention logic for assistant question type
""")


if __name__ == '__main__':
    main()
