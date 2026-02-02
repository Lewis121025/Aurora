#!/usr/bin/env python3
"""
深度分析 multi-session 失败原因
================================

分析 multi-session 类型只有 25.6% 准确率的根本原因：
1. 信息是否存储？
2. 检索是否找到？
3. 跨会话聚合是否失败？
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.embeddings.bailian import BailianEmbedding
import os
from dotenv import load_dotenv

load_dotenv()


def load_baseline_results() -> Dict[str, Any]:
    """加载 baseline 结果"""
    with open('longmemeval_baseline.json', 'r') as f:
        return json.load(f)


def load_original_data() -> List[Dict[str, Any]]:
    """加载原始数据"""
    data_path = Path('data/longmemeval/longmemeval_oracle_new.json')
    if not data_path.exists():
        data_path = Path('data/longmemeval/longmemeval_oracle.json')
    
    if not data_path.exists():
        print(f"ERROR: Cannot find data file at {data_path}")
        return []
    
    with open(data_path, 'r') as f:
        return json.load(f)


def analyze_storage_coverage(item: Dict, memory: AuroraMemory) -> Dict[str, Any]:
    """分析信息存储覆盖率"""
    total_turns = sum(len(session) for session in item['haystack_sessions'])
    stored_plots = len(memory.plots)
    
    # 检查每个 session 的存储情况
    session_storage = []
    plot_idx = 0
    for session_idx, session in enumerate(item['haystack_sessions']):
        session_turns = len(session)
        session_plots = []
        
        # 尝试匹配 session 的 plots（通过时间戳或顺序）
        for turn in session:
            # 简单匹配：按顺序查找
            if plot_idx < stored_plots:
                plot_id = list(memory.plots.keys())[plot_idx]
                plot = memory.plots[plot_id]
                session_plots.append({
                    'turn_text': f"{turn['role']}: {turn['content'][:50]}",
                    'stored': True,
                    'plot_id': plot_id[:8],
                    'plot_text': plot.text[:50]
                })
                plot_idx += 1
            else:
                session_plots.append({
                    'turn_text': f"{turn['role']}: {turn['content'][:50]}",
                    'stored': False
                })
        
        session_storage.append({
            'session_idx': session_idx,
            'total_turns': session_turns,
            'stored_turns': len([p for p in session_plots if p.get('stored')]),
            'plots': session_plots
        })
    
    return {
        'total_turns': total_turns,
        'stored_plots': stored_plots,
        'coverage': stored_plots / total_turns if total_turns > 0 else 0,
        'session_storage': session_storage
    }


def analyze_retrieval(item: Dict, memory: AuroraMemory, question: str, k: int = 10) -> Dict[str, Any]:
    """分析检索结果"""
    trace = memory.query(question, k=k)
    
    # 检查答案是否在检索结果中
    expected_answer = str(item['answer']).lower()
    answer_found_in_retrieval = False
    answer_found_in_context = False
    
    retrieved_texts = []
    for plot_id, score, kind in trace.ranked[:k]:
        plot = memory.plots.get(plot_id)
        if plot:
            text = plot.text.lower()
            retrieved_texts.append({
                'plot_id': plot_id[:8],
                'text': plot.text[:100],
                'score': float(score),
                'contains_answer': expected_answer in text
            })
            if expected_answer in text:
                answer_found_in_retrieval = True
    
    context = "\n".join([r['text'] for r in retrieved_texts])
    if expected_answer in context.lower():
        answer_found_in_context = True
    
    return {
        'retrieval_count': len(trace.ranked),
        'answer_found_in_retrieval': answer_found_in_retrieval,
        'answer_found_in_context': answer_found_in_context,
        'retrieved_texts': retrieved_texts[:5],  # Top 5
        'query_type': str(trace.query_type) if hasattr(trace, 'query_type') else 'unknown'
    }


def analyze_cross_session_aggregation(item: Dict, memory: AuroraMemory) -> Dict[str, Any]:
    """分析跨会话聚合能力"""
    # 检查答案涉及的会话
    answer_session_ids = item.get('answer_session_ids', [])
    
    if not answer_session_ids:
        return {'error': 'No answer_session_ids in item'}
    
    # 检查这些会话的信息是否都被存储
    sessions_info = []
    for session_idx in answer_session_ids:
        if session_idx < len(item['haystack_sessions']):
            session = item['haystack_sessions'][session_idx]
            session_text = " ".join([f"{t['role']}: {t['content']}" for t in session]).lower()
            
            # 检查这个 session 的内容是否在存储的 plots 中
            found_in_plots = False
            for plot in memory.plots.values():
                if session_text[:100] in plot.text.lower() or plot.text.lower() in session_text[:200]:
                    found_in_plots = True
                    break
            
            sessions_info.append({
                'session_idx': session_idx,
                'turns': len(session),
                'found_in_plots': found_in_plots,
                'sample_text': session[0]['content'][:50] if session else ''
            })
    
    return {
        'answer_sessions': answer_session_ids,
        'sessions_info': sessions_info,
        'all_sessions_stored': all(s['found_in_plots'] for s in sessions_info)
    }


def analyze_single_failure(item: Dict, baseline_result: Dict, embedder: BailianEmbedding) -> Dict[str, Any]:
    """分析单个失败案例"""
    question_id = item['question_id']
    question = item['question']
    expected_answer = item['answer']
    question_type = item['question_type']
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {question_id}")
    print(f"Question: {question}")
    print(f"Expected: {expected_answer}")
    print(f"Type: {question_type}")
    
    # 创建 memory（注意：这里没有设置 benchmark_mode=True，模拟实际运行）
    config = MemoryConfig(dim=1024, max_plots=5000)
    memory = AuroraMemory(cfg=config, seed=42, embedder=embedder)
    
    # Ingest 所有会话
    print(f"\nIngesting {len(item['haystack_sessions'])} sessions...")
    for session_idx, session in enumerate(item['haystack_sessions']):
        for turn in session:
            text = f"{turn['role'].capitalize()}: {turn['content']}"
            memory.ingest(text)
    
    print(f"Stored {len(memory.plots)} plots")
    
    # 1. 分析存储覆盖率
    storage_analysis = analyze_storage_coverage(item, memory)
    print(f"\n[Storage] Coverage: {storage_analysis['coverage']:.1%} ({storage_analysis['stored_plots']}/{storage_analysis['total_turns']})")
    
    # 2. 分析检索
    retrieval_analysis = analyze_retrieval(item, memory, question, k=10)
    print(f"\n[Retrieval] Found in top results: {retrieval_analysis['answer_found_in_retrieval']}")
    print(f"[Retrieval] Found in context: {retrieval_analysis['answer_found_in_context']}")
    print(f"[Retrieval] Retrieved {retrieval_analysis['retrieval_count']} results")
    
    # 3. 分析跨会话聚合
    aggregation_analysis = analyze_cross_session_aggregation(item, memory)
    if 'error' not in aggregation_analysis:
        print(f"\n[Aggregation] Answer spans {len(aggregation_analysis['answer_sessions'])} sessions")
        print(f"[Aggregation] All sessions stored: {aggregation_analysis['all_sessions_stored']}")
    
    # 4. 诊断失败原因
    failure_reason = "unknown"
    if storage_analysis['coverage'] < 0.8:
        failure_reason = "low_storage_coverage"
    elif not retrieval_analysis['answer_found_in_context']:
        failure_reason = "retrieval_failure"
    elif not aggregation_analysis.get('all_sessions_stored', False):
        failure_reason = "cross_session_missing"
    else:
        failure_reason = "answer_generation_failure"
    
    print(f"\n[Diagnosis] Failure reason: {failure_reason}")
    
    return {
        'question_id': question_id,
        'question': question,
        'expected_answer': expected_answer,
        'failure_reason': failure_reason,
        'storage_analysis': storage_analysis,
        'retrieval_analysis': retrieval_analysis,
        'aggregation_analysis': aggregation_analysis
    }


def main():
    print("="*80)
    print("Multi-Session Failure Analysis")
    print("="*80)
    
    # 加载数据
    print("\n1. Loading data...")
    baseline = load_baseline_results()
    data = load_original_data()
    
    if not data:
        print("ERROR: Cannot load original data")
        return
    
    # 找出失败的 multi-session 问题
    failed_multi_session = []
    for result in baseline['detailed_results']:
        if result['question_type'] == 'multi-session' and not result['is_correct']:
            question_id = result['question_id']
            # 找到对应的原始数据
            for item in data:
                if item['question_id'] == question_id:
                    failed_multi_session.append((item, result))
                    break
    
    print(f"\nFound {len(failed_multi_session)} failed multi-session questions")
    
    # 统计错误类型
    error_types = defaultdict(int)
    for _, result in failed_multi_session:
        error = result.get('error', 'none')
        if error:
            error_types[error] += 1
        else:
            error_types['no_error_but_wrong'] += 1
    
    print(f"\nError distribution:")
    for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")
    
    # 初始化 embedder
    print("\n2. Initializing embedder...")
    embedder = BailianEmbedding(
        api_key=os.getenv('AURORA_BAILIAN_API_KEY'),
        model='text-embedding-v4',
        dimension=1024
    )
    
    # 分析前 5 个典型失败案例
    print("\n3. Analyzing typical failure cases...")
    analysis_results = []
    
    # 优先分析没有错误的失败（逻辑错误）
    logic_failures = [(item, res) for item, res in failed_multi_session if not res.get('error')]
    error_failures = [(item, res) for item, res in failed_multi_session if res.get('error')]
    
    # 先分析逻辑失败
    for item, result in logic_failures[:3]:
        try:
            analysis = analyze_single_failure(item, result, embedder)
            analysis_results.append(analysis)
        except Exception as e:
            print(f"ERROR analyzing {item['question_id']}: {e}")
            import traceback
            traceback.print_exc()
    
    # 再分析错误失败
    for item, result in error_failures[:2]:
        try:
            analysis = analyze_single_failure(item, result, embedder)
            analysis_results.append(analysis)
        except Exception as e:
            print(f"ERROR analyzing {item['question_id']}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总分析结果
    print("\n" + "="*80)
    print("Summary Analysis")
    print("="*80)
    
    failure_reasons = defaultdict(int)
    for analysis in analysis_results:
        failure_reasons[analysis['failure_reason']] += 1
    
    print(f"\nFailure reason distribution (from {len(analysis_results)} analyzed):")
    for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    # 保存详细分析结果
    output_path = Path('multisession_failure_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({
            'summary': {
                'total_failed': len(failed_multi_session),
                'analyzed': len(analysis_results),
                'failure_reasons': dict(failure_reasons),
                'error_types': dict(error_types)
            },
            'detailed_analysis': analysis_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed analysis saved to {output_path}")


if __name__ == '__main__':
    main()
