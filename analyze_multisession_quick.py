#!/usr/bin/env python3
"""
快速分析 multi-session 失败原因（无需 API 调用）
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_data():
    """加载所有必要的数据"""
    # Baseline 结果
    with open('longmemeval_baseline.json', 'r') as f:
        baseline = json.load(f)
    
    # 原始数据（只读取必要的字段）
    data_path = Path('data/longmemeval/longmemeval_oracle_new.json')
    if not data_path.exists():
        data_path = Path('data/longmemeval/longmemeval_oracle.json')
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return baseline, data


def analyze_error_patterns(baseline: Dict, data: List[Dict]) -> Dict[str, Any]:
    """分析错误模式"""
    # 找出所有 multi-session 问题
    multi_session_results = [
        r for r in baseline['detailed_results'] 
        if r['question_type'] == 'multi-session'
    ]
    
    # 创建 question_id -> result 映射
    result_map = {r['question_id']: r for r in multi_session_results}
    
    # 创建 question_id -> item 映射
    item_map = {item['question_id']: item for item in data}
    
    # 分析失败案例
    failed = []
    successful = []
    
    for question_id, result in result_map.items():
        if question_id not in item_map:
            continue
        
        item = item_map[question_id]
        analysis = {
            'question_id': question_id,
            'question': item['question'],
            'expected_answer': str(item['answer']),
            'is_correct': result['is_correct'],
            'error': result.get('error'),
            'num_sessions': len(item.get('haystack_sessions', [])),
            'answer_session_ids': item.get('answer_session_ids', []),
            'question_length': len(item['question']),
            'answer_length': len(str(item['answer'])),
        }
        
        if result['is_correct']:
            successful.append(analysis)
        else:
            failed.append(analysis)
    
    return {
        'total': len(multi_session_results),
        'failed': len(failed),
        'successful': len(successful),
        'failed_cases': failed,
        'successful_cases': successful
    }


def categorize_failures(failed_cases: List[Dict]) -> Dict[str, Any]:
    """对失败案例进行分类"""
    categories = {
        'has_error': [],
        'no_error_logic_failure': [],
        'numeric_answer': [],
        'text_answer': [],
        'single_session_answer': [],
        'multi_session_answer': [],
    }
    
    for case in failed_cases:
        # 错误类型分类
        if case['error']:
            categories['has_error'].append(case)
        else:
            categories['no_error_logic_failure'].append(case)
        
        # 答案类型分类
        try:
            float(case['expected_answer'])
            categories['numeric_answer'].append(case)
        except ValueError:
            categories['text_answer'].append(case)
        
        # 会话跨度分类
        answer_sessions = case.get('answer_session_ids', [])
        if len(answer_sessions) == 1:
            categories['single_session_answer'].append(case)
        elif len(answer_sessions) > 1:
            categories['multi_session_answer'].append(case)
    
    return categories


def print_analysis_report(analysis: Dict, categories: Dict):
    """打印分析报告"""
    print("="*80)
    print("Multi-Session Failure Analysis Report")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total multi-session questions: {analysis['total']}")
    print(f"  Failed: {analysis['failed']} ({analysis['failed']/analysis['total']*100:.1f}%)")
    print(f"  Successful: {analysis['successful']} ({analysis['successful']/analysis['total']*100:.1f}%)")
    
    print(f"\nFailure Categories:")
    print(f"  Has error: {len(categories['has_error'])}")
    print(f"  Logic failure (no error): {len(categories['no_error_logic_failure'])}")
    
    print(f"\nAnswer Type Distribution:")
    print(f"  Numeric answers: {len(categories['numeric_answer'])}")
    print(f"  Text answers: {len(categories['text_answer'])}")
    
    print(f"\nSession Span Distribution:")
    print(f"  Single session answers: {len(categories['single_session_answer'])}")
    print(f"  Multi-session answers: {len(categories['multi_session_answer'])}")
    
    # 分析错误类型
    if categories['has_error']:
        error_types = defaultdict(int)
        for case in categories['has_error']:
            error_types[case['error']] += 1
        
        print(f"\nError Types:")
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
    
    # 显示典型失败案例
    print(f"\n{'='*80}")
    print("Typical Failure Cases (Logic Failures)")
    print("="*80)
    
    logic_failures = categories['no_error_logic_failure'][:5]
    for i, case in enumerate(logic_failures, 1):
        print(f"\n[{i}] {case['question_id']}")
        print(f"    Q: {case['question']}")
        print(f"    A: {case['expected_answer']}")
        print(f"    Sessions: {case['num_sessions']} total, answer spans {len(case.get('answer_session_ids', []))} sessions")
    
    # 显示错误案例
    if categories['has_error']:
        print(f"\n{'='*80}")
        print("Error Cases")
        print("="*80)
        
        error_cases = categories['has_error'][:5]
        for i, case in enumerate(error_cases, 1):
            print(f"\n[{i}] {case['question_id']}")
            print(f"    Q: {case['question']}")
            print(f"    A: {case['expected_answer']}")
            print(f"    Error: {case['error']}")


def compare_successful_vs_failed(successful: List[Dict], failed: List[Dict]) -> Dict[str, Any]:
    """对比成功和失败案例的特征"""
    def avg(attr, cases):
        values = [case.get(attr, 0) for case in cases]
        return sum(values) / len(values) if values else 0
    
    comparison = {
        'num_sessions': {
            'successful_avg': avg('num_sessions', successful),
            'failed_avg': avg('num_sessions', failed),
        },
        'question_length': {
            'successful_avg': avg('question_length', successful),
            'failed_avg': avg('question_length', failed),
        },
        'answer_length': {
            'successful_avg': avg('answer_length', successful),
            'failed_avg': avg('answer_length', failed),
        },
    }
    
    return comparison


def main():
    print("Loading data...")
    baseline, data = load_data()
    
    print("Analyzing failure patterns...")
    analysis = analyze_error_patterns(baseline, data)
    
    print("Categorizing failures...")
    categories = categorize_failures(analysis['failed_cases'])
    
    print("Comparing successful vs failed...")
    comparison = compare_successful_vs_failed(
        analysis['successful_cases'],
        analysis['failed_cases']
    )
    
    print_analysis_report(analysis, categories)
    
    print(f"\n{'='*80}")
    print("Success vs Failure Comparison")
    print("="*80)
    print(f"Average sessions:")
    print(f"  Successful: {comparison['num_sessions']['successful_avg']:.1f}")
    print(f"  Failed: {comparison['num_sessions']['failed_avg']:.1f}")
    print(f"\nAverage question length:")
    print(f"  Successful: {comparison['question_length']['successful_avg']:.1f}")
    print(f"  Failed: {comparison['question_length']['failed_avg']:.1f}")
    print(f"\nAverage answer length:")
    print(f"  Successful: {comparison['answer_length']['successful_avg']:.1f}")
    print(f"  Failed: {comparison['answer_length']['failed_avg']:.1f}")
    
    # 保存分析结果
    output = {
        'analysis': analysis,
        'categories': {k: len(v) for k, v in categories.items()},
        'comparison': comparison,
        'sample_failures': categories['no_error_logic_failure'][:10],
        'sample_errors': categories['has_error'][:10]
    }
    
    output_path = Path('multisession_quick_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis saved to {output_path}")


if __name__ == '__main__':
    main()
