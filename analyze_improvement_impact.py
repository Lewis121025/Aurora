#!/usr/bin/env python3
"""
分析从 68% 到 90% 的改进方向
============================

计算每个类型的提升对总分的影响，找出投入产出比最高的优化方向
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TypeInfo:
    name: str
    current: float  # 当前准确率 (%)
    target: float   # 目标准确率 (%)
    count: int      # 题目数量
    difficulty: str  # 难度评估: easy/medium/hard
    estimated_effort: int  # 预估工作量（天）


def calculate_impact(types_info: Dict[str, TypeInfo]) -> List[Tuple[str, Dict]]:
    """计算每个类型的提升对总分的影响"""
    total = sum(t.count for t in types_info.values())
    
    results = []
    
    for name, info in types_info.items():
        # 当前正确数
        current_correct = info.current / 100 * info.count
        
        # 目标正确数
        target_correct = info.target / 100 * info.count
        
        # 提升的正确数
        delta_correct = target_correct - current_correct
        
        # 对总分的提升（百分点）
        delta_overall = delta_correct / total * 100
        
        # 当前总分贡献
        current_contribution = current_correct / total * 100
        
        # 目标总分贡献
        target_contribution = target_correct / total * 100
        
        # 投入产出比（每单位工作量提升的百分点）
        roi = delta_overall / info.estimated_effort if info.estimated_effort > 0 else 0
        
        results.append((
            name,
            {
                'current_correct': current_correct,
                'target_correct': target_correct,
                'delta_correct': delta_correct,
                'delta_overall_pct': delta_overall,
                'current_contribution': current_contribution,
                'target_contribution': target_contribution,
                'roi': roi,
                'difficulty': info.difficulty,
                'estimated_effort': info.estimated_effort,
            }
        ))
    
    return results


def analyze_failure_patterns() -> Dict[str, Dict]:
    """分析失败案例的共同特征"""
    patterns = {
        'multi-session': {
            'common_failures': [
                '跨会话信息聚合失败',
                'VOI门控丢弃关键信息',
                '检索范围不足（k值太小）',
                '缺乏显式聚合机制',
            ],
            'fix_complexity': 'high',
            'requires': [
                '修复代码错误（类型转换）',
                '启用benchmark_mode',
                '增加检索k值',
                '改进聚合prompt',
                '实现会话标记和Story聚合',
            ],
        },
        'temporal-reasoning': {
            'common_failures': [
                '时序关系理解不足',
                '时间过滤不准确',
                '时间顺序推理失败',
            ],
            'fix_complexity': 'medium',
            'requires': [
                '改进时间过滤逻辑',
                '增强时序推理prompt',
                '优化时间嵌入',
            ],
        },
        'single-session-preference': {
            'common_failures': [
                '偏好信息未正确存储',
                '偏好提取失败',
            ],
            'fix_complexity': 'medium',
            'requires': [
                '改进偏好识别',
                '增强偏好存储',
            ],
        },
        'knowledge-update': {
            'common_failures': [
                '更新检测失败',
                '旧值未正确标记为superseded',
            ],
            'fix_complexity': 'medium',
            'requires': [
                '改进更新检测逻辑',
                '优化相似度阈值',
            ],
        },
        'single-session-user': {
            'common_failures': [
                '检索噪声',
                '上下文截断',
            ],
            'fix_complexity': 'low',
            'requires': [
                '优化检索',
                '增加上下文长度',
            ],
        },
        'single-session-assistant': {
            'common_failures': [
                'benchmark_mode导致检索噪声',
                '未优先检索assistant turns',
                'prompt弱化',
            ],
            'fix_complexity': 'low',
            'requires': [
                '为assistant问题优先检索assistant turns',
                '恢复上下文长度',
                '增强prompt',
            ],
        },
    }
    return patterns


def main():
    # 根据用户提供的数据和已有分析，设置类型信息
    types_info = {
        'multi-session': TypeInfo(
            name='multi-session',
            current=53.0,
            target=80.0,
            count=133,
            difficulty='hard',
            estimated_effort=5,  # 需要多步修复
        ),
        'temporal-reasoning': TypeInfo(
            name='temporal-reasoning',
            current=60.0,
            target=80.0,
            count=133,
            difficulty='medium',
            estimated_effort=3,
        ),
        'single-session-preference': TypeInfo(
            name='single-session-preference',
            current=53.0,
            target=80.0,
            count=30,
            difficulty='medium',
            estimated_effort=2,
        ),
        'knowledge-update': TypeInfo(
            name='knowledge-update',
            current=80.0,
            target=90.0,
            count=78,
            difficulty='medium',
            estimated_effort=2,
        ),
        'single-session-user': TypeInfo(
            name='single-session-user',
            current=87.0,
            target=95.0,  # 可以设更高目标
            count=70,
            difficulty='easy',
            estimated_effort=1,
        ),
        'single-session-assistant': TypeInfo(
            name='single-session-assistant',
            current=73.0,
            target=85.0,
            count=56,
            difficulty='easy',
            estimated_effort=1,
        ),
    }
    
    # 计算影响
    results = calculate_impact(types_info)
    
    # 按ROI排序
    results_by_roi = sorted(results, key=lambda x: x[1]['roi'], reverse=True)
    
    # 按delta_overall排序
    results_by_impact = sorted(results, key=lambda x: x[1]['delta_overall_pct'], reverse=True)
    
    # 获取失败模式
    failure_patterns = analyze_failure_patterns()
    
    print("=" * 80)
    print("从 68% 到 90% 的改进方向分析")
    print("=" * 80)
    
    # 计算当前总分
    total = sum(t.count for t in types_info.values())
    current_total_correct = sum(r[1]['current_correct'] for r in results)
    current_overall = current_total_correct / total * 100
    
    print(f"\n当前总分: {current_overall:.1f}%")
    print(f"目标总分: 90.0%")
    print(f"需要提升: {90.0 - current_overall:.1f} 个百分点")
    print(f"总题数: {total}")
    
    print("\n" + "=" * 80)
    print("1. 按投入产出比（ROI）排序 - 优先实施")
    print("=" * 80)
    print(f"{'类型':<30} {'当前':<8} {'目标':<8} {'提升(分)':<12} {'工作量(天)':<12} {'ROI':<10}")
    print("-" * 80)
    
    for name, info in results_by_roi:
        print(f"{name:<30} {info['current_contribution']:>6.1f}% {info['target_contribution']:>6.1f}% "
              f"{info['delta_overall_pct']:>10.2f}% {info['estimated_effort']:>10} "
              f"{info['roi']:>8.2f}")
    
    print("\n" + "=" * 80)
    print("2. 按总体影响排序 - 最大收益")
    print("=" * 80)
    print(f"{'类型':<30} {'当前':<8} {'目标':<8} {'提升(分)':<12} {'题目数':<10} {'难度':<10}")
    print("-" * 80)
    
    for name, info in results_by_impact:
        print(f"{name:<30} {info['current_contribution']:>6.1f}% {info['target_contribution']:>6.1f}% "
              f"{info['delta_overall_pct']:>10.2f}% {types_info[name].count:>8} "
              f"{info['difficulty']:<10}")
    
    print("\n" + "=" * 80)
    print("3. 失败案例共同特征分析")
    print("=" * 80)
    
    for name, pattern in failure_patterns.items():
        print(f"\n【{name}】")
        print(f"  常见失败原因:")
        for failure in pattern['common_failures']:
            print(f"    - {failure}")
        print(f"  修复复杂度: {pattern['fix_complexity']}")
        print(f"  需要改进:")
        for req in pattern['requires']:
            print(f"    - {req}")
    
    print("\n" + "=" * 80)
    print("4. 优先级排序的优化方向")
    print("=" * 80)
    
    # 综合排序：考虑ROI、影响、难度
    priority_list = []
    for name, info in results:
        # 综合得分 = ROI * 0.4 + delta_overall * 0.4 + (1/difficulty_score) * 0.2
        difficulty_score = {'easy': 1, 'medium': 2, 'hard': 3}[info['difficulty']]
        composite_score = (
            info['roi'] * 0.4 +
            info['delta_overall_pct'] * 0.4 +
            (1 / difficulty_score) * 20 * 0.2
        )
        priority_list.append((name, composite_score, info))
    
    priority_list.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'优先级':<6} {'类型':<30} {'综合得分':<12} {'预期提升':<12} {'工作量':<10} {'关键改进'}")
    print("-" * 100)
    
    for i, (name, score, info) in enumerate(priority_list, 1):
        pattern = failure_patterns.get(name, {})
        key_fixes = pattern.get('requires', [])[:2]  # 取前2个关键改进
        key_fix_str = "; ".join(key_fixes) if key_fixes else "N/A"
        
        print(f"{i:<6} {name:<30} {score:>10.2f} {info['delta_overall_pct']:>10.2f}% "
              f"{info['estimated_effort']:>8}天 {key_fix_str[:40]}")
    
    print("\n" + "=" * 80)
    print("5. 分阶段实施建议")
    print("=" * 80)
    
    # 阶段1：快速修复（1-2天）
    quick_wins = [r for r in priority_list if r[2]['estimated_effort'] <= 1]
    print("\n【阶段1：快速修复（1-2天）】")
    print("预期提升:", sum(r[2]['delta_overall_pct'] for r in quick_wins), "%")
    for name, _, info in quick_wins:
        print(f"  - {name}: +{info['delta_overall_pct']:.2f}%")
    
    # 阶段2：中等改进（3-5天）
    medium_effort = [r for r in priority_list if 2 <= r[2]['estimated_effort'] <= 3]
    print("\n【阶段2：中等改进（3-5天）】")
    print("预期提升:", sum(r[2]['delta_overall_pct'] for r in medium_effort), "%")
    for name, _, info in medium_effort:
        print(f"  - {name}: +{info['delta_overall_pct']:.2f}%")
    
    # 阶段3：长期优化（5+天）
    long_term = [r for r in priority_list if r[2]['estimated_effort'] >= 4]
    print("\n【阶段3：长期优化（5+天）】")
    print("预期提升:", sum(r[2]['delta_overall_pct'] for r in long_term), "%")
    for name, _, info in long_term:
        print(f"  - {name}: +{info['delta_overall_pct']:.2f}%")
    
    # 计算累计提升
    total_potential = sum(r[2]['delta_overall_pct'] for r in priority_list)
    print(f"\n总潜在提升: {total_potential:.2f}%")
    print(f"预期最终准确率: {current_overall + total_potential:.1f}%")
    
    # 保存结果
    output = {
        'current_overall': current_overall,
        'target_overall': 90.0,
        'gap': 90.0 - current_overall,
        'by_roi': [
            {
                'name': name,
                **info
            }
            for name, info in results_by_roi
        ],
        'by_impact': [
            {
                'name': name,
                **info
            }
            for name, info in results_by_impact
        ],
        'priority_list': [
            {
                'priority': i,
                'name': name,
                'composite_score': score,
                **info
            }
            for i, (name, score, info) in enumerate(priority_list, 1)
        ],
        'failure_patterns': failure_patterns,
    }
    
    with open('improvement_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到 improvement_analysis.json")


if __name__ == '__main__':
    main()
