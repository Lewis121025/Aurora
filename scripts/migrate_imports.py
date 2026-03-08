#!/usr/bin/env python3
"""
自动迁移 constants 导入到新的 config 模块
"""

import re
from pathlib import Path
from typing import Dict, Set

# 常量到新模块的完整映射
CONSTANT_TO_MODULE: Dict[str, str] = {
    # retrieval.py
    'INITIAL_SEARCH_K': 'aurora.core.config.retrieval',
    'ATTRACTOR_SEARCH_K': 'aurora.core.config.retrieval',
    'FEEDBACK_SEARCH_K': 'aurora.core.config.retrieval',
    'NEGATIVE_SAMPLE_SEARCH_K': 'aurora.core.config.retrieval',
    'DEFAULT_PAGERANK_DAMPING': 'aurora.core.config.retrieval',
    'DEFAULT_PAGERANK_MAX_ITER': 'aurora.core.config.retrieval',
    'DEFAULT_MEAN_SHIFT_STEPS': 'aurora.core.config.retrieval',
    'RELATIONSHIP_BONUS_SCORE': 'aurora.core.config.retrieval',
    'STORY_SIMILARITY_BONUS': 'aurora.core.config.retrieval',
    'MASS_BONUS_COEFFICIENT': 'aurora.core.config.retrieval',
    'SEMANTIC_NEIGHBORS_K': 'aurora.core.config.retrieval',
    'MAX_RECENT_PLOTS_FOR_RETRIEVAL': 'aurora.core.config.retrieval',
    'RECENT_PLOTS_FOR_FEEDBACK': 'aurora.core.config.retrieval',
    'RECENT_ENCODED_PLOTS_WINDOW': 'aurora.core.config.retrieval',

    # numeric.py
    'EPSILON': 'aurora.core.config.numeric',
    'EPSILON_LOG': 'aurora.core.config.numeric',
    'EPSILON_PRIOR': 'aurora.core.config.numeric',
    'TEXT_LENGTH_NORMALIZATION': 'aurora.core.config.numeric',
    'SNIPPET_MAX_LENGTH': 'aurora.core.config.numeric',
    'EVENT_SUMMARY_MAX_LENGTH': 'aurora.core.config.numeric',
    'FALLBACK_ACTION_MAX_LENGTH': 'aurora.core.config.numeric',
    'DEFAULT_LLM_TEMPERATURE': 'aurora.core.config.numeric',
    'DEFAULT_LLM_TIMEOUT': 'aurora.core.config.numeric',
    'DEFAULT_CACHE_TTL': 'aurora.core.config.numeric',
    'DEFAULT_CACHE_MAX_SIZE': 'aurora.core.config.numeric',
    'TRUST_BASE': 'aurora.core.config.numeric',

    # identity.py
    'REINFORCEMENT_WEIGHT': 'aurora.core.config.identity',
    'CHALLENGE_WEIGHT': 'aurora.core.config.identity',
    'NOVELTY_WEIGHT': 'aurora.core.config.identity',
    'IDENTITY_RELEVANCE_WEIGHT': 'aurora.core.config.identity',
    'VOI_DECISION_WEIGHT': 'aurora.core.config.identity',
    'RELATIONSHIP_HEALTH_WEIGHT': 'aurora.core.config.identity',
    'RELATIONSHIP_RECENCY_WEIGHT': 'aurora.core.config.identity',
    'RELATIONSHIP_POSITION_WEIGHT': 'aurora.core.config.identity',
    'MODERATE_SIMILARITY_MIN': 'aurora.core.config.identity',
    'MODERATE_SIMILARITY_MAX': 'aurora.core.config.identity',
    'ROLE_CONSISTENCY_THRESHOLD': 'aurora.core.config.identity',
    'IDENTITY_RELEVANCE_THRESHOLD': 'aurora.core.config.identity',
    'TENSION_SIMILARITY_MIN': 'aurora.core.config.identity',
    'TENSION_SIMILARITY_MAX': 'aurora.core.config.identity',
    'HARMONY_SIMILARITY_MIN': 'aurora.core.config.identity',
    'MAX_IDENTITY_DIMENSIONS': 'aurora.core.config.identity',
    'IDENTITY_DIMENSION_GROWTH_RATE': 'aurora.core.config.identity',
    'INTERACTION_COUNT_LOG_NORMALIZER': 'aurora.core.config.identity',
    'QUALITY_DELTA_COEFFICIENT': 'aurora.core.config.identity',
    'MAX_QUALITY_DELTA': 'aurora.core.config.identity',

    # storage.py
    'COLD_START_FORCE_STORE_COUNT': 'aurora.core.config.storage',
    'DENSITY_MIN_SAMPLES': 'aurora.core.config.storage',
    'DEFAULT_COLD_START_SURPRISE': 'aurora.core.config.storage',
    'MIN_STORE_PROB': 'aurora.core.config.storage',
    'WEAK_EDGE_MIN_WEIGHT': 'aurora.core.config.storage',
    'WEAK_EDGE_MIN_SUCCESSES': 'aurora.core.config.storage',
    'NODE_MERGE_SIMILARITY_THRESHOLD': 'aurora.core.config.storage',
    'ARCHIVE_STALE_DAYS_THRESHOLD': 'aurora.core.config.storage',
    'ARCHIVE_MIN_ACCESS_COUNT': 'aurora.core.config.storage',

    # coherence.py
    'OPPOSITION_SCORE_THRESHOLD': 'aurora.core.config.coherence',
    'HIGH_SIMILARITY_THRESHOLD': 'aurora.core.config.coherence',
    'ANTI_CORRELATION_THRESHOLD': 'aurora.core.config.coherence',
    'UNFINISHED_STORY_HOURS': 'aurora.core.config.coherence',
    'MAX_COHERENCE_PAIRS': 'aurora.core.config.coherence',
    'BELIEF_PROPAGATION_ITERATIONS': 'aurora.core.config.coherence',
    'COHERENCE_WEIGHTS': 'aurora.core.config.coherence',
    'CONFLICT_CHECK_SIMILARITY_THRESHOLD': 'aurora.core.config.coherence',
    'CONFLICT_CHECK_K': 'aurora.core.config.coherence',
    'CONFLICT_PROBABILITY_THRESHOLD': 'aurora.core.config.coherence',
    'MAX_CONFLICTS_PER_INGEST': 'aurora.core.config.coherence',
    'SEMANTIC_CONFLICT_WEIGHT': 'aurora.core.config.coherence',
    'KNOWLEDGE_TYPE_CONFLICT_WEIGHT': 'aurora.core.config.coherence',
    'CONCURRENT_TIME_THRESHOLD': 'aurora.core.config.coherence',

    # evolution.py
    'REFRAME_AGE_DAYS_THRESHOLD': 'aurora.core.config.evolution',
    'REFRAME_ACCESS_COUNT_THRESHOLD': 'aurora.core.config.evolution',
    'PERIODIC_REFLECTION_AGE_DAYS': 'aurora.core.config.evolution',
    'PERIODIC_REFLECTION_ACCESS_COUNT': 'aurora.core.config.evolution',
    'GROWTH_HINDRANCE_AGE_SECONDS': 'aurora.core.config.evolution',
    'STORY_ABANDONMENT_THRESHOLD_DAYS': 'aurora.core.config.evolution',
    'CLIMAX_TENSION_WINDOW': 'aurora.core.config.evolution',
    'CLIMAX_DECLINE_RATIO': 'aurora.core.config.evolution',
    'RESOLUTION_TENSION_DROP_RATIO': 'aurora.core.config.evolution',
    'RESOLUTION_MIN_ARC_LENGTH': 'aurora.core.config.evolution',

    # knowledge.py
    'KNOWLEDGE_CLASSIFICATION_MIN_CONFIDENCE': 'aurora.core.config.knowledge',
    'COMPLEMENTARY_TRAIT_SIM_MIN': 'aurora.core.config.knowledge',
    'COMPLEMENTARY_TRAIT_SIM_MAX': 'aurora.core.config.knowledge',
    'CONTRADICTORY_TRAIT_SIM_THRESHOLD': 'aurora.core.config.knowledge',
    'KNOWLEDGE_TYPE_WEIGHT_STATE': 'aurora.core.config.knowledge',
    'KNOWLEDGE_TYPE_WEIGHT_STATIC': 'aurora.core.config.knowledge',
    'KNOWLEDGE_TYPE_WEIGHT_TRAIT': 'aurora.core.config.knowledge',
    'KNOWLEDGE_TYPE_WEIGHT_VALUE': 'aurora.core.config.knowledge',
    'KNOWLEDGE_TYPE_WEIGHT_PREFERENCE': 'aurora.core.config.knowledge',
    'KNOWLEDGE_TYPE_WEIGHT_BEHAVIOR': 'aurora.core.config.knowledge',
    'UPDATE_TIME_GAP_THRESHOLD': 'aurora.core.config.knowledge',
    'REINFORCEMENT_TIME_WINDOW': 'aurora.core.config.knowledge',
    'UPDATE_HIGH_SIMILARITY_THRESHOLD': 'aurora.core.config.knowledge',
    'UPDATE_MODERATE_SIMILARITY_THRESHOLD': 'aurora.core.config.knowledge',
    'UPDATE_KEYWORDS': 'aurora.core.config.knowledge',
    'NUMERIC_CHANGE_INDICATORS': 'aurora.core.config.knowledge',

    # query_types.py
    'TEMPORAL_KEYWORDS': 'aurora.core.config.query_types',
    'CAUSAL_KEYWORDS': 'aurora.core.config.query_types',
    'MULTI_HOP_KEYWORDS': 'aurora.core.config.query_types',
    'AGGREGATION_KEYWORDS': 'aurora.core.config.query_types',
    'MULTI_HOP_K_MULTIPLIER': 'aurora.core.config.query_types',
    'TEMPORAL_SORT_WEIGHT': 'aurora.core.config.query_types',
    'MULTI_HOP_EXTRA_PAGERANK_ITER': 'aurora.core.config.query_types',
    'AGGREGATION_K_MULTIPLIER': 'aurora.core.config.query_types',
    'RECENT_ANCHOR_KEYWORDS': 'aurora.core.config.query_types',
    'EARLIEST_ANCHOR_KEYWORDS': 'aurora.core.config.query_types',
    'SPAN_ANCHOR_KEYWORDS': 'aurora.core.config.query_types',
    'TEMPORAL_DIVERSITY_BUCKETS': 'aurora.core.config.query_types',
    'TEMPORAL_DIVERSITY_MMR_LAMBDA': 'aurora.core.config.query_types',
    'FACTUAL_PLOT_PRIORITY_BOOST': 'aurora.core.config.query_types',
    'FACTUAL_SEMANTIC_WEIGHT': 'aurora.core.config.query_types',
    'FACTUAL_ATTRACTOR_WEIGHT': 'aurora.core.config.query_types',
    'BENCHMARK_DEFAULT_K': 'aurora.core.config.query_types',
    'BENCHMARK_MULTI_SESSION_K': 'aurora.core.config.query_types',
    'BENCHMARK_AGGREGATION_K': 'aurora.core.config.query_types',
    'AGGREGATION_CONTEXT_MAX_RESULTS': 'aurora.core.config.query_types',
    'SINGLE_SESSION_USER_K_MULTIPLIER': 'aurora.core.config.query_types',
    'SINGLE_SESSION_USER_MAX_CONTEXT': 'aurora.core.config.query_types',
    'KEYWORD_MATCH_BOOST': 'aurora.core.config.query_types',
    'KEYWORD_MATCH_MIN_RATIO': 'aurora.core.config.query_types',
    'USER_ROLE_PRIORITY_BOOST': 'aurora.core.config.query_types',
    'QUESTION_STOP_WORDS': 'aurora.core.config.query_types',
    'QUESTION_TYPE_HINT_MAPPINGS': 'aurora.core.config.query_types',
    'FACT_KEY_BOOST_MAX': 'aurora.core.config.query_types',
    'FACT_KEY_MATCH_THRESHOLD': 'aurora.core.config.query_types',
}


def extract_imports(content: str) -> Set[str]:
    """从文件内容中提取导入的常量"""
    imports = set()

    # 匹配 from aurora.core.constants import (...)
    pattern = r'from aurora\.core\.constants import \((.*?)\)'
    matches = re.findall(pattern, content, re.DOTALL)
    for match in matches:
        constants = [c.strip() for c in match.split(',') if c.strip()]
        imports.update(constants)

    # 匹配 from aurora.core.constants import X, Y, Z
    pattern = r'from aurora\.core\.constants import ([^\n]+)'
    matches = re.findall(pattern, content)
    for match in matches:
        if '(' not in match:  # 避免重复处理多行导入
            constants = [c.strip() for c in match.split(',') if c.strip()]
            imports.update(constants)

    return imports


def migrate_file(file_path: Path) -> bool:
    """迁移单个文件"""
    content = file_path.read_text()

    if 'from aurora.core.constants import' not in content:
        return False

    # 提取使用的常量
    imports = extract_imports(content)
    if not imports:
        return False

    # 按模块分组
    module_to_constants: Dict[str, Set[str]] = {}
    for constant in imports:
        module = CONSTANT_TO_MODULE.get(constant)
        if not module:
            print(f"  ⚠️  Unknown constant: {constant}")
            continue
        if module not in module_to_constants:
            module_to_constants[module] = set()
        module_to_constants[module].add(constant)

    # 移除旧导入
    content = re.sub(
        r'from aurora\.core\.constants import \([^)]+\)',
        '',
        content,
        flags=re.DOTALL
    )
    content = re.sub(
        r'from aurora\.core\.constants import [^\n]+\n',
        '',
        content
    )

    # 生成新导入
    new_imports = []
    for module in sorted(module_to_constants.keys()):
        constants = sorted(module_to_constants[module])
        if len(constants) == 1:
            new_imports.append(f'from {module} import {constants[0]}')
        elif len(constants) <= 3:
            new_imports.append(f'from {module} import {", ".join(constants)}')
        else:
            # 多行导入
            new_imports.append(f'from {module} import (')
            for i, const in enumerate(constants):
                if i < len(constants) - 1:
                    new_imports.append(f'    {const},')
                else:
                    new_imports.append(f'    {const},')
            new_imports.append(')')

    # 找到导入区域的结束位置
    lines = content.split('\n')
    import_end = 0
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            import_end = i + 1
        elif import_end > 0 and line.strip() and not line.startswith('#'):
            break

    # 插入新导入
    lines.insert(import_end, '\n'.join(new_imports))
    content = '\n'.join(lines)

    # 清理多余的空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    file_path.write_text(content)
    return True


def main():
    project_root = Path(__file__).parent.parent
    aurora_dir = project_root / 'aurora'

    files = list(aurora_dir.rglob('*.py'))
    migrated = 0

    for file_path in files:
        if 'config' in str(file_path) or 'migrate_constants' in str(file_path):
            continue

        try:
            if migrate_file(file_path):
                print(f'✓ {file_path.relative_to(project_root)}')
                migrated += 1
        except Exception as e:
            print(f'✗ {file_path.relative_to(project_root)}: {e}')

    print(f'\n迁移完成：{migrated} 个文件')


if __name__ == '__main__':
    main()
