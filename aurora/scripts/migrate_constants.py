"""
配置常量迁移脚本
==================

将 aurora.core.constants 的导入自动迁移到新的模块化结构。

用法：
    python -m aurora.scripts.migrate_constants <file_or_directory>

示例：
    python -m aurora.scripts.migrate_constants aurora/
    python -m aurora.scripts.migrate_constants aurora/core/memory/engine.py
"""

import re
import sys
from pathlib import Path
from typing import Dict, Set

# 常量到新模块的映射
CONSTANT_TO_MODULE = {
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
}


def analyze_file(file_path: Path) -> Set[str]:
    """分析文件中使用的常量"""
    content = file_path.read_text()
    used_constants = set()

    # 查找所有大写常量的使用
    for constant in CONSTANT_TO_MODULE.keys():
        if re.search(rf'\b{constant}\b', content):
            used_constants.add(constant)

    return used_constants


def migrate_file(file_path: Path, dry_run: bool = False) -> bool:
    """迁移单个文件"""
    content = file_path.read_text()
    original_content = content

    # 检查是否使用了 constants
    if 'from aurora.core.constants import' not in content and \
       'from aurora.core import constants' not in content:
        return False

    used_constants = analyze_file(file_path)
    if not used_constants:
        return False

    unknown_constants = sorted(c for c in used_constants if c not in CONSTANT_TO_MODULE)
    if unknown_constants:
        print(
            f"Skipping {file_path}: unmapped constants require manual migration: "
            f"{', '.join(unknown_constants)}"
        )
        return False

    # 按模块分组常量
    module_to_constants: Dict[str, Set[str]] = {}
    for constant in used_constants:
        module = CONSTANT_TO_MODULE[constant]
        if module not in module_to_constants:
            module_to_constants[module] = set()
        module_to_constants[module].add(constant)

    # 移除旧的导入
    content = re.sub(
        r'from aurora\.core\.constants import.*?\n',
        '',
        content
    )
    content = re.sub(
        r'from aurora\.core import constants\n',
        '',
        content
    )

    # 添加新的导入
    new_imports = []
    for module, constants in sorted(module_to_constants.items()):
        constants_str = ', '.join(sorted(constants))
        new_imports.append(f'from {module} import {constants_str}')

    # 在文件开头的导入区域插入新导入
    import_section_end = 0
    for i, line in enumerate(content.split('\n')):
        if line.startswith('from ') or line.startswith('import '):
            import_section_end = i + 1

    lines = content.split('\n')
    lines.insert(import_section_end, '\n'.join(new_imports))
    content = '\n'.join(lines)

    if dry_run:
        print(f"Would migrate {file_path}")
        print(f"  Used constants: {len(used_constants)}")
        print(f"  New imports: {len(new_imports)}")
        return True

    if content != original_content:
        file_path.write_text(content)
        print(f"✓ Migrated {file_path}")
        return True

    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m aurora.scripts.migrate_constants <path> [--dry-run]")
        sys.exit(1)

    path = Path(sys.argv[1])
    dry_run = '--dry-run' in sys.argv

    if path.is_file():
        files = [path]
    else:
        files = list(path.rglob('*.py'))

    migrated = 0
    for file_path in files:
        if migrate_file(file_path, dry_run=dry_run):
            migrated += 1

    print(f"\n{'Would migrate' if dry_run else 'Migrated'} {migrated} files")


if __name__ == '__main__':
    main()
