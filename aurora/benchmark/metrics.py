"""
AURORA 基准测试指标
=========================

基准评分的评估指标。

本模块提供:
    - exact_match: 精确字符串匹配
    - fuzzy_match: 模糊字符串匹配，带相似度阈值
    - llm_judge_score: LLM-as-a-Judge 评分
    - calculate_latency_stats: 延迟统计 (p50, p95, p99)
    - calculate_accuracy_by_capability: 按能力的准确率分解

指标哲学:
    - 指标应该是可组合和独立的
    - LLM-as-a-Judge 是语义评估的黄金标准
    - 延迟指标使用百分位数以确保稳健性
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aurora.benchmark.interface import BenchmarkCapability, BenchmarkResult


# -----------------------------------------------------------------------------
# LLM 提供者协议
# -----------------------------------------------------------------------------

class LLMProviderProtocol(Protocol):
    """用于 LLM-as-a-Judge 评分的 LLM 提供者协议。"""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """为给定的提示生成完成。"""
        ...


# -----------------------------------------------------------------------------
# 字符串匹配指标
# -----------------------------------------------------------------------------

"""
增强的匹配函数，具有容错评分
-------------------------------------------------
这些函数为基准评估提供宽松的匹配:
- 同义词支持 (SF = San Francisco)
- 数字格式规范化 (28 = 28岁 = twenty-eight)
- 日期格式规范化 (March 15 = 3/15 = 3月15日)
"""

# 用于容错匹配的常见同义词
_COMMON_SYNONYMS = {
    "san francisco": ["sf", "san fran"],
    "new york": ["nyc", "ny", "new york city"],
    "los angeles": ["la", "l.a."],
    "united states": ["usa", "us", "u.s.", "america"],
    "yes": ["yeah", "yep", "correct", "true", "是", "对"],
    "no": ["nope", "nah", "incorrect", "false", "否", "不是"],
}

# 构建反向查找
_SYNONYM_LOOKUP = {}
for canonical, variants in _COMMON_SYNONYMS.items():
    _SYNONYM_LOOKUP[canonical] = canonical
    for v in variants:
        _SYNONYM_LOOKUP[v.lower()] = canonical


def _normalize_number(text: str) -> str:
    """规范化数字表达式。"""
    result = text.lower()
    # 移除年龄/单位后缀
    result = re.sub(r'(\d+)\s*[岁年月日号天周个]', r'\1', result)
    result = re.sub(r'(\d+)\s*(?:years?\s*old|months?|days?)', r'\1', result)
    return result


def _normalize_date(text: str) -> str:
    """将日期表达式规范化为 M/D 格式。"""
    result = text.lower()
    month_map = {
        "january": "1", "jan": "1", "february": "2", "feb": "2",
        "march": "3", "mar": "3", "april": "4", "apr": "4",
        "may": "5", "june": "6", "jun": "6", "july": "7", "jul": "7",
        "august": "8", "aug": "8", "september": "9", "sep": "9",
        "october": "10", "oct": "10", "november": "11", "nov": "11",
        "december": "12", "dec": "12",
    }
    
    # Month Day format
    for month, num in month_map.items():
        result = re.sub(rf'\b{month}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b', rf'{num}/\1', result)
    
    # Chinese date format: 3月15日
    result = re.sub(r'(\d{1,2})月(\d{1,2})日?', r'\1/\2', result)
    
    return result


def exact_match(predicted: str, expected: str) -> float:
    """
    计算具有增强容错的精确匹配分数。

    执行不区分大小写的比较，包括:
    - 同义词支持 (SF = San Francisco)
    - 数字规范化 (28 = 28岁)
    - 日期规范化 (March 15 = 3/15)

    Args:
        predicted: 模型的预测答案
        expected: 预期/真实答案

    Returns:
        如果字符串匹配（包括同义词匹配），返回 1.0，否则返回 0.0

    Example:
        >>> exact_match("San Francisco", "san francisco")
        1.0
        >>> exact_match("SF", "San Francisco")
        1.0  # 现在通过同义词匹配!
        >>> exact_match("28岁", "28")
        1.0  # 现在通过数字规范化匹配!
    """
    if not predicted or not expected:
        return 0.0
    
    pred_norm = predicted.strip().lower()
    exp_norm = expected.strip().lower()
    
    # Direct match
    if pred_norm == exp_norm:
        return 1.0
    
    # Synonym match
    pred_canonical = _SYNONYM_LOOKUP.get(pred_norm, pred_norm)
    exp_canonical = _SYNONYM_LOOKUP.get(exp_norm, exp_norm)
    
    if pred_canonical == exp_canonical:
        return 1.0
    
    # Number normalization match
    pred_num = _normalize_number(pred_norm)
    exp_num = _normalize_number(exp_norm)
    
    if pred_num == exp_num:
        return 1.0
    
    # Date normalization match
    pred_date = _normalize_date(pred_norm)
    exp_date = _normalize_date(exp_norm)
    
    if pred_date == exp_date:
        return 1.0
    
    return 0.0


def fuzzy_match(
    predicted: str,
    expected: str,
    threshold: float = 0.7,  # 从 0.8 降低以提高容错性
) -> float:
    """
    使用序列匹配计算模糊匹配分数。

    使用 Python 的 SequenceMatcher 计算相似度比率。
    现在在阈值以下给予部分分数。

    Args:
        predicted: 模型的预测答案
        expected: 预期/真实答案
        threshold: 完全匹配的最小相似度（默认 0.7）

    Returns:
        如果 >= 阈值，返回相似度比率，否则返回部分分数

    Example:
        >>> fuzzy_match("San Francisco, CA", "San Francisco")
        0.81  # 完全分数，高于阈值
        >>> fuzzy_match("New York", "San Francisco")
        0.27  # 基于相似度的部分分数
    """
    if not predicted or not expected:
        return 0.0
    
    pred_norm = predicted.strip().lower()
    exp_norm = expected.strip().lower()
    
    # Apply normalizations before fuzzy matching
    pred_norm = _normalize_number(pred_norm)
    pred_norm = _normalize_date(pred_norm)
    exp_norm = _normalize_number(exp_norm)
    exp_norm = _normalize_date(exp_norm)
    
    ratio = SequenceMatcher(None, pred_norm, exp_norm).ratio()
    
    # Full credit if above threshold
    if ratio >= threshold:
        return ratio
    
    # Partial credit for lower similarity (scaled down)
    return ratio * 0.5


def contains_match(predicted: str, expected: str) -> float:
    """
    检查预期答案是否包含在预测答案中。

    现在包括:
    - 直接包含检查
    - 同义词包含检查
    - 关键词重叠评分

    Args:
        predicted: 模型的预测答案
        expected: 预期/真实答案

    Returns:
        基于包含质量的 [0.0, 1.0] 范围内的分数

    Example:
        >>> contains_match("The user lives in San Francisco, California.", "San Francisco")
        1.0
        >>> contains_match("The user lives in SF.", "San Francisco")
        0.95  # 同义词匹配
    """
    if not predicted or not expected:
        return 0.0
    
    pred_norm = predicted.strip().lower()
    exp_norm = expected.strip().lower()
    
    # Direct containment
    if exp_norm in pred_norm:
        return 1.0
    
    # Synonym containment
    exp_canonical = _SYNONYM_LOOKUP.get(exp_norm, exp_norm)
    if exp_canonical in _COMMON_SYNONYMS:
        for variant in _COMMON_SYNONYMS[exp_canonical]:
            if variant.lower() in pred_norm:
                return 0.95
    
    # Also check reverse: if pred has a variant of expected
    pred_canonical = _SYNONYM_LOOKUP.get(pred_norm.split()[0] if pred_norm else "", "")
    if pred_canonical and pred_canonical == exp_canonical:
        return 0.9
    
    # Keyword overlap
    exp_words = set(re.findall(r'\w+', exp_norm))
    pred_words = set(re.findall(r'\w+', pred_norm))
    
    if exp_words:
        overlap = exp_words & pred_words
        overlap_ratio = len(overlap) / len(exp_words)
        if overlap_ratio >= 0.5:
            return overlap_ratio * 0.8
    
    return 0.0


def token_f1(predicted: str, expected: str) -> float:
    """
    计算预测和预期之间的令牌级 F1 分数。

    通过在空格和标点符号上分割来标记化。
    常用于 QA 评估（例如 SQuAD）。

    Args:
        predicted: 模型的预测答案
        expected: 预期/真实答案

    Returns:
        [0.0, 1.0] 范围内的 F1 分数

    Example:
        >>> token_f1("San Francisco Bay Area", "San Francisco")
        0.8  # 2/2.5 (精确率-召回率调和平均值)
    """
    if not predicted or not expected:
        return 0.0
    
    # Simple tokenization
    def tokenize(s: str) -> set:
        return set(re.findall(r'\w+', s.lower()))
    
    pred_tokens = tokenize(predicted)
    exp_tokens = tokenize(expected)
    
    if not pred_tokens or not exp_tokens:
        return 0.0
    
    common = pred_tokens & exp_tokens
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    
    return 2 * precision * recall / (precision + recall)


# -----------------------------------------------------------------------------
# LLM-as-a-Judge 评分
# -----------------------------------------------------------------------------

# LLM-as-a-Judge 的默认提示模板
LLM_JUDGE_PROMPT_TEMPLATE = """你正在评估内存系统响应的质量。

查询: {query}
预期答案: {expected}
模型答案: {predicted}

评估模型的答案与预期答案相比对查询的解决程度。
考虑:
1. 事实正确性
2. 信息完整性
3. 与查询的相关性

提供 0 到 10 的分数，其中:
- 0-2: 完全错误或无关
- 3-4: 部分正确但缺少关键信息
- 5-6: 大部分正确，有轻微问题
- 7-8: 正确且信息完整
- 9-10: 优秀，匹配或超过预期答案

仅输出单个整数分数 (0-10)，不输出其他内容。"""


def llm_judge_score(
    predicted: str,
    expected: str,
    query: str,
    llm_provider: LLMProviderProtocol,
    prompt_template: Optional[str] = None,
    max_retries: int = 3,
) -> float:
    """
    使用 LLM-as-a-Judge 进行语义评估。

    将预测和预期答案发送给 LLM 进行评估。
    LLM 返回 0-10 的分数，该分数被规范化为 [0, 1]。

    Args:
        predicted: 模型的预测答案
        expected: 预期/真实答案
        query: 原始查询
        llm_provider: LLM 提供者实例（必须有 complete() 方法）
        prompt_template: 自定义提示模板（可选）
        max_retries: 解析失败时的重试次数

    Returns:
        [0.0, 1.0] 范围内的规范化分数

    Raises:
        ValueError: 如果 LLM 响应在重试后无法解析

    Example:
        score = llm_judge_score(
            predicted="The user enjoys hiking in the mountains",
            expected="The user likes outdoor activities, especially hiking",
            query="What are the user's hobbies?",
            llm_provider=my_llm,
        )
    """
    if not predicted:
        return 0.0
    
    template = prompt_template or LLM_JUDGE_PROMPT_TEMPLATE
    prompt = template.format(
        query=query,
        expected=expected or "(No expected answer provided)",
        predicted=predicted,
    )
    
    for attempt in range(max_retries):
        try:
            response = llm_provider.complete(prompt)
            
            # Parse the score from response
            score_text = response.strip()
            
            # Handle responses like "Score: 8" or just "8"
            match = re.search(r'\b(\d+)\b', score_text)
            if match:
                score = int(match.group(1))
                # Clamp to valid range and normalize
                score = max(0, min(10, score))
                return score / 10.0
            
        except Exception as e:
            logger.debug(f"LLM judge attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    
    # Fallback if all retries fail
    return 0.0


def llm_judge_score_batch(
    instances: List[tuple],
    llm_provider: LLMProviderProtocol,
    prompt_template: Optional[str] = None,
) -> List[float]:
    """
    使用 LLM-as-a-Judge 进行批量评估。

    对于大型数据集更高效，因为它可以潜在地
    批量请求到 LLM 提供者。

    Args:
        instances: (predicted, expected, query) 元组列表
        llm_provider: LLM 提供者实例
        prompt_template: 自定义提示模板（可选）

    Returns:
        [0.0, 1.0] 范围内的规范化分数列表
    """
    scores = []
    for predicted, expected, query in instances:
        try:
            score = llm_judge_score(
                predicted=predicted,
                expected=expected,
                query=query,
                llm_provider=llm_provider,
                prompt_template=prompt_template,
            )
            scores.append(score)
        except Exception as e:
            logger.debug(f"LLM judge scoring failed for instance: {e}")
            scores.append(0.0)
    return scores


# -----------------------------------------------------------------------------
# 延迟统计
# -----------------------------------------------------------------------------

def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """
    从延迟测量列表计算延迟统计。

    计算用于性能分析的常见百分位数指标。

    Args:
        latencies: 以毫秒为单位的延迟值列表

    Returns:
        包含以下内容的字典:
            - "mean_ms": 平均延迟
            - "std_ms": 标准差
            - "min_ms": 最小延迟
            - "max_ms": 最大延迟
            - "p50_ms": 50 百分位数（中位数）
            - "p90_ms": 90 百分位数
            - "p95_ms": 95 百分位数
            - "p99_ms": 99 百分位数
            - "count": 样本数

    Example:
        >>> stats = calculate_latency_stats([10.5, 15.2, 12.3, 50.1, 11.8])
        >>> print(f"p95: {stats['p95_ms']:.1f}ms")
        p95: 50.1ms
    """
    if not latencies:
        return {
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "count": 0,
        }
    
    arr = np.array(latencies, dtype=np.float64)
    
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "count": len(arr),
    }


# -----------------------------------------------------------------------------
# 按能力的准确率
# -----------------------------------------------------------------------------

def calculate_accuracy_by_capability(
    results: List["BenchmarkResult"],
) -> Dict[str, float]:
    """
    计算按能力维度分解的准确率指标。

    按能力对结果进行分组，并为每个组计算平均准确率，
    加上总体准确率。

    Args:
        results: BenchmarkResult 对象列表

    Returns:
        包含以下内容的字典:
            - "accuracy": 总体准确率
            - "accuracy_<capability>": 按能力的准确率
            - "count_<capability>": 按能力的样本数

    Example:
        >>> results = [...]  # BenchmarkResult 列表
        >>> metrics = calculate_accuracy_by_capability(results)
        >>> print(f"Overall: {metrics['accuracy']:.2%}")
        >>> print(f"Retrieval: {metrics['accuracy_accurate_retrieval']:.2%}")
    """
    if not results:
        return {"accuracy": 0.0}
    
    # Import here to avoid circular dependency
    from aurora.benchmark.interface import BenchmarkCapability
    
    # Group by capability
    by_capability: Dict[BenchmarkCapability, List[float]] = {}
    all_scores: List[float] = []
    
    for result in results:
        cap = result.capability
        if cap not in by_capability:
            by_capability[cap] = []
        by_capability[cap].append(result.score)
        all_scores.append(result.score)
    
    # Compute metrics
    metrics: Dict[str, float] = {
        "accuracy": float(np.mean(all_scores)) if all_scores else 0.0,
    }
    
    for cap, scores in by_capability.items():
        key = f"accuracy_{cap.value}"
        metrics[key] = float(np.mean(scores)) if scores else 0.0
        metrics[f"count_{cap.value}"] = float(len(scores))
    
    return metrics


def calculate_token_efficiency(
    results: List["BenchmarkResult"],
) -> Dict[str, float]:
    """
    计算令牌效率指标。

    要求结果在元数据中有 "tokens_used"。

    Args:
        results: BenchmarkResult 对象列表

    Returns:
        包含以下内容的字典:
            - "mean_tokens": 每个查询的平均令牌数
            - "tokens_per_correct": 每个正确答案的令牌数
            - "total_tokens": 使用的总令牌数
    """
    total_tokens = 0
    correct_tokens = 0
    correct_count = 0
    
    for result in results:
        tokens = result.metadata.get("tokens_used", 0)
        total_tokens += tokens
        
        if result.score >= 0.5:
            correct_tokens += tokens
            correct_count += 1
    
    return {
        "mean_tokens": total_tokens / len(results) if results else 0.0,
        "tokens_per_correct": correct_tokens / correct_count if correct_count > 0 else 0.0,
        "total_tokens": float(total_tokens),
    }


# -----------------------------------------------------------------------------
# 复合评分
# -----------------------------------------------------------------------------

def composite_score(
    predicted: str,
    expected: str,
    query: str,
    weights: Optional[Dict[str, float]] = None,
    llm_provider: Optional[LLMProviderProtocol] = None,
) -> float:
    """
    从多个指标计算加权复合分数。

    结合精确匹配、模糊匹配、包含和令牌 F1。
    如果有 LLM 提供者，包括 LLM-as-a-Judge 分数。

    Args:
        predicted: 模型的预测答案
        expected: 预期/真实答案
        query: 原始查询
        weights: 每个指标的可选权重字典
        llm_provider: 用于语义评估的可选 LLM 提供者

    Returns:
        [0.0, 1.0] 范围内的加权复合分数

    默认权重（不使用 LLM）:
        - exact_match: 0.3
        - fuzzy_match: 0.3
        - contains_match: 0.2
        - token_f1: 0.2

    默认权重（使用 LLM）:
        - llm_judge: 0.5
        - exact_match: 0.15
        - fuzzy_match: 0.15
        - contains_match: 0.1
        - token_f1: 0.1
    """
    if not predicted or not expected:
        return 0.0
    
    # Compute individual scores
    scores = {
        "exact_match": exact_match(predicted, expected),
        "fuzzy_match": fuzzy_match(predicted, expected),
        "contains_match": contains_match(predicted, expected),
        "token_f1": token_f1(predicted, expected),
    }
    
    # Add LLM judge if available
    if llm_provider is not None:
        try:
            scores["llm_judge"] = llm_judge_score(
                predicted, expected, query, llm_provider
            )
        except Exception as e:
            logger.debug(f"LLM judge scoring failed, falling back to non-LLM scoring: {e}")
    
    # Default weights
    if weights is None:
        if "llm_judge" in scores:
            weights = {
                "llm_judge": 0.5,
                "exact_match": 0.15,
                "fuzzy_match": 0.15,
                "contains_match": 0.1,
                "token_f1": 0.1,
            }
        else:
            weights = {
                "exact_match": 0.3,
                "fuzzy_match": 0.3,
                "contains_match": 0.2,
                "token_f1": 0.2,
            }
    
    # Compute weighted sum
    total_weight = 0.0
    weighted_sum = 0.0
    
    for metric, score in scores.items():
        w = weights.get(metric, 0.0)
        weighted_sum += w * score
        total_weight += w
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0
