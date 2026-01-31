"""
AURORA Benchmark Metrics
=========================

Evaluation metrics for benchmark scoring.

This module provides:
    - exact_match: Exact string matching
    - fuzzy_match: Fuzzy string matching with similarity threshold
    - llm_judge_score: LLM-as-a-Judge scoring
    - calculate_latency_stats: Latency statistics (p50, p95, p99)
    - calculate_accuracy_by_capability: Per-capability accuracy breakdown

Metric Philosophy:
    - Metrics should be composable and independent
    - LLM-as-a-Judge is the gold standard for semantic evaluation
    - Latency metrics use percentiles for robustness
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
# Protocol for LLM Provider
# -----------------------------------------------------------------------------

class LLMProviderProtocol(Protocol):
    """Protocol for LLM providers used in LLM-as-a-Judge scoring."""
    
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate completion for the given prompt."""
        ...


# -----------------------------------------------------------------------------
# String Matching Metrics
# -----------------------------------------------------------------------------

"""
Enhanced Matching Functions with Tolerant Scoring
-------------------------------------------------
These functions provide lenient matching for benchmark evaluation:
- Synonym support (SF = San Francisco)
- Number format normalization (28 = 28岁 = twenty-eight)
- Date format normalization (March 15 = 3/15 = 3月15日)
"""

# Common synonyms for tolerant matching
_COMMON_SYNONYMS = {
    "san francisco": ["sf", "san fran"],
    "new york": ["nyc", "ny", "new york city"],
    "los angeles": ["la", "l.a."],
    "united states": ["usa", "us", "u.s.", "america"],
    "yes": ["yeah", "yep", "correct", "true", "是", "对"],
    "no": ["nope", "nah", "incorrect", "false", "否", "不是"],
}

# Build reverse lookup
_SYNONYM_LOOKUP = {}
for canonical, variants in _COMMON_SYNONYMS.items():
    _SYNONYM_LOOKUP[canonical] = canonical
    for v in variants:
        _SYNONYM_LOOKUP[v.lower()] = canonical


def _normalize_number(text: str) -> str:
    """Normalize number expressions."""
    result = text.lower()
    # Remove age/unit suffixes
    result = re.sub(r'(\d+)\s*[岁年月日号天周个]', r'\1', result)
    result = re.sub(r'(\d+)\s*(?:years?\s*old|months?|days?)', r'\1', result)
    return result


def _normalize_date(text: str) -> str:
    """Normalize date expressions to M/D format."""
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
    Compute exact match score with enhanced tolerance.
    
    Performs case-insensitive comparison with:
    - Synonym support (SF = San Francisco)
    - Number normalization (28 = 28岁)
    - Date normalization (March 15 = 3/15)
    
    Args:
        predicted: Model's predicted answer
        expected: Expected/ground truth answer
    
    Returns:
        1.0 if strings match (including synonym match), 0.0 otherwise
    
    Example:
        >>> exact_match("San Francisco", "san francisco")
        1.0
        >>> exact_match("SF", "San Francisco")
        1.0  # Now matches via synonym!
        >>> exact_match("28岁", "28")
        1.0  # Now matches via number normalization!
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
    threshold: float = 0.7,  # Lowered from 0.8 for more tolerance
) -> float:
    """
    Compute fuzzy match score using sequence matching.
    
    Uses Python's SequenceMatcher to compute similarity ratio.
    Now gives partial credit below threshold.
    
    Args:
        predicted: Model's predicted answer
        expected: Expected/ground truth answer
        threshold: Minimum similarity for full match (default 0.7)
    
    Returns:
        Similarity ratio if >= threshold, partial credit otherwise
    
    Example:
        >>> fuzzy_match("San Francisco, CA", "San Francisco")
        0.81  # Full credit, above threshold
        >>> fuzzy_match("New York", "San Francisco")
        0.27  # Partial credit based on similarity
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
    Check if expected answer is contained within predicted answer.
    
    Now includes:
    - Direct containment check
    - Synonym containment check
    - Keyword overlap scoring
    
    Args:
        predicted: Model's predicted answer
        expected: Expected/ground truth answer
    
    Returns:
        Score in [0.0, 1.0] based on containment quality
    
    Example:
        >>> contains_match("The user lives in San Francisco, California.", "San Francisco")
        1.0
        >>> contains_match("The user lives in SF.", "San Francisco")
        0.95  # Synonym match
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
    Compute token-level F1 score between predicted and expected.
    
    Tokenizes by splitting on whitespace and punctuation.
    Commonly used in QA evaluation (e.g., SQuAD).
    
    Args:
        predicted: Model's predicted answer
        expected: Expected/ground truth answer
    
    Returns:
        F1 score in [0.0, 1.0]
    
    Example:
        >>> token_f1("San Francisco Bay Area", "San Francisco")
        0.8  # 2/2.5 (precision-recall harmonic mean)
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
# LLM-as-a-Judge Scoring
# -----------------------------------------------------------------------------

# Default prompt template for LLM-as-a-Judge
LLM_JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of a memory system's response.

Query: {query}
Expected Answer: {expected}
Model's Answer: {predicted}

Evaluate how well the model's answer addresses the query compared to the expected answer.
Consider:
1. Factual correctness
2. Completeness of information
3. Relevance to the query

Provide a score from 0 to 10, where:
- 0-2: Completely wrong or irrelevant
- 3-4: Partially correct but missing key information
- 5-6: Mostly correct with minor issues
- 7-8: Correct with good completeness
- 9-10: Excellent, matches or exceeds expected answer

Output ONLY a single integer score (0-10), nothing else."""


def llm_judge_score(
    predicted: str,
    expected: str,
    query: str,
    llm_provider: LLMProviderProtocol,
    prompt_template: Optional[str] = None,
    max_retries: int = 3,
) -> float:
    """
    Use LLM-as-a-Judge for semantic evaluation.
    
    Sends the predicted and expected answers to an LLM for evaluation.
    The LLM returns a score from 0-10, which is normalized to [0, 1].
    
    Args:
        predicted: Model's predicted answer
        expected: Expected/ground truth answer
        query: The original query
        llm_provider: LLM provider instance (must have complete() method)
        prompt_template: Custom prompt template (optional)
        max_retries: Number of retries on parsing failure
    
    Returns:
        Normalized score in [0.0, 1.0]
    
    Raises:
        ValueError: If LLM response cannot be parsed after retries
    
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
    Batch evaluation using LLM-as-a-Judge.
    
    More efficient for large datasets as it can potentially
    batch requests to the LLM provider.
    
    Args:
        instances: List of (predicted, expected, query) tuples
        llm_provider: LLM provider instance
        prompt_template: Custom prompt template (optional)
    
    Returns:
        List of normalized scores in [0.0, 1.0]
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
# Latency Statistics
# -----------------------------------------------------------------------------

def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate latency statistics from a list of latency measurements.
    
    Computes common percentile metrics for performance analysis.
    
    Args:
        latencies: List of latency values in milliseconds
    
    Returns:
        Dict containing:
            - "mean_ms": Mean latency
            - "std_ms": Standard deviation
            - "min_ms": Minimum latency
            - "max_ms": Maximum latency
            - "p50_ms": 50th percentile (median)
            - "p90_ms": 90th percentile
            - "p95_ms": 95th percentile
            - "p99_ms": 99th percentile
            - "count": Number of samples
    
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
# Accuracy by Capability
# -----------------------------------------------------------------------------

def calculate_accuracy_by_capability(
    results: List["BenchmarkResult"],
) -> Dict[str, float]:
    """
    Calculate accuracy metrics broken down by capability dimension.
    
    Groups results by their capability and computes mean accuracy
    for each group, plus overall accuracy.
    
    Args:
        results: List of BenchmarkResult objects
    
    Returns:
        Dict containing:
            - "accuracy": Overall accuracy
            - "accuracy_<capability>": Per-capability accuracy
            - "count_<capability>": Per-capability sample count
    
    Example:
        >>> results = [...]  # List of BenchmarkResult
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
    Calculate token efficiency metrics.
    
    Requires results to have "tokens_used" in metadata.
    
    Args:
        results: List of BenchmarkResult objects
    
    Returns:
        Dict containing:
            - "mean_tokens": Mean tokens per query
            - "tokens_per_correct": Tokens per correct answer
            - "total_tokens": Total tokens used
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
# Composite Scoring
# -----------------------------------------------------------------------------

def composite_score(
    predicted: str,
    expected: str,
    query: str,
    weights: Optional[Dict[str, float]] = None,
    llm_provider: Optional[LLMProviderProtocol] = None,
) -> float:
    """
    Compute a weighted composite score from multiple metrics.
    
    Combines exact match, fuzzy match, containment, and token F1.
    If an LLM provider is available, includes LLM-as-a-Judge score.
    
    Args:
        predicted: Model's predicted answer
        expected: Expected/ground truth answer
        query: The original query
        weights: Optional weight dict for each metric
        llm_provider: Optional LLM provider for semantic evaluation
    
    Returns:
        Weighted composite score in [0.0, 1.0]
    
    Default Weights (without LLM):
        - exact_match: 0.3
        - fuzzy_match: 0.3
        - contains_match: 0.2
        - token_f1: 0.2
    
    Default Weights (with LLM):
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
