"""
MemoryAgentBench Adapter
========================

Adapter for MemoryAgentBench (2025.07) - Academic benchmark for LLM Agent memory systems.

Dataset: HuggingFace `ai-hyz/MemoryAgentBench`
- 146 evaluation instances total:
  - 22 Accurate Retrieval (AR)
  - 6 Test-Time Learning (TTL)
  - 110 Long-Range Understanding (LRU)
  - 8 Conflict Resolution (CR)

AURORA Capability Mapping:
| Benchmark Capability | AURORA Implementation |
|---------------------|----------------------|
| Accurate Retrieval  | query() + FieldRetriever |
| Test-Time Learning  | ingest() + evolve() |
| Long-Range Understanding | Story/Theme materialization + plot synthesis |
| Conflict Resolution | contradiction edge scanning + resolution synthesis |

Usage:
    from aurora.benchmarks.adapters.memoryagentbench import MemoryAgentBenchAdapter
    from aurora.soul.engine import AuroraSoul, SoulConfig

    adapter = MemoryAgentBenchAdapter(llm_provider=llm)
    memory = AuroraSoul(cfg=SoulConfig())

    results, metrics = adapter.run_benchmark(
        dataset_path="ai-hyz/MemoryAgentBench",
        memory=memory,
    )

    print(metrics.accuracy)
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from aurora.benchmarks.interface import (
    BenchmarkAdapter,
    BenchmarkCapability,
    BenchmarkInstance,
    BenchmarkResult,
    EvaluationConfig,
    EvaluationMetrics,
)
from aurora.integrations.llm.Prompt.memoryagentbench_prompt import (
    MEMORYAGENTBENCH_JUDGE_SYSTEM_PROMPT,
    MEMORYAGENTBENCH_JUDGE_USER_PROMPT,
)
from aurora.integrations.llm.Prompt.qa_prompt import (
    build_qa_prompt,
    question_type_from_query_type,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Capability type mapping from dataset
CAPABILITY_MAPPING = {
    "accurate_retrieval": BenchmarkCapability.ACCURATE_RETRIEVAL,
    "ar": BenchmarkCapability.ACCURATE_RETRIEVAL,
    "test_time_learning": BenchmarkCapability.TEST_TIME_LEARNING,
    "ttl": BenchmarkCapability.TEST_TIME_LEARNING,
    "long_range_understanding": BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
    "lru": BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
    "conflict_resolution": BenchmarkCapability.CONFLICT_RESOLUTION,
    "cr": BenchmarkCapability.CONFLICT_RESOLUTION,
}

# Task type strings for compatibility with base interface
TASK_TYPE_AR = "accurate_retrieval"
TASK_TYPE_TTL = "test_time_learning"
TASK_TYPE_LRU = "long_range_understanding"
TASK_TYPE_CR = "conflict_resolution"

# Conversation turn markers
USER_MARKER = "User:"
ASSISTANT_MARKER = "Assistant:"
SYSTEM_MARKER = "System:"

# =============================================================================
# Helper Functions
# =============================================================================


def parse_conversation_turns(context: str) -> List[Dict[str, Any]]:
    """Parse conversation context into structured turns.

    Handles various formats:
    - "User: message\\nAssistant: response"
    - "Human: message\\nAI: response"
    - Multi-line messages

    Args:
        context: Raw conversation text

    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts
    """
    turns: List[Dict[str, Any]] = []

    # Normalize markers
    normalized = context.replace("Human:", "User:")
    normalized = normalized.replace("AI:", "Assistant:")
    normalized = normalized.replace("Bot:", "Assistant:")

    # Split by role markers
    pattern = r"(User:|Assistant:|System:)"
    parts = re.split(pattern, normalized)

    current_role = None
    current_content: List[str] = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part in ("User:", "Assistant:", "System:"):
            # Save previous turn
            if current_role and current_content:
                role = (
                    "user"
                    if current_role == "User:"
                    else ("assistant" if current_role == "Assistant:" else "system")
                )
                turns.append(
                    {
                        "role": role,
                        "content": " ".join(current_content).strip(),
                        "speaker": role,
                        "text": " ".join(current_content).strip(),
                    }
                )

            current_role = part
            current_content = []
        else:
            current_content.append(part)

    # Don't forget the last turn
    if current_role and current_content:
        role = (
            "user"
            if current_role == "User:"
            else ("assistant" if current_role == "Assistant:" else "system")
        )
        turns.append(
            {
                "role": role,
                "content": " ".join(current_content).strip(),
                "speaker": role,
                "text": " ".join(current_content).strip(),
            }
        )

    # If no structured format found, treat entire context as user message
    if not turns:
        turns.append(
            {
                "role": "user",
                "content": context.strip(),
                "speaker": "user",
                "text": context.strip(),
            }
        )

    return turns


def extract_conflicting_facts(context: str, metadata: Dict[str, Any]) -> List[str]:
    """Extract conflicting facts from context for CR capability.

    Args:
        context: Conversation context
        metadata: Instance metadata that may contain conflict info

    Returns:
        List of conflicting fact strings
    """
    facts: List[str] = []

    # Check metadata first
    if "conflicts" in metadata:
        facts.extend(metadata["conflicts"])

    if "old_fact" in metadata and "new_fact" in metadata:
        facts.append(f"Old: {metadata['old_fact']}")
        facts.append(f"New: {metadata['new_fact']}")

    # Pattern-based extraction
    conflict_patterns = [
        r"(?:originally|initially|before)[:\s]+(.+?)(?:,|\.|\n|but)",
        r"(?:now|currently|updated|changed to)[:\s]+(.+?)(?:,|\.|\n)",
        r"(?:correction|update|change)[:\s]+(.+?)(?:,|\.|\n)",
    ]

    for pattern in conflict_patterns:
        matches = re.findall(pattern, context, re.IGNORECASE)
        facts.extend(matches)

    return facts


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison.

    Args:
        answer: Raw answer string

    Returns:
        Normalized string
    """
    # Lowercase
    normalized = answer.lower().strip()

    # Remove common prefixes
    prefixes = [
        "the answer is",
        "answer:",
        "response:",
        "i think",
        "based on the context,",
    ]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()

    # Remove punctuation at edges
    normalized = normalized.strip(".,!?;:")

    return normalized


# =============================================================================
# Synonym Dictionary for Tolerant Matching
# =============================================================================

# Common synonyms and abbreviations
SYNONYMS: Dict[str, List[str]] = {
    # Cities
    "san francisco": ["sf", "san fran", "the city", "frisco"],
    "new york": ["nyc", "ny", "new york city", "the big apple"],
    "los angeles": ["la", "l.a.", "los angeles", "city of angels"],
    "washington": ["dc", "d.c.", "washington dc", "washington d.c."],
    "philadelphia": ["philly", "phl"],
    "las vegas": ["vegas", "lv"],
    # Countries
    "united states": ["usa", "us", "u.s.", "u.s.a.", "america"],
    "united kingdom": ["uk", "u.k.", "britain", "great britain"],
    "中国": ["china", "cn", "prc"],
    # Time expressions
    "today": ["now", "this day"],
    "yesterday": ["the day before", "one day ago"],
    "tomorrow": ["the next day", "one day later"],
    # Common phrases
    "yes": ["yeah", "yep", "yup", "correct", "right", "affirmative", "true", "是", "对"],
    "no": ["nope", "nah", "negative", "incorrect", "wrong", "false", "否", "不是"],
    # Occupations
    "doctor": ["dr", "physician", "医生"],
    "engineer": ["软件工程师", "software engineer", "developer", "工程师"],
    "teacher": ["instructor", "professor", "老师", "教师"],
    # Relationships
    "wife": ["spouse", "partner", "妻子", "太太", "老婆"],
    "husband": ["spouse", "partner", "丈夫", "老公"],
    "friend": ["朋友", "好友", "buddy", "pal"],
    "colleague": ["coworker", "同事", "workmate"],
}

# Build reverse lookup for efficiency
_SYNONYM_LOOKUP: Dict[str, str] = {}
for canonical, variants in SYNONYMS.items():
    _SYNONYM_LOOKUP[canonical] = canonical
    for variant in variants:
        _SYNONYM_LOOKUP[variant.lower()] = canonical


# =============================================================================
# Number Format Normalization
# =============================================================================

# English number words
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
}

# Chinese number words
CHINESE_NUMBERS = {
    "零": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
    "十一": "11",
    "十二": "12",
    "十三": "13",
    "十四": "14",
    "十五": "15",
    "十六": "16",
    "十七": "17",
    "十八": "18",
    "十九": "19",
    "二十": "20",
    "三十": "30",
    "四十": "40",
    "五十": "50",
    "六十": "60",
    "七十": "70",
    "八十": "80",
    "九十": "90",
    "百": "100",
}


def normalize_number(text: str) -> str:
    """Normalize number expressions in text.

    Converts:
    - "twenty-eight" -> "28"
    - "二十八" -> "28"
    - "28岁" -> "28"
    - "28 years old" -> "28"

    Args:
        text: Input text

    Returns:
        Text with normalized numbers
    """
    result = text.lower()

    # Remove Chinese age/unit suffixes
    result = re.sub(r"(\d+)\s*[岁年月日号天周个]", r"\1", result)

    # Remove English age/unit suffixes
    result = re.sub(r"(\d+)\s*(?:years?\s*old|yr(?:s)?|months?|days?|weeks?|hours?)", r"\1", result)

    # Convert English number words (handle compound like "twenty-eight")
    for word, digit in sorted(NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
        result = re.sub(rf"\b{word}\b", digit, result)

    # Handle compound numbers like "twenty eight" -> "28"
    def combine_tens_units(match: re.Match[str]) -> str:
        tens = match.group(1)
        units = match.group(2) if (match.lastindex or 0) >= 2 else "0"
        try:
            return str(int(tens) + int(units))
        except ValueError:
            return match.group(0)

    result = re.sub(
        r"\b(20|30|40|50|60|70|80|90)[\s-]*(1|2|3|4|5|6|7|8|9)\b", combine_tens_units, result
    )

    # Convert Chinese numbers (handle compound like "二十八")
    for chinese, digit in sorted(CHINESE_NUMBERS.items(), key=lambda x: -len(x[0])):
        result = result.replace(chinese, digit)

    # Handle Chinese compound numbers (e.g., "2十8" -> "28")
    def combine_chinese_compound(match: re.Match[str]) -> str:
        if match.group(2):  # Has units digit
            return str(int(match.group(1)) * 10 + int(match.group(2)))
        else:  # Just tens
            return str(int(match.group(1)) * 10)

    result = re.sub(r"(\d)10(\d)?", combine_chinese_compound, result)

    return result.strip()


# =============================================================================
# Date Format Normalization
# =============================================================================

# English month names
MONTH_NAMES = {
    "january": "1",
    "jan": "1",
    "february": "2",
    "feb": "2",
    "march": "3",
    "mar": "3",
    "april": "4",
    "apr": "4",
    "may": "5",
    "june": "6",
    "jun": "6",
    "july": "7",
    "jul": "7",
    "august": "8",
    "aug": "8",
    "september": "9",
    "sep": "9",
    "sept": "9",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}

# Chinese month names
CHINESE_MONTHS = {
    "一月": "1",
    "二月": "2",
    "三月": "3",
    "四月": "4",
    "五月": "5",
    "六月": "6",
    "七月": "7",
    "八月": "8",
    "九月": "9",
    "十月": "10",
    "十一月": "11",
    "十二月": "12",
}


def normalize_date(text: str) -> str:
    """Normalize date expressions to a canonical format.

    Converts various date formats to "M/D" or "YYYY/M/D":
    - "March 15" -> "3/15"
    - "3月15日" -> "3/15"
    - "15th of March" -> "3/15"
    - "2024-03-15" -> "2024/3/15"

    Args:
        text: Input text

    Returns:
        Text with normalized dates
    """
    result = text.lower()

    # Handle ISO format: 2024-03-15
    def iso_to_canonical(match: re.Match[str]) -> str:
        year = match.group(1)
        month = str(int(match.group(2)))
        day = str(int(match.group(3)))
        return f"{year}/{month}/{day}"

    result = re.sub(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", iso_to_canonical, result)

    # Handle "Month Day" format: March 15, March 15th
    def month_day_to_canonical(match: re.Match[str]) -> str:
        month_name = match.group(1).lower()
        day = match.group(2).rstrip("stndrdth")
        month = MONTH_NAMES.get(month_name, month_name)
        return f"{month}/{day}"

    month_pattern = "|".join(MONTH_NAMES.keys())
    result = re.sub(
        rf"\b({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", month_day_to_canonical, result
    )

    # Handle "Day Month" format: 15 March, 15th of March
    def day_month_to_canonical(match: re.Match[str]) -> str:
        day = match.group(1).rstrip("stndrdth")
        month_name = match.group(2).lower()
        month = MONTH_NAMES.get(month_name, month_name)
        return f"{month}/{day}"

    result = re.sub(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_pattern})\b",
        day_month_to_canonical,
        result,
    )

    # Handle Chinese date format: 3月15日
    def chinese_date_to_canonical(match: re.Match[str]) -> str:
        month = match.group(1)
        day = match.group(2)
        return f"{month}/{day}"

    result = re.sub(r"(\d{1,2})月(\d{1,2})日?", chinese_date_to_canonical, result)

    # Handle Chinese month names
    for chinese, num in CHINESE_MONTHS.items():
        result = result.replace(chinese, f"{num}月")

    return result.strip()


# =============================================================================
# Enhanced Matching Functions
# =============================================================================


def get_canonical_form(text: str) -> str:
    """Get canonical form of text using synonym lookup.

    Args:
        text: Input text

    Returns:
        Canonical form if found, otherwise original text
    """
    text_lower = text.lower().strip()
    return _SYNONYM_LOOKUP.get(text_lower, text_lower)


def expand_synonyms(text: str) -> List[str]:
    """Expand text to include all synonym variants.

    Args:
        text: Input text

    Returns:
        List of text variants including synonyms
    """
    text_lower = text.lower().strip()
    variants = [text_lower]

    # Check if text matches any canonical form or variant
    canonical = _SYNONYM_LOOKUP.get(text_lower)
    if canonical:
        variants.append(canonical)
        if canonical in SYNONYMS:
            variants.extend(v.lower() for v in SYNONYMS[canonical])

    # Also check if text is a canonical form
    if text_lower in SYNONYMS:
        variants.extend(v.lower() for v in SYNONYMS[text_lower])

    return list(set(variants))


def extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text.

    Removes stopwords and returns content words.

    Args:
        text: Input text

    Returns:
        List of keywords
    """
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        "dare",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "and",
        "but",
        "if",
        "or",
        "because",
        "until",
        "while",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        # Chinese stopwords
        "的",
        "是",
        "在",
        "了",
        "和",
        "与",
        "或",
        "但",
        "而",
        "也",
        "就",
        "都",
        "还",
        "有",
        "没",
        "不",
        "很",
        "这",
        "那",
        "它",
        "他",
        "她",
    }

    # Tokenize
    words = re.findall(r"\w+", text.lower())

    # Filter stopwords
    keywords = [w for w in words if w not in stopwords and len(w) > 1]

    return keywords


def fuzzy_match_score(predicted: str, expected: str, threshold: float = 0.6) -> float:
    """Compute fuzzy match score with tolerance.

    Uses SequenceMatcher for character-level similarity.

    Args:
        predicted: Predicted answer
        expected: Expected answer
        threshold: Minimum similarity for partial credit

    Returns:
        Similarity score in [0.0, 1.0]
    """
    from difflib import SequenceMatcher

    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if not pred_norm or not exp_norm:
        return 0.0

    ratio = SequenceMatcher(None, pred_norm, exp_norm).ratio()
    return ratio if ratio >= threshold else ratio * 0.5  # Give partial credit even below threshold


def contains_score(predicted: str, expected: str) -> float:
    """Check if expected answer is contained in predicted.

    Also checks for synonym matches.

    Args:
        predicted: Predicted answer
        expected: Expected answer

    Returns:
        Score in [0.0, 1.0]
    """
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if not pred_norm or not exp_norm:
        return 0.0

    # Direct containment
    if exp_norm in pred_norm:
        return 1.0

    # Check synonym containment
    exp_variants = expand_synonyms(exp_norm)
    for variant in exp_variants:
        if variant in pred_norm:
            return 0.95

    # Check if any significant keywords are contained
    exp_keywords = extract_keywords(exp_norm)
    if exp_keywords:
        matches = sum(1 for kw in exp_keywords if kw in pred_norm)
        keyword_score = matches / len(exp_keywords)
        if keyword_score > 0.5:
            return keyword_score * 0.8

    return 0.0


def semantic_similarity_score(
    predicted: str,
    expected: str,
    embedder: Optional[Callable[[str], np.ndarray]] = None,
) -> Optional[float]:
    """Compute semantic similarity using embeddings.

    Args:
        predicted: Predicted answer
        expected: Expected answer
        embedder: Embedding function

    Returns:
        Cosine similarity score in [0.0, 1.0], or None if embedder unavailable
    """
    if embedder is None:
        return None

    try:
        pred_emb = embedder(predicted)
        exp_emb = embedder(expected)

        # Ensure 1D vectors
        if pred_emb.ndim > 1:
            pred_emb = pred_emb.flatten()
        if exp_emb.ndim > 1:
            exp_emb = exp_emb.flatten()

        # Cosine similarity
        dot_product = np.dot(pred_emb, exp_emb)
        norm_pred = np.linalg.norm(pred_emb)
        norm_exp = np.linalg.norm(exp_emb)

        if norm_pred > 0 and norm_exp > 0:
            similarity = dot_product / (norm_pred * norm_exp)
            # Normalize from [-1, 1] to [0, 1]
            return float((similarity + 1) / 2)
    except Exception as e:
        logger.debug(f"Semantic similarity computation failed: {e}")

    return None


def exact_match_score(
    predicted: str,
    expected: str,
    embedder: Optional[Callable[[str], np.ndarray]] = None,
) -> Tuple[bool, float]:
    """Compute exact match score with enhanced tolerance.

    Uses multiple strategies for tolerant matching:
    1. Exact match after normalization
    2. Synonym matching
    3. Number format normalization (28 = 28岁 = twenty-eight)
    4. Date format normalization (March 15 = 3/15 = 3月15日)
    5. Substring containment
    6. Keyword overlap
    7. Fuzzy character matching
    8. Semantic similarity (if embedder available)

    Args:
        predicted: Predicted answer
        expected: Expected answer
        embedder: Optional embedding function for semantic similarity

    Returns:
        Tuple of (is_correct, score)
    """
    if not predicted or not expected:
        return False, 0.0

    # Basic normalization
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    # 1. Exact match
    if pred_norm == exp_norm:
        return True, 1.0

    # 2. Synonym matching
    pred_canonical = get_canonical_form(pred_norm)
    exp_canonical = get_canonical_form(exp_norm)

    if pred_canonical == exp_canonical:
        return True, 1.0

    # Check all variants
    pred_variants = expand_synonyms(pred_norm)
    exp_variants = expand_synonyms(exp_norm)

    if set(pred_variants) & set(exp_variants):
        return True, 0.98

    # 3. Number format normalization
    pred_num_norm = normalize_number(pred_norm)
    exp_num_norm = normalize_number(exp_norm)

    if pred_num_norm == exp_num_norm:
        return True, 1.0

    # Check if normalized numbers appear in the text
    pred_numbers = set(re.findall(r"\d+", pred_num_norm))
    exp_numbers = set(re.findall(r"\d+", exp_num_norm))

    if exp_numbers and exp_numbers <= pred_numbers:
        # Expected numbers are subset of predicted numbers
        return True, 0.95

    # 4. Date format normalization
    pred_date_norm = normalize_date(pred_norm)
    exp_date_norm = normalize_date(exp_norm)

    if pred_date_norm == exp_date_norm:
        return True, 1.0

    # Check date containment
    if exp_date_norm in pred_date_norm:
        return True, 0.95

    # 5. Substring containment (with both directions)
    containment = contains_score(pred_norm, exp_norm)
    if containment >= 0.8:
        return True, containment

    # 6. Word overlap with keyword weighting
    pred_words = set(pred_norm.split())
    exp_words = set(exp_norm.split())

    if pred_words and exp_words:
        overlap = pred_words & exp_words

        # Give extra weight to keywords
        pred_keywords = set(extract_keywords(pred_norm))
        exp_keywords = set(extract_keywords(exp_norm))

        keyword_overlap = pred_keywords & exp_keywords

        if exp_keywords:
            keyword_score = len(keyword_overlap) / len(exp_keywords)
            if keyword_score >= 0.8:
                return True, 0.85 + (keyword_score - 0.8)
            elif keyword_score >= 0.5:
                return True, 0.7 + keyword_score * 0.3

        # Standard word overlap
        if exp_words:
            overlap_score = len(overlap) / len(exp_words)
            if overlap_score >= 0.7:
                return True, overlap_score
            elif overlap_score >= 0.4:
                return False, overlap_score * 0.8

    # 7. Fuzzy matching
    fuzzy_score = fuzzy_match_score(pred_norm, exp_norm, threshold=0.5)
    if fuzzy_score >= 0.8:
        return True, fuzzy_score
    elif fuzzy_score >= 0.6:
        return False, fuzzy_score * 0.9

    # 8. Semantic similarity (if embedder available)
    if embedder is not None:
        semantic_score = semantic_similarity_score(predicted, expected, embedder)
        if semantic_score is not None:
            if semantic_score >= 0.9:
                return True, semantic_score
            elif semantic_score >= 0.8:
                return True, semantic_score * 0.95
            elif semantic_score >= 0.7:
                return False, semantic_score * 0.8
            # Blend with fuzzy score
            blended = max(fuzzy_score, semantic_score * 0.7)
            if blended > 0.5:
                return False, blended

    # Return best partial score
    best_partial = max(fuzzy_score, containment, 0.0)
    if best_partial > 0.3:
        return False, best_partial

    return False, 0.0


def capability_to_task_type(capability: BenchmarkCapability) -> str:
    """Convert BenchmarkCapability to task_type string."""
    mapping = {
        BenchmarkCapability.ACCURATE_RETRIEVAL: TASK_TYPE_AR,
        BenchmarkCapability.TEST_TIME_LEARNING: TASK_TYPE_TTL,
        BenchmarkCapability.LONG_RANGE_UNDERSTANDING: TASK_TYPE_LRU,
        BenchmarkCapability.CONFLICT_RESOLUTION: TASK_TYPE_CR,
    }
    return mapping.get(capability, TASK_TYPE_AR)


def task_type_to_capability(task_type: str) -> BenchmarkCapability:
    """Convert task_type string to BenchmarkCapability."""
    return CAPABILITY_MAPPING.get(task_type.lower(), BenchmarkCapability.ACCURATE_RETRIEVAL)


# =============================================================================
# MemoryAgentBench Adapter
# =============================================================================


class MemoryAgentBenchAdapter(BenchmarkAdapter):
    """MemoryAgentBench (2025.07) adapter for AURORA evaluation.

    Evaluates four core memory capabilities:
    1. Accurate Retrieval (AR) - Extracting precise information
    2. Test-Time Learning (TTL) - Applying new rules without retraining
    3. Long-Range Understanding (LRU) - Coherent summarization
    4. Conflict Resolution (CR) - Handling contradictory updates

    Attributes:
        llm: LLM provider for evaluation (optional, for LLM-as-Judge)
        embedder: Embedding function (optional, uses memory's embedder if not provided)
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        embedder: Optional[Callable[[str], np.ndarray]] = None,
        seed: int = 0,
    ):
        """Initialize the adapter.

        Args:
            llm_provider: LLM provider implementing complete_json() for judging
            embedder: Embedding function for similarity computation
            seed: Random seed for reproducibility
        """
        super().__init__(llm_provider=llm_provider, seed=seed)
        self.embedder = embedder
        self._instances_cache: Dict[str, List[BenchmarkInstance]] = {}
        self._config: Optional[EvaluationConfig] = None

    @property
    def name(self) -> str:
        """Benchmark name."""
        return "MemoryAgentBench"

    @property
    def capabilities(self) -> List[BenchmarkCapability]:
        """Supported capabilities."""
        return [
            BenchmarkCapability.ACCURATE_RETRIEVAL,
            BenchmarkCapability.TEST_TIME_LEARNING,
            BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
            BenchmarkCapability.CONFLICT_RESOLUTION,
        ]

    # -------------------------------------------------------------------------
    # Dataset Loading
    # -------------------------------------------------------------------------

    def load_dataset(
        self,
        path: str,
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[BenchmarkInstance]:
        """Load MemoryAgentBench dataset.

        Supports:
        - HuggingFace: "ai-hyz/MemoryAgentBench"
        - Local JSON: "/path/to/dataset.json"
        - Local directory: "/path/to/dataset/" (with split files)

        Args:
            path: Dataset source (HuggingFace ID or local path)
            subset: Dataset split (default: "test")
            limit: Maximum number of instances to load

        Returns:
            List of BenchmarkInstance objects
        """
        split = subset or "test"
        cache_key = f"{path}:{split}"

        # Check cache
        if cache_key in self._instances_cache:
            cached_instances = self._instances_cache[cache_key]
            if limit:
                return cached_instances[:limit]
            return cached_instances

        instances: List[BenchmarkInstance] = []

        # Try HuggingFace first
        if not path.startswith("/") and not path.startswith("./"):
            instances = self._load_from_huggingface(path, split)

        # Fall back to local file
        if not instances:
            instances = self._load_from_local(path, split)

        # Cache results
        if instances:
            self._instances_cache[cache_key] = instances

        logger.info(f"Loaded {len(instances)} instances from {path} ({split})")

        if limit:
            return instances[:limit]
        return instances

    def _load_from_huggingface(
        self,
        dataset_id: str,
        split: str,
    ) -> List[BenchmarkInstance]:
        """Load dataset from HuggingFace Hub.

        Args:
            dataset_id: HuggingFace dataset identifier
            split: Dataset split

        Returns:
            List of BenchmarkInstance objects
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning("HuggingFace datasets not installed. Install with: pip install datasets")
            return []

        try:
            dataset = load_dataset(dataset_id, split=split)
            instances = []

            for idx, item in enumerate(dataset):
                instance = self._parse_hf_item(item, idx)
                if instance:
                    instances.append(instance)

            return instances

        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")
            return []

    def _load_from_local(
        self,
        path: str,
        split: str,
    ) -> List[BenchmarkInstance]:
        """Load dataset from local file or directory.

        Args:
            path: Local path to JSON file or directory
            split: Dataset split (used for directory structure)

        Returns:
            List of BenchmarkInstance objects
        """
        path_obj = Path(path)
        instances: List[BenchmarkInstance] = []

        if path_obj.is_file():
            # Single JSON file
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else data.get("data", [])
            for idx, item in enumerate(items):
                instance = self._parse_local_item(item, idx)
                if instance:
                    instances.append(instance)

        elif path_obj.is_dir():
            # Directory with split files
            split_file = path_obj / f"{split}.json"
            if split_file.exists():
                return self._load_from_local(str(split_file), split)

            # Try loading all JSON files
            for json_file in path_obj.glob("*.json"):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                items = data if isinstance(data, list) else data.get("data", [])
                for idx, item in enumerate(items):
                    instance = self._parse_local_item(item, len(instances) + idx)
                    if instance:
                        instances.append(instance)

        return instances

    def _parse_hf_item(
        self,
        item: Dict[str, Any],
        idx: int,
    ) -> Optional[BenchmarkInstance]:
        """Parse HuggingFace dataset item into BenchmarkInstance.

        Expected fields (flexible):
        - capability/type: Capability type string
        - context/conversation/history: Conversation context
        - question/query: The question to answer
        - answer/expected_answer/ground_truth: Expected answer

        Args:
            item: Raw dataset item
            idx: Item index for ID generation

        Returns:
            BenchmarkInstance or None if parsing fails
        """
        try:
            # Extract capability
            cap_str = item.get("capability") or item.get("type") or "ar"
            capability = CAPABILITY_MAPPING.get(
                cap_str.lower(),
                BenchmarkCapability.ACCURATE_RETRIEVAL,
            )
            task_type = capability_to_task_type(capability)

            # Extract context
            context = item.get("context") or item.get("conversation") or item.get("history") or ""

            # Extract question
            question = item.get("question") or item.get("query") or item.get("input") or ""

            # Extract expected answer
            expected = (
                item.get("answer")
                or item.get("expected_answer")
                or item.get("ground_truth")
                or item.get("output")
                or ""
            )

            if not question or not expected:
                return None

            # Parse turns from context
            turns = parse_conversation_turns(context) if context else []

            # Extract conflicting facts for CR
            metadata = item.get("metadata", {})
            conflicting_facts = extract_conflicting_facts(context, metadata)

            # Add capability info to metadata
            metadata["capability"] = capability.value
            metadata["conflicting_facts"] = conflicting_facts
            metadata["raw_item"] = item

            return BenchmarkInstance(
                id=item.get("id", f"mab_{idx}"),
                query=question,
                capability=capability,
                context=context,
                expected_answer=expected,
                task_type=task_type,
                conversation_history=turns,
                ground_truth=expected,
                metadata=metadata,
                reasoning_type=capability.value,
            )

        except Exception as e:
            logger.warning(f"Failed to parse item {idx}: {e}")
            return None

    def _parse_local_item(
        self,
        item: Dict[str, Any],
        idx: int,
    ) -> Optional[BenchmarkInstance]:
        """Parse local JSON item (same logic as HuggingFace).

        Args:
            item: Raw JSON item
            idx: Item index

        Returns:
            BenchmarkInstance or None
        """
        return self._parse_hf_item(item, idx)

    # -------------------------------------------------------------------------
    # Memory Preparation
    # -------------------------------------------------------------------------

    def _prepare_memory_for_instance(
        self,
        instance: BenchmarkInstance,
        memory: Any,
        clear_first: bool = True,
    ) -> None:
        """Prepare memory state by ingesting conversation history.

        Args:
            instance: Benchmark instance with conversation history
            memory: AURORA memory instance (direct engine or AuroraRuntime)
            clear_first: Whether to clear memory before ingestion
        """
        # Clear previous instance's memory to prevent pollution
        if clear_first:
            if hasattr(memory, "clear"):
                memory.clear()
            elif hasattr(memory, "plots"):
                # Fallback: manually clear core stores
                memory.plots.clear()
                memory.stories.clear()
                memory.themes.clear()
                if hasattr(memory, "graph"):
                    memory.graph.g.clear()
                if hasattr(memory, "vindex"):
                    memory.vindex.ids.clear()
                    memory.vindex.vecs.clear()
                    memory.vindex.kinds.clear()
                if hasattr(memory, "_relationship_story_index"):
                    memory._relationship_story_index.clear()
                if hasattr(memory, "_identity_dimensions"):
                    memory._identity_dimensions.clear()

        turns = instance.conversation_history

        # Check memory type and call appropriate method
        has_accept_interaction = hasattr(memory, "accept_interaction")
        has_ingest = hasattr(memory, "ingest")

        for i, turn in enumerate(turns):
            event_id = f"bench_{uuid.uuid4().hex[:8]}_{i}"
            content = turn.get("content") or turn.get("text", "")
            role = turn.get("role") or turn.get("speaker", "user")

            if has_accept_interaction:
                # AuroraRuntime style
                if role == "user":
                    # Look for next assistant response
                    assistant_response = ""
                    if i + 1 < len(turns):
                        next_turn = turns[i + 1]
                        next_role = next_turn.get("role") or next_turn.get("speaker", "")
                        if next_role == "assistant":
                            assistant_response = next_turn.get("content") or next_turn.get(
                                "text", ""
                            )

                    memory.accept_interaction(
                        event_id=event_id,
                        session_id="benchmark_session",
                        user_message=content,
                        agent_message=assistant_response,
                    )

            elif has_ingest:
                # Direct memory-engine style
                if role == "user":
                    assistant_response = ""
                    if i + 1 < len(turns):
                        next_turn = turns[i + 1]
                        next_role = next_turn.get("role") or next_turn.get("speaker", "")
                        if next_role == "assistant":
                            assistant_response = next_turn.get("content") or next_turn.get(
                                "text", ""
                            )

                    interaction_text = f"User: {content}"
                    if assistant_response:
                        interaction_text += f"\nAssistant: {assistant_response}"

                    memory.ingest(
                        interaction_text=interaction_text,
                        event_id=event_id,
                    )

        # Trigger evolution if available
        if hasattr(memory, "evolve"):
            memory.evolve()

    # -------------------------------------------------------------------------
    # Capability-Specific Evaluation
    # -------------------------------------------------------------------------

    def _evaluate_ar(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Accurate Retrieval capability.

        Tests the system's ability to extract precise information from
        extended interaction history.

        AURORA Implementation:
        - Uses query() with FieldRetriever
        - Evaluates retrieval precision

        Args:
            instance: Benchmark instance
            memory: AURORA memory instance

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        # Query memory
        query_result = self._query_memory(memory, instance.query)
        predicted_answer = self._extract_answer_from_retrieval(
            query_result,
            instance.query,
            memory,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Evaluate answer
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )

        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
            metadata={"query_result": query_result}
            if self._config and self._config.save_traces
            else {},
        )

    def _evaluate_ttl(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Test-Time Learning capability.

        Tests the system's ability to apply newly learned rules without
        parameter updates.

        AURORA Implementation:
        - ingest() learns rules from context
        - query() applies rules to answer questions
        - Evolution consolidates learned rules

        Args:
            instance: Benchmark instance
            memory: AURORA memory instance

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        # The context should contain rules to learn
        # These were already ingested in _prepare_memory
        # Now query to apply the rules

        query_result = self._query_memory(memory, instance.query)

        # For TTL, we may need to generate an answer using the rules
        predicted_answer = self._generate_ttl_answer(
            query_result,
            instance.query,
            memory,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Evaluate
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )

        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
        )

    def _evaluate_lru(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Long-Range Understanding capability.

        Tests the system's ability to form coherent summaries across
        extended narratives.

        AURORA Implementation:
        - Story aggregation organizes related plots
        - Theme emergence captures patterns
        - Materialized views and retrieved plots provide coherent summaries

        Args:
            instance: Benchmark instance
            memory: AURORA memory instance

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        # Query for relevant content
        query_result = self._query_memory(memory, instance.query, k=15)

        # Generate summary using narrative reconstruction
        predicted_answer = self._generate_lru_summary(
            query_result,
            instance.query,
            memory,
        )

        latency_ms = (time.time() - start_time) * 1000

        # LRU typically needs LLM-as-Judge for proper evaluation
        # since summaries can be semantically equivalent but worded differently
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )

        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
        )

    def _evaluate_cr(
        self,
        instance: BenchmarkInstance,
        memory: Any,
    ) -> BenchmarkResult:
        """Evaluate Conflict Resolution capability.

        Tests the system's ability to handle contradictory information
        and produce the most current/correct answer.

        AURORA Implementation:
        - Contradiction edges mark incompatible updates
        - Retrieval applies bounded inhibition to stale/conflicting paths
        - Resolution synthesis prefers newer and better-supported facts

        Args:
            instance: Benchmark instance
            memory: AURORA memory instance

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        # Get conflicting facts from metadata
        conflicting_facts = instance.metadata.get("conflicting_facts", [])

        # Query with awareness of conflicts
        query_result = self._query_memory(memory, instance.query)

        # Generate answer with conflict resolution
        predicted_answer = self._generate_cr_answer(
            query_result,
            instance.query,
            conflicting_facts,
            memory,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Evaluate
        is_correct, score = self._evaluate_answer(
            predicted_answer,
            instance.ground_truth,
        )

        return BenchmarkResult(
            instance_id=instance.id,
            capability=instance.capability,
            predicted=predicted_answer,
            expected=instance.expected_answer,
            score=score,
            latency_ms=latency_ms,
            task_type=instance.task_type,
            prediction=predicted_answer,
            ground_truth=instance.ground_truth,
            is_correct=is_correct,
            retrieval_count=len(query_result.get("ranked", [])),
        )

    # -------------------------------------------------------------------------
    # Main Evaluation Entry Point
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        instance: BenchmarkInstance,
        memory: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Evaluate a single benchmark instance.

        Routes to capability-specific evaluation methods.

        Args:
            instance: The benchmark instance to evaluate
            memory: AURORA memory instance
            **kwargs: Additional options (e.g., config)

        Returns:
            BenchmarkResult with evaluation metrics
        """
        # Store config if provided
        self._config = kwargs.get("config")

        # CRITICAL: Check for HashEmbedding and warn
        using_hash_embedding = self._check_hash_embedding_warning(memory)

        # Prepare memory with conversation history
        self._prepare_memory_for_instance(instance, memory)

        # Get capability from task_type or reasoning_type
        capability = task_type_to_capability(instance.reasoning_type or instance.task_type)

        # Route to capability-specific evaluation
        if capability == BenchmarkCapability.ACCURATE_RETRIEVAL:
            result = self._evaluate_ar(instance, memory)
        elif capability == BenchmarkCapability.TEST_TIME_LEARNING:
            result = self._evaluate_ttl(instance, memory)
        elif capability == BenchmarkCapability.LONG_RANGE_UNDERSTANDING:
            result = self._evaluate_lru(instance, memory)
        elif capability == BenchmarkCapability.CONFLICT_RESOLUTION:
            result = self._evaluate_cr(instance, memory)
        else:
            # Default to AR for unknown capabilities
            result = self._evaluate_ar(instance, memory)

        # Add HashEmbedding warning to result metadata if applicable
        if using_hash_embedding:
            if result.metadata is None:
                result.metadata = {}
            result.metadata["warning"] = "HASH_EMBEDDING_USED"
            result.metadata["embedding_warning"] = (
                "Results may be unreliable: HashEmbedding produces random vectors. "
                "Configure a real embedding provider for accurate benchmarks."
            )

        return result

    # -------------------------------------------------------------------------
    # Results Aggregation (Override for capability-specific metrics)
    # -------------------------------------------------------------------------

    def aggregate_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, float]:
        """Aggregate evaluation results into summary metrics.

        Overrides base method to add capability-specific metrics.

        Args:
            results: List of individual evaluation results

        Returns:
            Dict mapping metric names to values
        """
        # Use base implementation for basic metrics
        metrics = super().aggregate_results(results)

        # Add capability-specific metrics
        for cap in [
            BenchmarkCapability.ACCURATE_RETRIEVAL,
            BenchmarkCapability.TEST_TIME_LEARNING,
            BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
            BenchmarkCapability.CONFLICT_RESOLUTION,
        ]:
            task_type = capability_to_task_type(cap)
            cap_results = [
                r
                for r in results
                if r.task_type == task_type or (r.capability and r.capability == cap)
            ]

            if cap_results:
                cap_total = len(cap_results)
                cap_correct = sum(1 for r in cap_results if r.is_correct or r.score >= 0.5)
                cap_scores = [r.score for r in cap_results]

                # Use short capability names for keys
                cap_short = cap.value[:3]  # e.g., "acc", "tes", "lon", "con"
                metrics[f"accuracy_{cap_short}"] = cap_correct / cap_total if cap_total > 0 else 0.0
                metrics[f"count_{cap_short}"] = float(cap_total)
                metrics[f"avg_score_{cap_short}"] = (
                    float(np.mean(cap_scores)) if cap_scores else 0.0
                )

        # Store for get_evaluation_metrics()
        self._last_metrics = metrics
        self._last_results = results

        return metrics

    def get_evaluation_metrics(
        self, results: Optional[List[BenchmarkResult]] = None
    ) -> EvaluationMetrics:
        """Get EvaluationMetrics object from results.

        Args:
            results: List of results (uses last evaluated if None)

        Returns:
            EvaluationMetrics object with detailed breakdown
        """
        if results is None:
            results = getattr(self, "_last_results", [])

        if not results:
            return EvaluationMetrics()

        # Aggregate if not already done
        metrics_dict = self.aggregate_results(results)

        # Build metrics_by_type
        metrics_by_type: Dict[str, Dict[str, float]] = {}

        for cap in [
            BenchmarkCapability.ACCURATE_RETRIEVAL,
            BenchmarkCapability.TEST_TIME_LEARNING,
            BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
            BenchmarkCapability.CONFLICT_RESOLUTION,
        ]:
            task_type = capability_to_task_type(cap)
            cap_short = cap.value[:3]

            if f"accuracy_{cap_short}" in metrics_dict:
                metrics_by_type[task_type] = {
                    "accuracy": metrics_dict.get(f"accuracy_{cap_short}", 0.0),
                    "count": metrics_dict.get(f"count_{cap_short}", 0.0),
                    "avg_score": metrics_dict.get(f"avg_score_{cap_short}", 0.0),
                }

        return EvaluationMetrics(
            total_instances=int(metrics_dict.get("total_instances", 0)),
            correct_instances=int(metrics_dict.get("correct_instances", 0)),
            accuracy=metrics_dict.get("accuracy", 0.0),
            avg_score=metrics_dict.get("avg_score", 0.0),
            avg_latency_ms=metrics_dict.get("mean_latency_ms", 0.0),
            metrics_by_type=metrics_by_type,
            p50_latency_ms=metrics_dict.get("p50_latency_ms", 0.0),
            p99_latency_ms=metrics_dict.get("p99_latency_ms", 0.0),
        )

    # -------------------------------------------------------------------------
    # Convenience Method for Config-Based Benchmark Run
    # -------------------------------------------------------------------------

    def run_benchmark_with_config(
        self,
        memory: Any,
        source: str,
        config: Optional[EvaluationConfig] = None,
        split: str = "test",
    ) -> Tuple[List[BenchmarkResult], EvaluationMetrics]:
        """Run full benchmark evaluation with configuration.

        Args:
            memory: AURORA memory instance
            source: Dataset source
            config: Evaluation configuration
            split: Dataset split

        Returns:
            Tuple of (individual results, aggregate metrics)
        """
        config = config or EvaluationConfig()
        self._config = config

        # Load dataset
        instances = self.load_dataset(source, subset=split)

        # Apply filters
        if config.capabilities_filter:
            filter_types = [capability_to_task_type(c) for c in config.capabilities_filter]
            instances = [
                i
                for i in instances
                if i.task_type in filter_types
                or i.reasoning_type in [c.value for c in config.capabilities_filter]
            ]

        if config.max_instances:
            instances = instances[: config.max_instances]

        # Evaluate instances
        results: List[BenchmarkResult] = []

        for i, instance in enumerate(instances):
            if config.verbose:
                capability = task_type_to_capability(instance.reasoning_type or instance.task_type)
                print(f"[{i + 1}/{len(instances)}] Evaluating {instance.id} ({capability.value})")

            try:
                result = self.evaluate(instance, memory, config=config)
                results.append(result)

                if config.verbose:
                    status = "✓" if result.is_correct else "✗"
                    print(
                        f"  {status} Score: {result.score:.2f}, Latency: {result.latency_ms:.0f}ms"
                    )

            except Exception as e:
                error_result = BenchmarkResult(
                    instance_id=instance.id,
                    task_type=instance.task_type,
                    prediction="",
                    ground_truth=instance.ground_truth,
                    score=0.0,
                    is_correct=False,
                    error_message=str(e),
                )
                results.append(error_result)

                if config.verbose:
                    print(f"  ✗ Error: {e}")

        # Aggregate results and get EvaluationMetrics
        metrics = self.get_evaluation_metrics(results)

        return results, metrics

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _check_hash_embedding_warning(self, memory: Any) -> bool:
        """Check if memory is using HashEmbedding and log warning.

        Args:
            memory: AURORA memory instance

        Returns:
            True if using HashEmbedding (unreliable), False otherwise
        """
        # Check via capability method if available
        if hasattr(memory, "is_using_hash_embedding"):
            using_hash = bool(memory.is_using_hash_embedding())
        # Check embedder directly
        elif hasattr(memory, "embedder"):
            from aurora.integrations.embeddings.hash import HashEmbedding

            using_hash = isinstance(memory.embedder, HashEmbedding)
        # Check via mem attribute (AuroraRuntime)
        elif hasattr(memory, "mem") and hasattr(memory.mem, "embedder"):
            from aurora.integrations.embeddings.hash import HashEmbedding

            using_hash = isinstance(memory.mem.embedder, HashEmbedding)
        else:
            using_hash = False

        if using_hash:
            logger.warning(
                "⚠️ BENCHMARK WARNING: Memory is using HashEmbedding!\n"
                "  - HashEmbedding produces RANDOM vectors, not semantic embeddings.\n"
                "  - Benchmark results will be UNRELIABLE and near-random.\n"
                "  - Configure a real embedding provider for accurate evaluation:\n"
                "    export AURORA_BAILIAN_LLM_API_KEY='your-key' or\n"
                "    export AURORA_ARK_API_KEY='your-key'"
            )

        return bool(using_hash)

    def _query_memory(
        self,
        memory: Any,
        query_text: str,
        k: int = 8,
    ) -> Dict[str, Any]:
        """Query AURORA memory and return results.

        Handles both direct memory engines and AuroraRuntime.

        Args:
            memory: AURORA memory instance
            query_text: Query string
            k: Number of results to retrieve

        Returns:
            Dictionary with query results
        """
        result = {}

        if hasattr(memory, "query"):
            trace = memory.query(text=query_text, k=k)

            # Extract results based on trace type
            if hasattr(trace, "ranked"):
                result["ranked"] = trace.ranked
            if hasattr(trace, "attractor_path"):
                result["attractor_path"] = trace.attractor_path
            if hasattr(trace, "query_emb"):
                result["query_emb"] = trace.query_emb

            result["raw_trace"] = trace

        return result

    def _extract_answer_from_retrieval(
        self,
        query_result: Dict[str, Any],
        question: str,
        memory: Any,
    ) -> str:
        """Extract answer from retrieval results.

        For AR, we extract the most relevant content from retrieved plots,
        then use LLM (if available) to extract precise answer.

        Args:
            query_result: Query results from memory
            question: Original question
            memory: AURORA memory instance

        Returns:
            Extracted answer string
        """
        ranked = query_result.get("ranked", [])

        if not ranked:
            return "No relevant information found."

        # Get top results with scores
        context_texts: List[str] = []
        context_scores: List[float] = []

        for item in ranked[:5]:  # Use more context for better extraction
            # Handle different result formats
            if isinstance(item, tuple):
                node_id, score, kind = item
            else:
                node_id = item.get("id", "")
                score = item.get("score", 0.0)
                kind = item.get("kind", "plot")

            # Get content from memory
            content = self._get_node_content(memory, node_id, kind)
            if content:
                context_texts.append(content)
                context_scores.append(score)

        if not context_texts:
            return "No relevant information found."

        # Try LLM-based answer extraction
        if self.llm is not None:
            try:
                return self._llm_extract_answer(
                    question,
                    context_texts,
                    query_type=getattr(query_result.get("raw_trace"), "query_type", None),
                )
            except Exception as e:
                logger.debug(f"LLM extraction failed: {e}")

        # Fallback: heuristic-based extraction with scores
        return self._heuristic_extract_answer(question, context_texts, context_scores)

    def _llm_extract_answer(
        self,
        question: str,
        context: List[str],
        *,
        query_type: Any = None,
    ) -> str:
        """Use LLM to extract precise answer from context.

        Supports both English and Chinese questions/answers.
        Uses type-specific prompts for better accuracy.

        Args:
            question: The question to answer (English or Chinese)
            context: List of context strings

        Returns:
            Extracted answer
        """
        context_text = "\n\n".join(context[:5])

        # Reuse the main retrieval chain's query type instead of analyzing again.
        qtype = question_type_from_query_type(query_type)
        base_prompt = build_qa_prompt(
            question=question,
            context=context_text,
            question_type_hint=qtype,
            max_context_length=5000,
        )

        # Enhance with bilingual support for MemoryAgentBench
        prompt = f"""{base_prompt}

Note: 根据以下上下文，简洁地回答问题。只提供具体答案，除非必要，否则不需要完整句子。
For questions like:
- 地点问题 (在哪/哪里/住在): 返回地点名称
- 人物问题 (谁/叫什么/名字): 返回人名
- 时间问题 (什么时候/几点/哪天): 返回具体时间
- 数量问题 (多少/几个): 返回数字
- 偏好问题 (喜欢什么/最爱): 返回偏好内容"""

        system = """You are a precise answer extraction system that supports both English and Chinese.
Give only the direct answer, not explanations.
你是一个精确的答案提取系统，同时支持中英文。只给出直接答案，不需要解释。"""

        answer = str(
            self.llm.complete(
                prompt,
                system=system,
                temperature=0.0,
                max_tokens=100,
            )
        )

        return answer.strip()

    def _heuristic_extract_answer(
        self,
        question: str,
        context: List[str],
        scores: Optional[List[float]] = None,
    ) -> str:
        """Heuristic-based answer extraction without LLM.

        Enhanced implementation with:
        - Chinese keyword support for question type detection
        - Retrieval score weighting for candidate answers
        - Improved answer formatting (removing quotes, parentheses)
        - Yes/No question polarity analysis

        Args:
            question: The question to answer (English or Chinese)
            context: List of context strings
            scores: Optional retrieval scores for weighting (higher = more relevant)

        Returns:
            Extracted and formatted answer
        """
        question_lower = question.lower()
        full_context = " ".join(context)

        # Build weighted context if scores provided
        if scores and len(scores) == len(context):
            # Sort contexts by score (highest first)
            weighted_items = sorted(zip(context, scores), key=lambda x: x[1], reverse=True)
            weighted_context = [item[0] for item in weighted_items]
        else:
            weighted_context = context

        # =====================================================================
        # Location questions (中英文)
        # =====================================================================
        if any(
            kw in question_lower
            for kw in [
                "where",
                "city",
                "location",
                "place",
                "地点",
                "在哪",
                "住在",
                "位置",
                "城市",
                "哪里",
                "什么地方",
            ]
        ):
            # Extract subject from question (who we're asking about)
            question_subjects = self._extract_question_subjects(question)

            # Filter common non-locations
            non_locations = {
                "User",
                "Assistant",
                "I",
                "The",
                "My",
                "This",
                "That",
                "Yes",
                "No",
                "What",
                "Where",
                "When",
                "Who",
                "How",
                "Why",
                "Hello",
                "Hi",
                "Thanks",
            }

            # Build (location, source_context) pairs for better subject matching
            location_candidates: List[Tuple[str, str]] = []

            for ctx in weighted_context:
                # Extract English locations
                eng_locs = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", ctx)
                for loc in eng_locs:
                    if loc not in non_locations:
                        location_candidates.append((loc, ctx))

                # Extract Chinese locations
                chi_locs = re.findall(
                    r"(?:在|住在|去|来自|位于)\s*([^\s,，。！？]{2,10}(?:市|省|区|县|镇|村|路|街|国|洲))",
                    ctx,
                )
                for loc in chi_locs:
                    location_candidates.append((loc, ctx))

                # Extract locations from patterns like "lives in X"
                loc_pats = re.findall(
                    r"(?:live[sd]? in|located in|from|at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                    ctx,
                    re.IGNORECASE,
                )
                for loc in loc_pats:
                    if loc not in non_locations:
                        location_candidates.append((loc, ctx))

            if location_candidates:
                # If we have question subjects, prefer locations from contexts mentioning them
                if question_subjects:
                    for subj in question_subjects:
                        subj_lower = subj.lower()
                        for loc, ctx in location_candidates:
                            if subj in ctx or subj_lower in ctx.lower():
                                # Make sure we're not returning the subject as the location
                                if loc != subj and loc.lower() != subj_lower:
                                    return self._format_answer(loc)

                    # If subject found but no different location, try extracting from
                    # the context containing the subject
                    for subj in question_subjects:
                        subj_lower = subj.lower()
                        for ctx in weighted_context:
                            if subj in ctx or subj_lower in ctx.lower():
                                # Extract location patterns specifically
                                loc_extract = re.findall(
                                    r"(?:live[sd]? in|from|in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                                    ctx,
                                    re.IGNORECASE,
                                )
                                for loc in loc_extract:
                                    if loc not in non_locations and loc != subj:
                                        return self._format_answer(loc)

                # Otherwise prefer from highest-scored context (first in weighted list)
                return self._format_answer(location_candidates[0][0])

        # =====================================================================
        # Person questions (中英文)
        # =====================================================================
        if any(
            kw in question_lower
            for kw in ["who", "person", "name", "谁", "名字", "叫什么", "是谁", "哪个人", "什么人"]
        ):
            # Look for names in patterns like "I'm X" or "name is X"
            name_matches = re.findall(
                r"(?:I'm|I am|name is|called|名叫|我是|他是|她是|叫)\s*([A-Z][a-z]+|[\u4e00-\u9fa5]{2,4})",
                full_context,
            )

            # Also look for capitalized names after common patterns
            name_patterns = re.findall(r"(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s*([A-Z][a-z]+)", full_context)
            name_matches.extend(name_patterns)

            if name_matches:
                # Prefer from highest-scored context
                for ctx in weighted_context[:3]:
                    for name in name_matches:
                        if name in ctx:
                            return self._format_answer(name)
                return self._format_answer(name_matches[-1])

        # =====================================================================
        # Time questions (中英文)
        # =====================================================================
        if any(
            kw in question_lower
            for kw in [
                "when",
                "time",
                "date",
                "什么时候",
                "日期",
                "时间",
                "几点",
                "哪天",
                "何时",
                "多久",
            ]
        ):
            # Extract various time formats
            time_patterns = [
                # Full dates: 2024-01-15, 01/15/2024
                r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
                r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
                # Times: 10:30, 3:00 PM
                r"\b(\d{1,2}:\d{2}(?:\s*[AP]M)?)\b",
                # Relative time phrases
                r"(?:on|at|in)\s+(\d{1,2}[/:]\d{2}|\d{4}|\w+day|\w+ \d+)",
                # Named dates
                r"\b((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+\w+\s+\d+)?)\b",
                r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?)\b",
                # Chinese dates
                r"(\d{4}年\d{1,2}月\d{1,2}日)",
                r"(\d{1,2}月\d{1,2}日)",
                r"((?:上午|下午|晚上)?\d{1,2}[点时]\d{0,2}分?)",
                r"((?:星期|周)[一二三四五六日天])",
            ]

            all_times = []
            for pattern in time_patterns:
                matches = re.findall(pattern, full_context, re.IGNORECASE)
                all_times.extend(matches)

            if all_times:
                # Prefer from highest-scored context
                for ctx in weighted_context[:3]:
                    for t in all_times:
                        if t in ctx:
                            return self._format_answer(t)
                return self._format_answer(all_times[-1])

        # =====================================================================
        # Number/age questions (中英文)
        # =====================================================================
        if any(
            kw in question_lower
            for kw in [
                "how many",
                "how much",
                "age",
                "number",
                "count",
                "多少",
                "几个",
                "年龄",
                "多大",
                "几岁",
                "数量",
                "多长",
                "多远",
            ]
        ):
            # Extract numbers with optional units
            number_patterns = [
                # Numbers with units
                r"\b(\d+(?:\.\d+)?\s*(?:years?|months?|days?|hours?|minutes?|岁|年|月|天|个|人|次|件|元|块|米|公里|千米)?)\b",
                # Plain numbers (fallback)
                r"\b(\d+(?:\.\d+)?)\b",
            ]

            all_numbers = []
            for pattern in number_patterns:
                matches = re.findall(pattern, full_context, re.IGNORECASE)
                all_numbers.extend(matches)

            if all_numbers:
                # For age questions, look for specific patterns
                if any(kw in question_lower for kw in ["age", "old", "年龄", "多大", "几岁"]):
                    age_patterns = re.findall(
                        r"(\d+)\s*(?:years?\s*old|岁|yr)", full_context, re.IGNORECASE
                    )
                    if age_patterns:
                        return self._format_answer(age_patterns[-1])

                # Extract from highest-scored context
                for ctx in weighted_context[:3]:
                    for num in all_numbers:
                        if str(num) in ctx:
                            return self._format_answer(str(num).strip())

                # Return most recent number (often the answer)
                return self._format_answer(str(all_numbers[-1]).strip())

        # =====================================================================
        # Yes/No questions - analyze context polarity
        # =====================================================================
        is_yes_no_question = any(
            question_lower.startswith(kw)
            for kw in [
                "is ",
                "are ",
                "was ",
                "were ",
                "do ",
                "does ",
                "did ",
                "can ",
                "will ",
                "should ",
            ]
        ) or any(
            kw in question
            for kw in ["是否", "是不是", "有没有", "能不能", "会不会", "吗？", "吗?", "吗"]
        )

        if is_yes_no_question:
            # Analyze polarity in context
            positive_indicators = [
                "yes",
                "correct",
                "right",
                "true",
                "indeed",
                "definitely",
                "certainly",
                "是",
                "对",
                "正确",
                "确实",
                "没错",
                "当然",
                "肯定",
                "喜欢",
                "爱",
            ]
            negative_indicators = [
                "no",
                "not",
                "wrong",
                "false",
                "incorrect",
                "never",
                "don't",
                "doesn't",
                "didn't",
                "否",
                "不",
                "不是",
                "没有",
                "错",
                "不对",
                "没",
                "别",
                "不喜欢",
                "不爱",
            ]

            # Extract key concepts from question for matching
            # Remove question words to get the topic/action being asked about
            question_concepts = self._extract_question_concepts(question)

            # Look for context sentences that relate to question concepts
            for ctx in weighted_context[:3]:
                ctx_lower = ctx.lower()

                # Check if any question concept is in this context
                has_concept = any(
                    concept in ctx_lower or concept in ctx for concept in question_concepts
                )

                if has_concept:
                    # Count negative patterns around concept mentions
                    # Chinese negation patterns: 不X, 没X, 不喜欢, 没有
                    chi_neg_patterns = [
                        r"不喜欢",
                        r"不爱",
                        r"不想",
                        r"不会",
                        r"不能",
                        r"没有",
                        r"没想",
                        r"讨厌",
                        r"不吃",
                        r"不用",
                    ]
                    chi_pos_patterns = [
                        r"喜欢",
                        r"爱",
                        r"想",
                        r"会",
                        r"能",
                        r"有",
                        r"可以",
                        r"很棒",
                        r"很好",
                    ]

                    # Check for explicit negation patterns in Chinese
                    for neg_pat in chi_neg_patterns:
                        if re.search(neg_pat, ctx):
                            # Verify negation relates to question topic
                            for concept in question_concepts:
                                if concept in ctx or concept in ctx_lower:
                                    return "No"

                    # Check for positive patterns
                    for pos_pat in chi_pos_patterns:
                        if re.search(pos_pat, ctx):
                            # Make sure there's no negation before it
                            match = re.search(pos_pat, ctx)
                            if match:
                                prefix = ctx[max(0, match.start() - 2) : match.start()]
                                if not any(neg in prefix for neg in ["不", "没", "别"]):
                                    for concept in question_concepts:
                                        if concept in ctx or concept in ctx_lower:
                                            return "Yes"

                    # Check surrounding context for English negation
                    for concept in question_concepts:
                        if concept in ctx_lower:
                            idx = ctx_lower.find(concept)
                            surrounding = ctx_lower[
                                max(0, idx - 30) : min(len(ctx_lower), idx + 30)
                            ]

                            if any(
                                neg in surrounding
                                for neg in ["not", "no", "never", "don't", "doesn't"]
                            ):
                                return "No"
                            elif any(
                                pos in surrounding for pos in ["is", "are", "does", "will", "can"]
                            ):
                                return "Yes"

            # Count indicators in full context as fallback
            pos_count = sum(1 for ind in positive_indicators if ind in full_context.lower())
            neg_count = sum(1 for ind in negative_indicators if ind in full_context.lower())

            if neg_count > pos_count:
                return "No"
            elif pos_count > neg_count:
                return "Yes"

        # =====================================================================
        # Preference/favorite questions (中英文)
        # =====================================================================
        if any(
            kw in question_lower
            for kw in [
                "favorite",
                "prefer",
                "like",
                "love",
                "best",
                "喜欢",
                "最爱",
                "偏好",
                "喜爱",
                "爱好",
            ]
        ):
            # Look for preference patterns
            pref_patterns = [
                r"(?:favorite|prefer|like|love)\s+(\w+(?:\s+\w+)?)",
                r"(?:喜欢|最爱|爱好|偏好)\s*([\u4e00-\u9fa5]+|\w+)",
                r"(\w+)\s+(?:is my favorite|is the best)",
            ]

            for pattern in pref_patterns:
                matches = re.findall(pattern, full_context, re.IGNORECASE)
                if matches:
                    return self._format_answer(matches[-1])

        # =====================================================================
        # Default: extract from relevant context
        # =====================================================================
        # Define stop words for both languages
        stop_words = {
            "what",
            "who",
            "when",
            "where",
            "how",
            "why",
            "is",
            "are",
            "the",
            "a",
            "an",
            "do",
            "does",
            "did",
            "was",
            "were",
            "has",
            "have",
            "had",
            "will",
            "would",
            "什么",
            "谁",
            "哪",
            "怎么",
            "为什么",
            "是",
            "有",
            "的",
            "了",
            "吗",
            "呢",
        }
        question_words = set(question_lower.split()) - stop_words

        # Also extract Chinese tokens
        chinese_tokens = re.findall(r"[\u4e00-\u9fa5]+", question_lower)
        question_words.update(chinese_tokens)

        # Find sentence containing question keywords in highest-scored context
        for ctx in weighted_context:
            sentences = re.split(r"[.!?。！？]", ctx)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 5:
                    continue

                sentence_lower = sentence.lower()
                # Check for keyword overlap
                for word in question_words:
                    if len(word) > 1 and word in sentence_lower:
                        # Extract the core answer (after common answer prefixes)
                        answer_prefixes = [
                            r"^(?:the answer is|it is|i think|i believe|答案是|是)\s*[:\s]*",
                        ]
                        cleaned = sentence
                        for prefix in answer_prefixes:
                            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)

                        return self._format_answer(cleaned)

        # Last resort: return formatted first context
        return (
            self._format_answer(context[0][:200]) if context else "No relevant information found."
        )

    def _format_answer(self, answer: str) -> str:
        """Format and clean extracted answer.

        Removes unnecessary punctuation, quotes, parentheses,
        and extracts core answer content.

        Args:
            answer: Raw answer string

        Returns:
            Cleaned and formatted answer
        """
        if not answer:
            return ""

        # Remove leading/trailing whitespace
        result = answer.strip()

        # Remove surrounding quotes (single, double, Chinese quotes)
        quote_patterns = [
            (r'^["\'](.+)["\']$', r"\1"),
            (r'^[""](.+)[""]$', r"\1"),
            (r"^『(.+)』$", r"\1"),
            (r"^「(.+)」$", r"\1"),
        ]
        for pattern, replacement in quote_patterns:
            result = re.sub(pattern, replacement, result)

        # Remove surrounding parentheses/brackets
        bracket_patterns = [
            (r"^\((.+)\)$", r"\1"),
            (r"^\[(.+)\]$", r"\1"),
            (r"^（(.+)）$", r"\1"),
            (r"^【(.+)】$", r"\1"),
        ]
        for pattern, replacement in bracket_patterns:
            result = re.sub(pattern, replacement, result)

        # Remove common answer prefixes
        prefix_patterns = [
            r"^(?:the answer is|answer:|response:|i think|i believe)\s*[:\s]*",
            r"^(?:答案是|答案：|回答：|我认为|我觉得)\s*[:\s]*",
            r"^(?:based on the context,?|according to the text,?)\s*",
            r"^(?:根据上下文，?|根据文本，?)\s*",
        ]
        for pattern in prefix_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        # Remove trailing punctuation that shouldn't be part of the answer
        result = result.rstrip(".,;:!?。，；：！？")

        # If result is too long, try to extract just the core noun phrase
        if len(result) > 100:
            # Try to find a shorter, more direct answer
            # Look for patterns like "X is Y" or "the Y is X"
            direct_patterns = [
                r"(?:is|are|was|were)\s+([^,.!?]+)",
                r"([^,.!?]+)\s+(?:is|are|was|were)",
            ]
            for pattern in direct_patterns:
                match = re.search(pattern, result, re.IGNORECASE)
                if match and len(match.group(1).strip()) > 2:
                    short_answer = match.group(1).strip()
                    if len(short_answer) < len(result):
                        result = short_answer
                        break

        return result.strip()

    def _extract_question_subjects(self, question: str) -> List[str]:
        """Extract subjects (entities being asked about) from a question.

        Identifies names, pronouns, and key noun phrases that indicate
        who/what the question is about.

        Args:
            question: The question string

        Returns:
            List of subject strings found in the question
        """
        subjects: List[str] = []

        # Extract capitalized names (English)
        names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", question)
        # Filter out question words and common non-subject words
        non_subjects = {
            "What",
            "Where",
            "When",
            "Who",
            "How",
            "Why",
            "Which",
            "Does",
            "Did",
            "Do",
            "Is",
            "Are",
            "Was",
            "Were",
            "Can",
            "Will",
            "The",
            "This",
            "That",
            "These",
            "Those",
        }
        subjects.extend([name for name in names if name not in non_subjects])

        # Extract Chinese names (2-4 character sequences that might be names)
        # Common patterns: after 他/她/它 or before 的
        chi_name_patterns = [
            r"(?:他|她|它)叫?(\w{2,4})",
            r"(\w{2,4})(?:住在|喜欢|是)",
        ]
        for pattern in chi_name_patterns:
            matches = re.findall(pattern, question)
            subjects.extend(matches)

        # Also include pronouns mapped to generic markers
        pronouns = {
            "he": "he",
            "she": "she",
            "it": "it",
            "they": "they",
            "他": "他",
            "她": "她",
            "它": "它",
            "他们": "他们",
        }
        question_lower = question.lower()
        for pronoun in pronouns:
            if pronoun in question_lower or pronoun in question:
                subjects.append(pronoun)

        return subjects

    def _extract_question_concepts(self, question: str) -> List[str]:
        """Extract key concepts (verbs, objects) from a question.

        Identifies the action/state and object being asked about,
        useful for Yes/No question analysis.

        Args:
            question: The question string

        Returns:
            List of concept strings (verbs, objects, key phrases)
        """
        concepts: List[str] = []
        question_lower = question.lower()

        # Define stop words to filter out
        stop_words = {
            # English
            "is",
            "are",
            "was",
            "were",
            "do",
            "does",
            "did",
            "can",
            "will",
            "would",
            "should",
            "could",
            "the",
            "a",
            "an",
            "it",
            "this",
            "that",
            "there",
            "he",
            "she",
            "they",
            "his",
            "her",
            "their",
            "to",
            "for",
            "of",
            "in",
            "on",
            "at",
            "by",
            "with",
            "about",
            "if",
            "or",
            "and",
            "but",
            # Chinese
            "吗",
            "呢",
            "的",
            "了",
            "是",
            "有",
            "在",
            "和",
            "与",
            "或",
            "但",
            "这",
            "那",
            "它",
            "他",
            "她",
            "们",
        }

        # Extract English words (filter stop words)
        eng_words = re.findall(r"\b([a-z]+)\b", question_lower)
        for word in eng_words:
            if word not in stop_words and len(word) > 2:
                concepts.append(word)

        # Extract Chinese segments (2+ characters, filter stop words)
        chi_segments = re.findall(r"[\u4e00-\u9fa5]{2,}", question)
        for seg in chi_segments:
            if seg not in stop_words:
                concepts.append(seg)

        # Also extract common verb+object patterns
        verb_obj_patterns = [
            # English: "like X", "eat X", "have X"
            r"(?:like|love|eat|drink|have|want|need|use)\s+(\w+)",
            # Chinese: verb + object
            r"(?:喜欢|爱|吃|喝|有|要|需要|用)\s*([\u4e00-\u9fa5]+)",
        ]

        for pattern in verb_obj_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            concepts.extend(matches)

        return list(set(concepts))  # Deduplicate

    def _generate_ttl_answer(
        self,
        query_result: Dict[str, Any],
        question: str,
        memory: Any,
    ) -> str:
        """Generate answer for TTL by extracting and applying learned rules.

        TTL (Test-Time Learning) tests the system's ability to apply newly learned
        rules without parameter updates. This method:
        1. Extracts rules/constraints/preferences from retrieved context
        2. Applies those rules to answer the current question

        Args:
            query_result: Query results from memory
            question: Question to answer
            memory: AURORA memory

        Returns:
            Generated answer that respects learned rules
        """
        ranked = query_result.get("ranked", [])

        if not ranked:
            return "No relevant information found."

        # Collect context texts from retrieved results
        context_texts: List[str] = []
        for item in ranked[:8]:  # Use more context for rule extraction
            if isinstance(item, tuple):
                node_id, score, kind = item
            else:
                node_id = item.get("id", "")
                kind = item.get("kind", "plot")

            content = self._get_node_content(memory, node_id, kind)
            if content:
                context_texts.append(content)

        if not context_texts:
            return "No relevant information found."

        # Extract rules from context
        extracted_rules = self._extract_rules_from_context(context_texts)

        # If LLM is available, use it for rule extraction and application
        if self.llm is not None:
            try:
                return self._llm_apply_rules(question, context_texts, extracted_rules)
            except Exception as e:
                logger.debug(f"LLM rule application failed: {e}")

        # Fallback: heuristic-based rule application
        return self._heuristic_apply_rules(question, context_texts, extracted_rules)

    def _extract_rules_from_context(
        self,
        context_texts: List[str],
    ) -> List[Dict[str, Any]]:
        """Extract rules, constraints, and preferences from context.

        Identifies patterns indicating user-defined rules such as:
        - "always", "never", "must", "should"
        - "prefer", "preference", "like", "don't like"
        - "rule:", "constraint:", "requirement:"
        - Conditional rules: "if X then Y", "when X do Y"

        Args:
            context_texts: List of context strings to analyze

        Returns:
            List of extracted rules as dictionaries with:
            - type: "constraint" | "preference" | "conditional" | "directive"
            - content: The rule text
            - keywords: Matched keywords
            - priority: Numeric priority (higher = more important)
        """
        rules: List[Dict[str, Any]] = []
        full_context = " ".join(context_texts)

        # Rule indicator patterns with priorities
        rule_patterns = [
            # High priority: explicit rules/constraints
            (r"(?:rule|constraint|requirement)[:\s]+(.+?)(?:\.|$|\n)", "constraint", 10),
            (r"(?:must|always)\s+(.+?)(?:\.|$|\n)", "constraint", 9),
            (r"(?:never|don't|do not|cannot|must not)\s+(.+?)(?:\.|$|\n)", "constraint", 9),
            # Medium priority: strong preferences
            (r"(?:prefer|preference)[:\s]+(.+?)(?:\.|$|\n)", "preference", 7),
            (r"(?:should|ought to)\s+(.+?)(?:\.|$|\n)", "directive", 6),
            (r"(?:like|love|enjoy)\s+(.+?)(?:\.|$|\n)", "preference", 5),
            (r"(?:dislike|hate|avoid)\s+(.+?)(?:\.|$|\n)", "preference", 5),
            # Conditional rules
            (r"(?:if|when|whenever)\s+(.+?)\s*(?:then|,)\s*(.+?)(?:\.|$|\n)", "conditional", 8),
            # Instructions/directives
            (r"(?:please|kindly)\s+(.+?)(?:\.|$|\n)", "directive", 4),
            (r"(?:remember that|note that|keep in mind)\s+(.+?)(?:\.|$|\n)", "directive", 6),
            # Format/style preferences
            (r"(?:format|style)[:\s]+(.+?)(?:\.|$|\n)", "preference", 5),
            (r"(?:use|using)\s+(.+?)(?:\.|$|\n)", "directive", 4),
        ]

        for pattern, rule_type, priority in rule_patterns:
            matches = re.findall(pattern, full_context, re.IGNORECASE)
            for match in matches:
                # Handle tuple matches (for conditional patterns)
                if isinstance(match, tuple):
                    content = f"if {match[0]} then {match[1]}"
                else:
                    content = match.strip()

                if content and len(content) > 3:  # Filter out very short matches
                    rules.append(
                        {
                            "type": rule_type,
                            "content": content,
                            "priority": priority,
                            "pattern": pattern[:30],  # Store pattern for debugging
                        }
                    )

        # Also extract sentences containing rule keywords
        rule_keywords = [
            "always",
            "never",
            "must",
            "should",
            "prefer",
            "rule",
            "constraint",
            "requirement",
            "don't",
            "cannot",
            "only",
            "specifically",
            "exactly",
            "format",
            "style",
            "use",
            "avoid",
            "remember",
        ]

        for ctx in context_texts:
            sentences = re.split(r"[.!?]", ctx)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sentence_lower = sentence.lower()
                matched_keywords = [kw for kw in rule_keywords if kw in sentence_lower]

                if matched_keywords and sentence not in [r["content"] for r in rules]:
                    # Determine rule type from keywords
                    if any(kw in sentence_lower for kw in ["never", "don't", "cannot", "avoid"]):
                        rule_type = "constraint"
                        priority = 8
                    elif any(kw in sentence_lower for kw in ["always", "must", "only", "exactly"]):
                        rule_type = "constraint"
                        priority = 8
                    elif any(kw in sentence_lower for kw in ["prefer", "like"]):
                        rule_type = "preference"
                        priority = 5
                    else:
                        rule_type = "directive"
                        priority = 4

                    rules.append(
                        {
                            "type": rule_type,
                            "content": sentence,
                            "keywords": matched_keywords,
                            "priority": priority,
                        }
                    )

        # Sort by priority (highest first) and deduplicate
        rules.sort(key=lambda x: x["priority"], reverse=True)

        # Deduplicate similar rules
        unique_rules: List[Dict[str, Any]] = []
        seen_contents: set[str] = set()

        for rule in rules:
            content_normalized = normalize_answer(rule["content"])
            if content_normalized not in seen_contents:
                seen_contents.add(content_normalized)
                unique_rules.append(rule)

        return unique_rules[:10]  # Limit to top 10 rules

    def _llm_apply_rules(
        self,
        question: str,
        context_texts: List[str],
        extracted_rules: List[Dict[str, Any]],
    ) -> str:
        """Use LLM to extract rules and apply them to answer the question.

        Args:
            question: The question to answer
            context_texts: Context containing potential rules
            extracted_rules: Pre-extracted rules from heuristics

        Returns:
            Answer that respects the learned rules
        """
        context_text = "\n\n".join(context_texts[:5])

        # Format extracted rules for LLM
        rules_text = ""
        if extracted_rules:
            rules_text = "\n\nPre-identified rules and constraints:\n"
            for i, rule in enumerate(extracted_rules[:5], 1):
                rules_text += f"{i}. [{rule['type'].upper()}] {rule['content']}\n"

        prompt = f"""You are analyzing a conversation context to identify and apply user-defined rules.

Context:
{context_text}
{rules_text}

Task: Answer the following question by applying any relevant rules, constraints, or preferences learned from the context.

Instructions:
1. First, identify any rules, constraints, or preferences in the context that relate to the question
2. Apply those rules to formulate your answer
3. If the question asks about a rule or preference, state the rule directly
4. If the question requires applying a rule, show how the rule applies

Question: {question}

Answer (apply relevant rules):"""

        system = """You are a precise rule extraction and application system. 
Your job is to:
1. Identify rules, constraints, and preferences from context
2. Apply those rules to answer questions accurately
3. Give direct answers that respect user-defined rules

Rules can take forms like:
- Explicit rules: "rule:", "constraint:", "requirement:"  
- Directives: "always", "never", "must", "should"
- Preferences: "prefer", "like", "don't like"
- Conditionals: "if X then Y", "when X do Y"

Provide concise answers that demonstrate understanding of the learned rules."""

        answer = str(
            self.llm.complete(
                prompt,
                system=system,
                temperature=0.0,
                max_tokens=200,
            )
        )

        return answer.strip()

    def _heuristic_apply_rules(
        self,
        question: str,
        context_texts: List[str],
        extracted_rules: List[Dict[str, Any]],
    ) -> str:
        """Apply extracted rules heuristically without LLM.

        Uses pattern matching to find rules relevant to the question
        and formulate an answer.

        Args:
            question: The question to answer
            context_texts: Context containing rules
            extracted_rules: Pre-extracted rules

        Returns:
            Answer based on heuristic rule application
        """
        question_lower = question.lower()
        question_words = set(question_lower.split())

        # Remove common question words for matching
        stop_words = {
            "what",
            "who",
            "when",
            "where",
            "how",
            "why",
            "is",
            "are",
            "the",
            "a",
            "an",
            "my",
            "your",
            "do",
            "does",
            "should",
            "would",
        }
        question_keywords = question_words - stop_words

        # Score rules by relevance to question
        scored_rules: List[Tuple[Dict[str, Any], float]] = []

        for rule in extracted_rules:
            rule_content_lower = rule["content"].lower()
            rule_words = set(rule_content_lower.split())

            # Calculate overlap score
            overlap = len(question_keywords & rule_words)
            relevance_score = overlap / max(len(question_keywords), 1)

            # Boost score for high-priority rules
            priority_boost = rule.get("priority", 5) / 10.0
            final_score = relevance_score + priority_boost

            if relevance_score > 0 or rule.get("priority", 0) >= 8:
                scored_rules.append((rule, final_score))

        # Sort by score
        scored_rules.sort(key=lambda x: x[1], reverse=True)

        # Handle different question types

        # Question about preferences/rules
        if any(
            kw in question_lower
            for kw in ["prefer", "preference", "rule", "constraint", "like", "want"]
        ):
            if scored_rules:
                best_rule = scored_rules[0][0]
                return str(best_rule["content"])

            # Search context directly for preference statements
            for ctx in context_texts:
                ctx_lower = ctx.lower()
                if any(kw in ctx_lower for kw in ["prefer", "like", "want", "rule"]):
                    # Extract the preference statement
                    sentences = re.split(r"[.!?]", ctx)
                    for sentence in sentences:
                        if any(kw in sentence.lower() for kw in ["prefer", "like", "want", "rule"]):
                            return sentence.strip()

        # Yes/No questions about rules
        if question_lower.startswith(("should", "can", "do", "does", "is", "are", "will")):
            # Check if any constraint applies
            for rule, _ in scored_rules:
                if rule["type"] == "constraint":
                    rule_lower = rule["content"].lower()
                    # Check for negative constraints
                    if any(
                        neg in rule_lower for neg in ["never", "don't", "cannot", "avoid", "not"]
                    ):
                        # Check if question topic matches constraint
                        for keyword in question_keywords:
                            if keyword in rule_lower:
                                return "No, based on the rule: " + str(rule["content"])
                    else:
                        for keyword in question_keywords:
                            if keyword in rule_lower:
                                return "Yes, based on the rule: " + str(rule["content"])

        # "What" questions - might be asking about specific rules
        if question_lower.startswith("what"):
            if scored_rules:
                return str(scored_rules[0][0]["content"])

        # "How" questions - might need to apply conditional rules
        if question_lower.startswith("how"):
            for rule, _ in scored_rules:
                if rule["type"] == "conditional":
                    return str(rule["content"])

        # Default: return most relevant rule content or extract from context
        if scored_rules:
            return str(scored_rules[0][0]["content"])

        # Last resort: use basic answer extraction
        return self._heuristic_extract_answer(question, context_texts)

    def _generate_lru_summary(
        self,
        query_result: Dict[str, Any],
        question: str,
        memory: Any,
    ) -> str:
        """Generate summary for LRU using current graph-first memory views.

        Args:
            query_result: Query results
            question: Question/topic for summary
            memory: AURORA memory

        Returns:
            Generated summary
        """
        ranked = query_result.get("ranked", [])

        if not ranked:
            return "Insufficient information for summary."

        # Collect all relevant content
        contents: List[str] = []
        for item in ranked:
            if isinstance(item, tuple):
                node_id, score, kind = item
            else:
                node_id = item.get("id", "")
                kind = item.get("kind", "plot")

            content = self._get_node_content(memory, node_id, kind)
            if content:
                contents.append(content)

        if not contents:
            return "Insufficient information for summary."

        # Prefer higher-level story/theme summaries when available.
        summary_parts: List[str] = []
        for item in ranked:
            if isinstance(item, tuple):
                node_id, _score, kind = item
            else:
                node_id = item.get("id", "")
                kind = item.get("kind", "plot")
            content = self._get_node_content(memory, node_id, kind)
            if content and content not in summary_parts:
                summary_parts.append(content)

        if summary_parts:
            return "\n".join(summary_parts[:5])

        # Fallback: concatenate plot contents
        return "\n".join(contents[:5])

    def _generate_cr_answer(
        self,
        query_result: Dict[str, Any],
        question: str,
        conflicting_facts: List[str],
        memory: Any,
    ) -> str:
        """Generate answer for CR with conflict resolution.

        Enhanced implementation that:
        1. Detects conflicts between candidates (semantic similarity + text patterns)
        2. Resolves conflicts by preferring newer, more reliable information
        3. Tracks reasoning trace for debugging

        Uses graph contradiction scanning and resolution synthesis patterns.

        Args:
            query_result: Query results
            question: Question to answer
            conflicting_facts: Known conflicting facts
            memory: AURORA memory

        Returns:
            Answer with conflicts resolved
        """
        ranked = query_result.get("ranked", [])

        if not ranked:
            return "No relevant information found."

        # Collect candidates with full metadata
        candidates: List[Dict[str, Any]] = []

        for item in ranked:
            if isinstance(item, tuple):
                node_id, score, kind = item
            else:
                node_id = item.get("id", "")
                score = item.get("score", 0.0)
                kind = item.get("kind", "plot")

            content = self._get_node_content(memory, node_id, kind)

            # Get full metadata
            recency = 0.0
            embedding = None

            if kind == "plot" and hasattr(memory, "plots"):
                plot = memory.plots.get(node_id)
                if plot:
                    recency = plot.ts
                    embedding = getattr(plot, "embedding", None)

            if content:
                candidates.append(
                    {
                        "id": node_id,
                        "content": content,
                        "score": score,
                        "recency": recency,
                        "embedding": embedding,
                        "kind": kind,
                    }
                )

        if not candidates:
            return "No relevant information found."

        # Detect and resolve conflicts
        resolved_answer, reasoning_trace = self._detect_and_resolve_conflicts(
            candidates, question, conflicting_facts
        )

        # Log reasoning trace for debugging
        if reasoning_trace.get("conflicts_detected"):
            logger.debug(f"CR reasoning: {reasoning_trace}")

        return resolved_answer

    def _detect_and_resolve_conflicts(
        self,
        candidates: List[Dict[str, Any]],
        question: str,
        conflicting_facts: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """Detect conflicts between candidates and resolve them.

        Conflict detection uses:
        1. Text pattern matching (negation patterns)
        2. Semantic similarity (high sim = same topic, check for opposition)
        3. Known conflicting facts from metadata

        Resolution strategy:
        - Prefer newer information (higher timestamp)
        - Track resolution reasoning

        Args:
            candidates: List of candidate dicts with content, score, recency, embedding
            question: Original question
            conflicting_facts: Known conflicts from benchmark metadata

        Returns:
            (resolved_answer, reasoning_trace)
        """
        reasoning_trace: Dict[str, Any] = {
            "conflicts_detected": [],
            "resolution_strategy": None,
            "chosen_candidate": None,
            "reasoning": [],
        }

        if len(candidates) < 2:
            # No conflict possible with single candidate
            reasoning_trace["reasoning"].append("Only one candidate, no conflict detection needed")
            return str(candidates[0]["content"]), reasoning_trace

        # Negation patterns for contradiction detection
        negation_patterns = [
            # English
            ("is", "is not"),
            ("are", "are not"),
            ("was", "was not"),
            ("can", "cannot"),
            ("can", "can't"),
            ("will", "will not"),
            ("will", "won't"),
            ("should", "should not"),
            ("should", "shouldn't"),
            ("does", "does not"),
            ("does", "doesn't"),
            ("has", "has not"),
            ("has", "hasn't"),
            ("yes", "no"),
            ("true", "false"),
            # Chinese
            ("是", "不是"),
            ("能", "不能"),
            ("会", "不会"),
            ("应该", "不应该"),
            ("可以", "不可以"),
            ("有", "没有"),
            ("对", "错"),
        ]

        # Detect conflicts pairwise
        conflict_pairs: List[Tuple[int, int, str, float]] = []  # (i, j, reason, severity)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                cand_i = candidates[i]
                cand_j = candidates[j]

                conflict_reason, severity = self._check_candidate_conflict(
                    cand_i, cand_j, negation_patterns, conflicting_facts
                )

                if conflict_reason:
                    conflict_pairs.append((i, j, conflict_reason, severity))
                    reasoning_trace["conflicts_detected"].append(
                        {
                            "candidate_a": cand_i["id"],
                            "candidate_b": cand_j["id"],
                            "reason": conflict_reason,
                            "severity": severity,
                        }
                    )

        # If no conflicts detected, return highest scoring candidate
        if not conflict_pairs:
            reasoning_trace["reasoning"].append(
                "No conflicts detected, using highest scoring candidate"
            )
            # Sort by score, then recency
            candidates.sort(key=lambda x: (x["score"], x["recency"]), reverse=True)
            reasoning_trace["chosen_candidate"] = candidates[0]["id"]
            return candidates[0]["content"], reasoning_trace

        # Resolve conflicts: prefer newer information
        reasoning_trace["resolution_strategy"] = "prefer_recent_information"
        reasoning_trace["reasoning"].append(
            f"Detected {len(conflict_pairs)} conflict(s), resolving by preferring newer information"
        )

        # Find candidates involved in conflicts
        conflict_candidates_idx: set[int] = set()
        for i, j, _, _ in conflict_pairs:
            conflict_candidates_idx.add(i)
            conflict_candidates_idx.add(j)

        # Among conflicting candidates, prefer the most recent
        conflicting_candidates = [candidates[idx] for idx in conflict_candidates_idx]

        # Sort by recency (most recent first), then score
        conflicting_candidates.sort(key=lambda x: (x["recency"], x["score"]), reverse=True)

        chosen = conflicting_candidates[0]
        reasoning_trace["chosen_candidate"] = chosen["id"]
        reasoning_trace["reasoning"].append(
            f"Selected candidate {chosen['id']} (recency={chosen['recency']:.0f}, score={chosen['score']:.3f})"
        )

        # Build resolved answer with conflict awareness
        resolved_answer = str(chosen["content"])

        # If there were multiple conflicts, note that this is the most recent info
        if len(conflict_pairs) > 1:
            reasoning_trace["reasoning"].append(
                "Multiple conflicting facts found, using most recent information"
            )

        return resolved_answer, reasoning_trace

    def _check_candidate_conflict(
        self,
        cand_a: Dict[str, Any],
        cand_b: Dict[str, Any],
        negation_patterns: List[Tuple[str, str]],
        conflicting_facts: List[str],
    ) -> Tuple[Optional[str], float]:
        """Check if two candidates conflict.

        Args:
            cand_a: First candidate
            cand_b: Second candidate
            negation_patterns: List of (positive, negative) pattern pairs
            conflicting_facts: Known conflicts from benchmark

        Returns:
            (conflict_reason, severity) or (None, 0.0) if no conflict
        """
        text_a = cand_a["content"].lower()
        text_b = cand_b["content"].lower()
        emb_a = cand_a.get("embedding")
        emb_b = cand_b.get("embedding")

        # 1. Check text pattern contradictions
        for pos, neg in negation_patterns:
            # Check if one has positive and other has negative
            a_has_pos = pos in text_a and neg not in text_a
            a_has_neg = neg in text_a
            b_has_pos = pos in text_b and neg not in text_b
            b_has_neg = neg in text_b

            if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                return f"Text contradiction: '{pos}' vs '{neg}'", 0.8

        # 2. Check semantic similarity for same-topic detection
        if emb_a is not None and emb_b is not None:
            try:
                # Ensure embeddings are numpy arrays
                if not isinstance(emb_a, np.ndarray):
                    emb_a = np.array(emb_a)
                if not isinstance(emb_b, np.ndarray):
                    emb_b = np.array(emb_b)

                # Compute cosine similarity
                norm_a = np.linalg.norm(emb_a)
                norm_b = np.linalg.norm(emb_b)

                if norm_a > 0 and norm_b > 0:
                    sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)

                    # High similarity (same topic) + different content = potential conflict
                    if sim > 0.7:
                        # Same topic, check for content differences
                        content_diff = self._compute_content_difference(text_a, text_b)
                        if content_diff > 0.5:
                            return (
                                f"Semantic conflict: high similarity ({sim:.2f}) but different content",
                                0.6,
                            )

                    # Anti-correlation indicates opposition
                    if sim < -0.2:
                        return f"Semantic opposition: negative similarity ({sim:.2f})", 0.7
            except Exception as e:
                logger.debug(f"Embedding comparison failed: {e}")

        # 3. Check against known conflicting facts
        for conflict_fact in conflicting_facts:
            conflict_lower = conflict_fact.lower()
            if conflict_lower in text_a or conflict_lower in text_b:
                if conflict_lower in text_a and conflict_lower not in text_b:
                    return f"Known conflict with: {conflict_fact[:50]}", 0.9
                if conflict_lower in text_b and conflict_lower not in text_a:
                    return f"Known conflict with: {conflict_fact[:50]}", 0.9

        # 4. Check for value/quantity contradictions (numbers)
        numbers_a = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", text_a))
        numbers_b = set(re.findall(r"\b(\d+(?:\.\d+)?)\b", text_b))

        if numbers_a and numbers_b:
            # If both have numbers but they differ, check if they're answering same question
            common_words = set(text_a.split()) & set(text_b.split())
            if len(common_words) > 5 and numbers_a != numbers_b:
                # Same context but different numbers = possible conflict
                return f"Numeric conflict: {numbers_a} vs {numbers_b}", 0.5

        return None, 0.0

    def _compute_content_difference(self, text_a: str, text_b: str) -> float:
        """Compute content difference between two texts.

        Uses Jaccard distance on word sets.

        Returns:
            0.0 = identical, 1.0 = completely different
        """
        # Simple word-level Jaccard distance
        words_a = set(text_a.split())
        words_b = set(text_b.split())

        if not words_a and not words_b:
            return 0.0

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        if union == 0:
            return 0.0

        jaccard_sim = intersection / union
        return 1.0 - jaccard_sim

    def _get_node_content(
        self,
        memory: Any,
        node_id: str,
        kind: str,
    ) -> str:
        """Get content string from a memory node.

        Args:
            memory: AURORA memory instance
            node_id: Node ID
            kind: Node type ("plot", "story", "theme")

        Returns:
            Content string
        """
        if kind == "plot" and hasattr(memory, "plots"):
            plot = memory.plots.get(node_id)
            if plot:
                return str(plot.text)

        elif kind == "story" and hasattr(memory, "stories"):
            story = memory.stories.get(node_id)
            if story:
                plot_ids = list(getattr(story, "plot_ids", []))
                plot_texts = [
                    str(memory.plots[plot_id].text)
                    for plot_id in plot_ids
                    if hasattr(memory, "plots") and plot_id in memory.plots
                ]
                if plot_texts:
                    return " ".join(plot_texts[:3])
                return f"Story with {len(plot_ids)} events"

        elif kind == "theme" and hasattr(memory, "themes"):
            theme = memory.themes.get(node_id)
            if theme:
                label = (
                    getattr(theme, "label", "")
                    or getattr(theme, "name", "")
                    or getattr(theme, "description", "")
                )
                if label:
                    return str(label)

        return ""

    def _evaluate_answer(
        self,
        predicted: str,
        expected: str,
    ) -> Tuple[bool, float]:
        """Evaluate predicted answer against expected.

        Evaluation strategy (in priority order):
        1. LLM-as-Judge (most accurate, requires LLM)
        2. Enhanced exact_match_score with:
           - Synonym matching
           - Number/date normalization
           - Keyword overlap
           - Semantic similarity (if embedder available)

        Args:
            predicted: Predicted answer
            expected: Expected answer

        Returns:
            Tuple of (is_correct, score)
        """
        # Priority 1: LLM-as-Judge (most accurate semantic evaluation)
        if self._config and self._config.use_llm_judge and self.llm is not None:
            try:
                llm_result = self._llm_judge_evaluate(predicted, expected)
                logger.debug(f"LLM judge: {llm_result}")
                return llm_result
            except Exception as e:
                logger.warning(f"LLM judge failed: {e}, falling back to enhanced matching")

        # Priority 2: Enhanced exact match with semantic similarity
        # Pass embedder for semantic similarity scoring
        return exact_match_score(predicted, expected, embedder=self.embedder)

    def _llm_judge_evaluate(
        self,
        predicted: str,
        expected: str,
    ) -> Tuple[bool, float]:
        """Evaluate using LLM-as-Judge.

        Args:
            predicted: Predicted answer
            expected: Expected answer

        Returns:
            Tuple of (is_correct, score)
        """
        from pydantic import BaseModel

        class JudgeResult(BaseModel):
            is_correct: bool
            score: float
            reasoning: str

        user_prompt = MEMORYAGENTBENCH_JUDGE_USER_PROMPT.format(
            question="Does the predicted answer correctly match the expected answer?",
            expected_answer=expected,
            predicted_answer=predicted,
        )

        result = self.llm.complete_json(
            system=MEMORYAGENTBENCH_JUDGE_SYSTEM_PROMPT,
            user=user_prompt,
            schema=JudgeResult,
            temperature=0.0,
        )

        return result.is_correct, result.score


# =============================================================================
# Utility Functions
# =============================================================================


# =============================================================================
# CLI Support
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MemoryAgentBench evaluation")
    parser.add_argument(
        "--source",
        default="ai-hyz/MemoryAgentBench",
        help="Dataset source (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum instances to evaluate",
    )
    parser.add_argument(
        "--capability",
        choices=["ar", "ttl", "lru", "cr"],
        default=None,
        help="Filter by capability",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Create adapter
    adapter = MemoryAgentBenchAdapter()

    # Load dataset
    instances = adapter.load_dataset(args.source, subset=args.split)

    print(f"Loaded {len(instances)} instances")

    # Show capability distribution
    from collections import Counter

    cap_dist = Counter(i.reasoning_type for i in instances)
    print(f"Capability distribution: {dict(cap_dist)}")

    # Example: Print first instance
    if instances:
        print("\nFirst instance:")
        print(f"  ID: {instances[0].id}")
        print(f"  Task Type: {instances[0].task_type}")
        print(f"  Capability: {instances[0].reasoning_type}")
        print(f"  Query: {instances[0].query[:100]}...")
        print(f"  Expected: {instances[0].ground_truth[:100]}...")
