# AURORA LongMemEval Baseline Report

> [!WARNING]
> 归档说明：本文档为历史评测记录，仅保留作参考，不再代表当前生产实现。当前 canonical 入口请以 `README.md`、`docs/README.md`、`docs/adr/002-graph-first-emergence.md`、`docs/research/benchmark_guide.md` 和 `aurora.soul` / `aurora.runtime` 代码为准。

**Date:** 2026-02-01  
**Dataset:** LongMemEval Oracle (500 questions)  
**AURORA Version:** Current develop branch

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Questions** | 500 |
| **Correct Answers** | 240 |
| **Overall Accuracy** | 48.0% |
| **Errors** | 32 |
| **Evaluation Time** | 54.3 minutes |

## Accuracy by Question Type

| Question Type | Correct | Total | Accuracy |
|---------------|---------|-------|----------|
| single-session-user | 52 | 70 | **74.3%** |
| single-session-assistant | 41 | 56 | **73.2%** |
| knowledge-update | 51 | 78 | **65.4%** |
| temporal-reasoning | 55 | 133 | 41.4% |
| multi-session | 34 | 133 | 25.6% |
| single-session-preference | 7 | 30 | 23.3% |

## Performance Analysis

### High Performance (≥60%)

These question types demonstrate AURORA's core strengths:

1. **single-session-user (74.3%)**: User-stated information within a session
2. **single-session-assistant (73.2%)**: Information from assistant responses
3. **knowledge-update (65.4%)**: Tracking knowledge changes over time

### Medium Performance (40-60%)

1. **temporal-reasoning (41.4%)**: Questions about timing and sequence of events

### Low Performance (<40%)

Areas requiring improvement:

1. **multi-session (25.6%)**: Cross-session information aggregation
2. **single-session-preference (23.3%)**: User preferences and opinions

## Error Distribution

| Question Type | Errors |
|---------------|--------|
| multi-session | 22 |
| temporal-reasoning | 8 |
| knowledge-update | 2 |

Most errors occurred in multi-session questions, likely due to:
- API rate limiting
- Complex context requirements
- Long conversation histories

## Key Findings

1. **Strength in Single-Session Queries**: AURORA excels at retrieving information from recent, focused conversations (73-74% accuracy)

2. **Good Knowledge Update Tracking**: The system handles knowledge evolution well (65.4%), reflecting effective use of the `KnowledgeClassifier` and timeline mechanisms

3. **Multi-Session Challenge**: Cross-session context aggregation is the biggest weakness (25.6%), indicating need for better session boundary handling

4. **Preference Detection Gap**: User preference queries perform poorly (23.3%), suggesting the relational layer needs enhancement for opinion/preference tracking

5. **Temporal Reasoning Moderate**: Timeline-based queries show moderate performance (41.4%), indicating room for improvement in temporal context organization

## Comparison with Expected Benchmarks

| System | Accuracy | Notes |
|--------|----------|-------|
| AURORA (Baseline) | **48.0%** | Current implementation |
| Random Baseline | ~20% | Expected random chance |
| GPT-4 + Full Context | ~70% | Upper bound estimate |

## Recommendations for Improvement

### Priority 1: Multi-Session Enhancement
- Implement session-aware retrieval that can aggregate across conversation boundaries
- Add session-level summaries as retrieval targets
- Consider session clustering for related conversations

### Priority 2: Preference Detection
- Enhance `RelationalContext` to better capture user opinions and preferences
- Add explicit preference tracking in the ingestion pipeline
- Improve query type detection for preference-related queries

### Priority 3: Temporal Reasoning
- Strengthen `TimelineGroup` utilization in retrieval
- Add explicit temporal markers to plots during ingestion
- Improve temporal query parsing and routing

### Priority 4: Error Handling
- Add retry logic for API failures
- Implement batch processing with rate limiting
- Add timeout handling for complex queries

## Technical Details

### Configuration Used
```python
config = SoulConfig(dim=1024, max_plots=5000)
embedder = BailianEmbedding(model='text-embedding-v4', dimension=1024)
llm = ArkLLM(model='doubao-1-5-pro-32k-250115')
```

### Evaluation Method
- Fresh `AuroraSoul` instance per question
- Ingest all sessions, then query
- Top-5 retrieval for context building
- LLM-based answer generation
- Keyword-based matching for evaluation

## Files Generated

- `longmemeval_baseline.json` - Full results with detailed per-question data
- `run_longmemeval_baseline.py` - Evaluation script with checkpointing

## Next Steps

1. Analyze failure cases in multi-session questions
2. Implement session-aware retrieval enhancements
3. Re-run evaluation after improvements
4. Compare with MemoryAgentBench results
