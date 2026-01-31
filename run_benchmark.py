#!/usr/bin/env python3
"""Run MemoryAgentBench Full Benchmark"""
import sys
import os
import time
sys.path.insert(0, '.')

print('=' * 70)
print('MemoryAgentBench Full Benchmark')
print('=' * 70)
print()

from aurora.benchmark.interface import BenchmarkCapability, BenchmarkResult
from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.embeddings.bailian import BailianEmbedding
from aurora.llm.ark import ArkLLM
from datasets import load_dataset
from typing import List

# 创建真实的 Embedding 和 LLM
embedder = BailianEmbedding(
    api_key=os.getenv('AURORA_BAILIAN_API_KEY'),
    model='text-embedding-v4',
    dimension=1024
)

llm = ArkLLM(
    api_key=os.getenv('AURORA_ARK_API_KEY'),
    model=os.getenv('AURORA_ARK_LLM_MODEL', 'doubao-1-5-pro-32k-250115'),
    base_url=os.getenv('AURORA_ARK_BASE_URL', 'https://ark.cn-beijing.volces.com/api/v3')
)

print(f'Embedding: BailianEmbedding (text-embedding-v4)')
print(f'LLM: ArkLLM (doubao-1-5-pro-32k)')
print()

# 配置
config = MemoryConfig(dim=1024, max_plots=2000)

# 加载所有 splits
print('Loading dataset from HuggingFace...')
splits = ['Accurate_Retrieval', 'Test_Time_Learning', 'Long_Range_Understanding', 'Conflict_Resolution']

all_results: List[BenchmarkResult] = []
results_by_capability = {
    'ar': [],  
    'ttl': [],  
    'lru': [],  
    'cr': [],  
}

capability_map = {
    'Accurate_Retrieval': ('ar', BenchmarkCapability.ACCURATE_RETRIEVAL),
    'Test_Time_Learning': ('ttl', BenchmarkCapability.TEST_TIME_LEARNING),
    'Long_Range_Understanding': ('lru', BenchmarkCapability.LONG_RANGE_UNDERSTANDING),
    'Conflict_Resolution': ('cr', BenchmarkCapability.CONFLICT_RESOLUTION),
}

total_instances = 0
for split_name in splits:
    try:
        ds = load_dataset('ai-hyz/MemoryAgentBench', split=split_name)
        print(f"\n{'='*70}")
        print(f'Split: {split_name} ({len(ds)} instances)')
        print('='*70)
        
        cap_key, capability = capability_map[split_name]
        
        for idx, item in enumerate(ds):
            # Get context
            context = item.get('context', '')
            
            # Get questions and answers (lists)
            questions = item.get('questions', [])
            answers = item.get('answers', [])
            
            # Process each Q&A pair
            for q_idx, (question, expected) in enumerate(zip(questions, answers)):
                total_instances += 1
                
                # Create fresh memory for each test
                memory = AuroraMemory(cfg=config, seed=42 + idx + q_idx)
                memory.embedder = embedder  # Replace with real embedder
                
                print(f"\n[{total_instances}] {cap_key.upper()}")
                print(f'  Q: {question[:80]}...' if len(question) > 80 else f'  Q: {question}')
                print(f'  Expected: {expected[:60]}...' if len(expected) > 60 else f'  Expected: {expected}')
                
                start_time = time.time()
                
                try:
                    # Ingest context
                    if context:
                        turns = context.split('\n')
                        for turn in turns:
                            turn = turn.strip()
                            if turn:
                                memory.ingest(turn)
                        memory.evolve()
                    
                    # Query
                    trace = memory.query(question, k=8)
                    
                    # Extract answer from retrieval
                    retrieved_texts = []
                    for r in trace.ranked[:5]:
                        if hasattr(r, 'id'):
                            plot_id = r.id
                        else:
                            plot_id = r.get('id', '') if isinstance(r, dict) else str(r)
                        
                        if plot_id in memory.plots:
                            retrieved_texts.append(memory.plots[plot_id].text)
                    
                    context_text = '\n'.join(retrieved_texts)
                    
                    # Use LLM to extract answer if available
                    if llm and context_text:
                        prompt = f'''Based on the following context, answer the question precisely.

Context:
{context_text}

Question: {question}

Provide a concise and accurate answer based only on the context provided. If the information is not in the context, say "Information not found".'''
                        
                        try:
                            answer = llm.complete(prompt)
                            prediction = answer.strip() if answer else ''
                        except Exception as e:
                            prediction = context_text[:200] if context_text else ''
                    else:
                        prediction = context_text[:200] if context_text else ''
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Scoring
                    expected_lower = expected.lower().strip()
                    prediction_lower = prediction.lower().strip()
                    
                    if expected_lower in prediction_lower:
                        score = 1.0
                    elif any(word in prediction_lower for word in expected_lower.split() if len(word) > 3):
                        matched_words = sum(1 for word in expected_lower.split() if len(word) > 3 and word in prediction_lower)
                        total_words = sum(1 for word in expected_lower.split() if len(word) > 3)
                        score = matched_words / total_words if total_words > 0 else 0.0
                    else:
                        score = 0.0
                    
                    is_correct = score >= 0.5
                    
                    result = BenchmarkResult(
                        instance_id=f'{split_name}_{idx}_{q_idx}',
                        capability=capability,
                        task_type=cap_key,
                        predicted=prediction[:200],
                        expected=expected,
                        prediction=prediction[:200],
                        ground_truth=expected,
                        score=score,
                        is_correct=is_correct,
                        latency_ms=latency_ms,
                    )
                    
                    status = '✓' if is_correct else '✗'
                    print(f'  {status} Score: {score:.2f}, Latency: {latency_ms:.0f}ms')
                    if prediction:
                        print(f'  Pred: {prediction[:60]}...' if len(prediction) > 60 else f'  Pred: {prediction}')
                    
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    result = BenchmarkResult(
                        instance_id=f'{split_name}_{idx}_{q_idx}',
                        capability=capability,
                        task_type=cap_key,
                        predicted='',
                        expected=expected,
                        score=0.0,
                        is_correct=False,
                        latency_ms=latency_ms,
                        error_message=str(e),
                    )
                    print(f'  ✗ Error: {str(e)[:50]}')
                
                all_results.append(result)
                results_by_capability[cap_key].append(result)
            
    except Exception as e:
        print(f'Error loading {split_name}: {e}')

# Calculate metrics
print('\n' + '=' * 70)
print('MemoryAgentBench Results Summary')
print('=' * 70)

total_correct = sum(1 for r in all_results if r.is_correct)
overall_accuracy = total_correct / len(all_results) if all_results else 0

print(f'\nTotal Instances: {len(all_results)}')
print(f'Overall Accuracy: {overall_accuracy:.2%}')
print()

for cap_key, cap_results in results_by_capability.items():
    if cap_results:
        correct = sum(1 for r in cap_results if r.is_correct)
        accuracy = correct / len(cap_results)
        avg_score = sum(r.score for r in cap_results) / len(cap_results)
        avg_latency = sum(r.latency_ms for r in cap_results) / len(cap_results)
        
        cap_name = {
            'ar': 'Accurate Retrieval (AR)',
            'ttl': 'Test-Time Learning (TTL)',
            'lru': 'Long-Range Understanding (LRU)',
            'cr': 'Conflict Resolution (CR)',
        }[cap_key]
        
        print(f'{cap_name}:')
        print(f'  Instances: {len(cap_results)}')
        print(f'  Accuracy: {accuracy:.2%}')
        print(f'  Avg Score: {avg_score:.2f}')
        print(f'  Avg Latency: {avg_latency:.0f}ms')
        print()
