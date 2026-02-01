#!/usr/bin/env python3
"""
Full MemoryAgentBench Benchmark Runner
======================================

Runs the MemoryAgentBench evaluation against AURORA.

Dataset Structure:
- Accurate_Retrieval: 22 items × 100 questions = 2200 QA pairs
- Test_Time_Learning: 6 items × 200 questions = 1200 QA pairs
- Long_Range_Understanding: 110 items × 1 question = 110 QA pairs
- Conflict_Resolution: 8 items × 100 questions = 800 QA pairs
Total: ~4310 QA pairs

Usage:
    # Full evaluation (takes 1-2 hours)
    python run_full_mab_benchmark.py --full
    
    # Sampled evaluation (5-15 minutes, default)
    python run_full_mab_benchmark.py
    
    # Custom sample size per capability
    python run_full_mab_benchmark.py --samples 20
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Ensure local module is importable
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def load_mab_dataset(samples_per_cap: Optional[int] = None) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Load MemoryAgentBench dataset from HuggingFace.
    
    Args:
        samples_per_cap: If set, sample this many QA pairs per capability
        
    Returns:
        Tuple of (list of QA instances, stats dict)
    """
    from datasets import load_dataset
    
    print("Loading MemoryAgentBench from HuggingFace...")
    dataset = load_dataset("ai-hyz/MemoryAgentBench")
    
    # Map split names to capability codes
    splits = {
        "Accurate_Retrieval": "AR",
        "Test_Time_Learning": "TTL", 
        "Long_Range_Understanding": "LRU",
        "Conflict_Resolution": "CR",
    }
    
    all_instances = []
    stats = {"total": 0, "sampled": 0}
    
    for split_name, cap_code in splits.items():
        if split_name not in dataset:
            print(f"  Warning: Split '{split_name}' not found")
            continue
        
        split_data = dataset[split_name]
        split_instances = []
        
        for item_idx, item in enumerate(split_data):
            context = item.get("context", "")
            questions = item.get("questions", [])
            answers = item.get("answers", [])
            metadata = item.get("metadata", {})
            
            # Create one instance per question-answer pair
            for q_idx, (question, answer) in enumerate(zip(questions, answers)):
                # Handle answer that might be a list of variants
                if isinstance(answer, list):
                    # Take the first non-empty answer as ground truth
                    answer = answer[0] if answer else ""
                
                instance = {
                    "id": f"mab_{cap_code}_{item_idx:04d}_{q_idx:03d}",
                    "capability": cap_code,
                    "context": context,
                    "question": question,
                    "expected_answer": answer,
                    "item_idx": item_idx,
                    "q_idx": q_idx,
                }
                split_instances.append(instance)
        
        total_for_cap = len(split_instances)
        stats[f"{cap_code}_total"] = total_for_cap
        stats["total"] += total_for_cap
        
        # Sample if requested
        if samples_per_cap and samples_per_cap < total_for_cap:
            import random
            random.seed(42)  # Reproducible sampling
            split_instances = random.sample(split_instances, samples_per_cap)
            print(f"  {split_name}: {total_for_cap} total, sampled {len(split_instances)}")
        else:
            print(f"  {split_name}: {len(split_instances)} QA pairs")
        
        stats[f"{cap_code}_sampled"] = len(split_instances)
        stats["sampled"] += len(split_instances)
        all_instances.extend(split_instances)
    
    return all_instances, stats


def parse_context_to_turns(context: str, max_context_chars: int = 50000, max_turns: int = 50) -> list:
    """
    Parse context string into conversation turns.
    
    Handles various formats like "User: message\\nAssistant: response"
    
    Args:
        context: Full context string
        max_context_chars: Maximum characters to process (default 50k)
        max_turns: Maximum number of turns to keep
    """
    import re
    
    # Truncate very long contexts to a reasonable size
    # Try to keep the most recent content (end of context)
    if len(context) > max_context_chars:
        context = context[-max_context_chars:]
        # Try to start at a sentence boundary
        first_period = context.find('. ')
        if first_period > 0 and first_period < 1000:
            context = context[first_period + 2:]
    
    turns = []
    
    # Normalize markers
    normalized = context.replace("Human:", "User:")
    normalized = normalized.replace("AI:", "Assistant:")
    normalized = normalized.replace("Bot:", "Assistant:")
    
    # Split by role markers
    pattern = r"(User:|Assistant:|System:)"
    parts = re.split(pattern, normalized)
    
    current_role = None
    current_content = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part in ("User:", "Assistant:", "System:"):
            # Save previous turn
            if current_role and current_content:
                role = "user" if current_role == "User:" else (
                    "assistant" if current_role == "Assistant:" else "system"
                )
                content = " ".join(current_content).strip()
                # Limit individual turn content
                if len(content) > 4000:
                    content = content[:4000]
                turns.append({
                    "role": role,
                    "content": content,
                    "speaker": role,
                    "text": content,
                })
            
            current_role = part
            current_content = []
        else:
            current_content.append(part)
    
    # Don't forget the last turn
    if current_role and current_content:
        role = "user" if current_role == "User:" else (
            "assistant" if current_role == "Assistant:" else "system"
        )
        content = " ".join(current_content).strip()
        if len(content) > 4000:
            content = content[:4000]
        turns.append({
            "role": role,
            "content": content,
            "speaker": role,
            "text": content,
        })
    
    # If no structured format found, treat as plain text chunks
    if not turns and context.strip():
        chunk_size = 4000
        text = context.strip()
        for i in range(0, min(len(text), max_context_chars), chunk_size):
            chunk = text[i:i+chunk_size]
            turns.append({
                "role": "user",
                "content": chunk,
                "speaker": "user",
                "text": chunk,
            })
    
    # Limit total number of turns
    if len(turns) > max_turns:
        turns = turns[-max_turns:]
    
    return turns


def evaluate_instance(
    instance: Dict,
    memory,
    adapter,
    embedder,
) -> Dict[str, Any]:
    """
    Evaluate a single QA instance.
    
    Args:
        instance: QA instance dict
        memory: AuroraMemory instance
        adapter: MemoryAgentBenchAdapter
        embedder: Embedding provider
        
    Returns:
        Result dict with score, prediction, latency, etc.
    """
    from aurora.benchmark.interface import BenchmarkInstance, BenchmarkCapability
    
    cap_mapping = {
        "AR": BenchmarkCapability.ACCURATE_RETRIEVAL,
        "TTL": BenchmarkCapability.TEST_TIME_LEARNING,
        "LRU": BenchmarkCapability.LONG_RANGE_UNDERSTANDING,
        "CR": BenchmarkCapability.CONFLICT_RESOLUTION,
    }
    
    # Parse context into conversation history
    conversation_history = parse_context_to_turns(instance["context"])
    
    # Create BenchmarkInstance
    bench_instance = BenchmarkInstance(
        id=instance["id"],
        capability=cap_mapping.get(instance["capability"]),
        context=instance["context"],
        query=instance["question"],
        expected_answer=instance["expected_answer"],
        task_type=instance["capability"].lower(),
        conversation_history=conversation_history,
    )
    
    start_time = time.time()
    
    try:
        # Memory is already fresh (created for each instance)
        # Use adapter's evaluate method
        result = adapter.evaluate(bench_instance, memory)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "id": instance["id"],
            "capability": instance["capability"],
            "score": result.score,
            "is_correct": result.score >= 0.5,
            "prediction": result.predicted,
            "expected": instance["expected_answer"],
            "latency_ms": latency_ms,
            "error": None,
        }
        
    except Exception as e:
        import traceback
        latency_ms = (time.time() - start_time) * 1000
        return {
            "id": instance["id"],
            "capability": instance["capability"],
            "score": 0.0,
            "is_correct": False,
            "prediction": "",
            "expected": instance["expected_answer"],
            "latency_ms": latency_ms,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="Run MemoryAgentBench evaluation")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (all ~4310 QA pairs)")
    parser.add_argument("--samples", type=int, default=10, help="Samples per capability (default: 10)")
    args = parser.parse_args()
    
    samples_per_cap = None if args.full else args.samples
    
    print("=" * 70)
    print("AURORA MemoryAgentBench Evaluation")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'FULL' if args.full else f'SAMPLED ({args.samples} per capability)'}")
    print()
    
    # Import AURORA components
    from aurora.algorithms.aurora_core import AuroraMemory
    from aurora.algorithms.models.config import MemoryConfig
    from aurora.benchmark.adapters import MemoryAgentBenchAdapter
    from aurora.embeddings.bailian import BailianEmbedding
    from aurora.llm.ark import ArkLLM
    
    # Check API keys
    bailian_key = os.getenv('AURORA_BAILIAN_API_KEY')
    ark_key = os.getenv('AURORA_ARK_API_KEY')
    
    if not bailian_key or not ark_key:
        print("ERROR: Missing API keys!")
        print(f"  AURORA_BAILIAN_API_KEY: {'set' if bailian_key else 'MISSING'}")
        print(f"  AURORA_ARK_API_KEY: {'set' if ark_key else 'MISSING'}")
        sys.exit(1)
    
    print("API Configuration:")
    print(f"  Bailian API Key: ...{bailian_key[-8:]}")
    print(f"  Ark API Key: ...{ark_key[-8:]}")
    print()
    
    # Create API clients
    print("Initializing API clients...")
    embedder = BailianEmbedding(
        api_key=bailian_key,
        model='text-embedding-v4',
        dimension=1024
    )
    llm = ArkLLM(
        api_key=ark_key,
        model='doubao-1-5-pro-32k-250115'
    )
    print("  Embedder: BailianEmbedding (text-embedding-v4)")
    print("  LLM: ArkLLM (doubao-1-5-pro-32k-250115)")
    print()
    
    # Load dataset
    instances, stats = load_mab_dataset(samples_per_cap)
    print(f"\nTotal QA pairs to evaluate: {len(instances)}")
    print()
    
    # Create adapter with LLM for judging
    adapter = MemoryAgentBenchAdapter(llm_provider=llm, seed=42, embedder=embedder)
    
    # Create memory config
    config = MemoryConfig(
        max_plots=500,
        dim=1024,
    )
    
    # Run evaluation
    print("=" * 70)
    print("Starting Evaluation")
    print("=" * 70)
    print()
    
    results = []
    errors = []
    start_time = time.time()
    
    # Group instances by capability for organized output
    by_cap = defaultdict(list)
    for inst in instances:
        by_cap[inst["capability"]].append(inst)
    
    for cap_code in ["AR", "TTL", "LRU", "CR"]:
        cap_instances = by_cap.get(cap_code, [])
        if not cap_instances:
            continue
        
        cap_names = {
            "AR": "ACCURATE RETRIEVAL",
            "TTL": "TEST-TIME LEARNING",
            "LRU": "LONG-RANGE UNDERSTANDING",
            "CR": "CONFLICT RESOLUTION",
        }
        
        print(f"\n>>> Processing {cap_names[cap_code]} ({len(cap_instances)} instances) <<<")
        print("-" * 60)
        
        for idx, instance in enumerate(cap_instances):
            # Create fresh memory for each instance
            memory = AuroraMemory(cfg=config, seed=42)
            # Replace hash embedder with real embedder
            memory.embedder = embedder
            
            result = evaluate_instance(instance, memory, adapter, embedder)
            results.append(result)
            
            if result["error"]:
                errors.append(result)
            
            # Progress indicator
            status = "✓" if result["is_correct"] else "✗"
            print(f"  [{idx+1}/{len(cap_instances)}] {result['id']:<25} "
                  f"Score: {result['score']:.2f} {status} "
                  f"({result['latency_ms']:.0f}ms)")
    
    total_time = time.time() - start_time
    
    # Compute metrics
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print()
    
    # Overall metrics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    avg_score = sum(r["score"] for r in results) / total if total > 0 else 0
    
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    print(f"Overall Results:")
    print(f"  Total Instances: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {correct/total*100:.1f}%")
    print(f"  Average Score: {avg_score:.3f}")
    print(f"  Average Latency: {avg_latency:.0f}ms")
    print(f"  Total Time: {total_time:.1f}s")
    print(f"  Errors: {len(errors)}")
    print()
    
    # Per-capability metrics
    print("Per-Capability Results:")
    print("-" * 70)
    
    cap_results = defaultdict(list)
    for r in results:
        cap_results[r["capability"]].append(r)
    
    summary = {}
    for cap_code in ["AR", "TTL", "LRU", "CR"]:
        cap_r = cap_results.get(cap_code, [])
        if not cap_r:
            continue
        
        cap_total = len(cap_r)
        cap_correct = sum(1 for r in cap_r if r["is_correct"])
        cap_avg = sum(r["score"] for r in cap_r) / cap_total if cap_total > 0 else 0
        cap_latencies = [r["latency_ms"] for r in cap_r if r["latency_ms"] > 0]
        cap_avg_lat = sum(cap_latencies) / len(cap_latencies) if cap_latencies else 0
        
        cap_names = {
            "AR": "ACCURATE RETRIEVAL",
            "TTL": "TEST-TIME LEARNING",
            "LRU": "LONG-RANGE UNDERSTANDING",
            "CR": "CONFLICT RESOLUTION",
        }
        
        summary[cap_code] = {
            "total": cap_total,
            "correct": cap_correct,
            "accuracy": cap_correct/cap_total if cap_total > 0 else 0,
            "avg_score": cap_avg,
            "avg_latency_ms": cap_avg_lat,
        }
        
        print(f"  {cap_names[cap_code]} ({cap_code}):")
        print(f"    Total: {cap_total}, Correct: {cap_correct}, Accuracy: {cap_correct/cap_total*100:.1f}%")
        print(f"    Avg Score: {cap_avg:.3f}, Avg Latency: {cap_avg_lat:.0f}ms")
        print()
    
    # Final summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"| Capability               | Total | Correct | Accuracy | Avg Score |")
    print(f"|--------------------------|-------|---------|----------|-----------|")
    for cap_code in ["AR", "TTL", "LRU", "CR"]:
        if cap_code in summary:
            s = summary[cap_code]
            cap_names = {
                "AR": "Accurate Retrieval",
                "TTL": "Test-Time Learning",
                "LRU": "Long-Range Understanding",
                "CR": "Conflict Resolution",
            }
            name = cap_names[cap_code]
            print(f"| {name:<24} | {s['total']:>5} | {s['correct']:>7} | {s['accuracy']*100:>7.1f}% | {s['avg_score']:>9.3f} |")
    
    print(f"|--------------------------|-------|---------|----------|-----------|")
    print(f"| {'OVERALL':<24} | {total:>5} | {correct:>7} | {correct/total*100:>7.1f}% | {avg_score:>9.3f} |")
    print()
    
    # Dataset coverage info
    if not args.full:
        print("Note: This is a SAMPLED evaluation.")
        print(f"  Full dataset contains ~{stats['total']} QA pairs")
        print(f"  Run with --full for complete evaluation")
        print()
    
    # Save results
    output_file = Path(__file__).parent / "mab_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": "full" if args.full else f"sampled_{args.samples}",
            "total_instances": total,
            "correct_instances": correct,
            "overall_accuracy": correct/total if total > 0 else 0,
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
            "total_time_s": total_time,
            "errors_count": len(errors),
            "dataset_stats": stats,
            "by_capability": summary,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")
    print()
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
