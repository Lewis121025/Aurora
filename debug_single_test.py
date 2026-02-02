#!/usr/bin/env python3
"""Debug a single LongMemEval test case."""

import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.embeddings.bailian import BailianEmbedding
from aurora.llm.ark import ArkLLM


def main():
    print("=== Debug Single Test Case ===\n")
    
    # Initialize
    print("1. Initializing embedder...")
    embedder = BailianEmbedding(
        api_key=os.getenv('AURORA_BAILIAN_API_KEY'),
        model='text-embedding-v3',
        dimension=1024
    )
    
    print("2. Initializing LLM...")
    llm = ArkLLM(
        api_key=os.getenv('AURORA_ARK_API_KEY'),
        model=os.getenv('AURORA_ARK_MODEL', 'doubao-1-5-pro-32k-250115'),
        base_url='https://ark.cn-beijing.volces.com/api/v3'
    )
    
    # Load one test case
    print("3. Loading data...")
    with open('data/longmemeval/longmemeval_oracle_new.json', 'r') as f:
        data = json.load(f)
    
    item = data[0]  # First item
    print(f"   Question type: {item['question_type']}")
    print(f"   Question: {item['question'][:80]}...")
    print(f"   Expected: {item['answer']}")
    print(f"   Sessions: {len(item['haystack_sessions'])}")
    
    print("\n4. Creating memory with benchmark_mode=True...")
    config = MemoryConfig(dim=1024, max_plots=5000, benchmark_mode=True)
    memory = AuroraMemory(cfg=config, seed=42, embedder=embedder, benchmark_mode=True)
    
    # Ingest first session only (to keep it fast)
    print("\n5. Ingesting first session...")
    session = item['haystack_sessions'][0]
    for i, turn in enumerate(session[:10]):  # First 10 turns
        text = f"{turn['role'].capitalize()}: {turn['content']}"
        print(f"   Turn {i+1}: {text[:50]}...")
        try:
            memory.ingest(text)
        except Exception as e:
            print(f"   ERROR during ingest: {e}")
            traceback.print_exc()
            return
    
    print(f"\n   Plots stored: {len(memory.plots)}")
    
    # Query
    print("\n6. Querying...")
    try:
        trace = memory.query(item['question'], k=15)
        print(f"   Results: {len(trace.ranked)}")
        
        # Show top results - ranked is List[Tuple[id, score, kind]]
        print("\n7. Top results:")
        for plot_id, score, kind in trace.ranked[:3]:
            plot = memory.plots.get(plot_id)
            if plot:
                print(f"   Score {score:.3f}: {plot.text[:60]}...")
        
        # Collect context
        context = ""
        for plot_id, score, kind in trace.ranked[:10]:
            plot = memory.plots.get(plot_id)
            if plot:
                context += plot.text + "\n"
        
        # Generate answer
        print("\n8. Generating answer...")
        prompt = f"""Answer the question based on the conversation history.

Context:
{context[:3000]}

Question: {item['question']}

Answer (be brief and specific):"""
        
        answer = llm.complete(prompt, max_tokens=100).strip()
        print(f"   Answer: {answer}")
        print(f"   Expected: {item['answer']}")
        
        # Evaluate
        expected = str(item['answer']).lower().strip()
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        match = expected in answer_lower or expected in context_lower
        print(f"\n9. Match: {match}")
        
    except Exception as e:
        print(f"   ERROR during query/answer: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()
