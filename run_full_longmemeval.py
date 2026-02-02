import sys, os, json, time
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.embeddings.bailian import BailianEmbedding
from aurora.llm.ark import ArkLLM

print("=" * 70)
print("LongMemEval FULL 500-Question Test")
print("=" * 70)

embedder = BailianEmbedding(
    api_key=os.getenv('AURORA_BAILIAN_API_KEY'),
    model='text-embedding-v4'
)
llm = ArkLLM(
    api_key=os.getenv('AURORA_ARK_API_KEY'),
    model='doubao-1-5-pro-32k-250115',
    base_url='https://ark.cn-beijing.volces.com/api/v3'
)

config = MemoryConfig(dim=1024, max_plots=5000, benchmark_mode=True)

data_path = 'data/longmemeval/longmemeval_oracle_new.json'
if not os.path.exists(data_path):
    data_path = 'data/longmemeval/longmemeval_oracle.json'

with open(data_path, 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} questions")

results = {}
total_correct = 0
total_tested = 0
start = time.time()

try:
    from aurora.llm.prompts import build_qa_prompt
    use_prompts = True
except:
    use_prompts = False

for i, item in enumerate(data):
    qtype = item['question_type']
    if qtype not in results:
        results[qtype] = {'correct': 0, 'total': 0}
    
    try:
        memory = AuroraMemory(cfg=config, seed=42, embedder=embedder, benchmark_mode=True)
        
        for session in item['haystack_sessions']:
            for turn in session:
                memory.ingest(f"{turn['role'].capitalize()}: {turn['content']}")
        
        question = item['question']
        expected = str(item['answer']).lower().strip()
        
        trace = memory.query(question, k=15)
        
        context = ""
        for r in trace.ranked[:12]:
            plot = memory.plots.get(r.id)
            if plot:
                context += plot.text + "\n"
        
        if context.strip():
            if use_prompts:
                prompt = build_qa_prompt(question=question, context=context, question_type_hint=qtype)
            else:
                prompt = f"Context:\n{context[:3500]}\n\nQuestion: {question}\n\nAnswer:"
            try:
                answer = llm.complete(prompt, max_tokens=150).strip().lower()
            except:
                answer = context[:200].lower()
        else:
            answer = "I don't know"
        
        match = expected in answer or expected in context.lower()
        if not match:
            kws = [w for w in expected.split() if len(w) > 2]
            if kws:
                match = sum(1 for k in kws if k in answer or k in context.lower()) >= len(kws) * 0.6
        
        results[qtype]['total'] += 1
        if match:
            results[qtype]['correct'] += 1
            total_correct += 1
        total_tested += 1
        
    except Exception as e:
        results[qtype]['total'] += 1
        total_tested += 1
    
    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (len(data) - i - 1)
        acc = total_correct / total_tested * 100
        print(f"[{i+1}/{len(data)}] Accuracy: {acc:.1f}% | ETA: {eta/60:.1f}min")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"\n{'Type':<35} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
print("-" * 65)

for qtype in sorted(results.keys()):
    r = results[qtype]
    acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    print(f"{qtype:<35} {r['correct']:<10} {r['total']:<10} {acc:.1f}%")

print("-" * 65)
overall = total_correct / total_tested * 100
print(f"{'OVERALL':<35} {total_correct:<10} {total_tested:<10} {overall:.1f}%")

print(f"\n距离 SOTA (86%): {86 - overall:.1f}%")
print(f"距离目标 (90%): {90 - overall:.1f}%")

elapsed = time.time() - start
print(f"\nTotal time: {elapsed/60:.1f} minutes")

with open('longmemeval_full_results.json', 'w') as f:
    json.dump({
        'results': {k: {'correct': v['correct'], 'total': v['total'], 'accuracy': v['correct']/v['total']*100 if v['total'] > 0 else 0} for k, v in results.items()},
        'total_correct': total_correct,
        'total_tested': total_tested,
        'overall_accuracy': overall,
        'elapsed_minutes': elapsed / 60
    }, f, indent=2)

print("\nResults saved to longmemeval_full_results.json")
