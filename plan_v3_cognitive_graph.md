# Aurora Cognitive Graph Architecture (v3.0)

## The Core Philosophy: "Everything is a Spark"

We do not want to bolt on a separate Knowledge Graph database (like Neo4j) or introduce rigid relational schemas (`Subject-Predicate-Object`). That violates the thermodynamic, mathematically elegant foundation of Aurora.

Instead of adding new data structures, we **expand the definition of a Spark and its Topology**. We embrace "Connectionism" (like neural networks): meaning arises from the *structure of connections* between simple nodes, not from complex node types.

## 1. Universal Resonance & Structural Forgetting (The Abyss)

**The Current Problem:** Forgetting is binary. A node hits 0 energy, we delete it.
**The Root Solution (The Abyss Concept):**
- A memory never truly dies; it just loses its structural meaning.
- When a Spark's energy drops below a critical threshold (e.g., `energy < 0.1`), we do **not** delete it from `state.sparks`.
- Instead, we **sever its temporal links** (`prev_id` and `next_id` are set to `None`).
- The spark falls into the "Abyss" (it becomes an isolated node). It no longer appears during linear episodic replay.
- **Quantum Revival:** If a future input has an incredibly high vector cosine similarity (a "Déjà vu" trigger), this isolated node can be structurally "re-attached" via `resonant_links`. This perfectly models implicit memory.
- **True Deletion:** Only when the hard capacity (4096) is breached do we garbage-collect the oldest nodes *that are already in the Abyss*.

## 2. Abstract "Fact" Nodes as Attractors (Semantic Gravity)

**The Current Problem:** We have `fossil` nodes (summaries), but how does the system learn discrete facts ("User's dog is named Max") without building a brittle Entity-Relationship system?
**The Root Solution (Semantic Attractors):**
- Introduce a third type of spark: `type: "concept"`.
- When the internal dream process (or compression) identifies an objective truth, it generates a `"concept"` spark (e.g., text: `[Concept]: User loves coffee`).
- **The Gravity of Concepts:** Concept sparks have **infinite decay half-life** (they don't lose energy naturally) but they have **no temporal edges** (`prev_id/next_id` are always None). They sit "above" the timeline.
- When an `episodic` spark is formed, the physical engine immediately calculates its vector similarity to all existing `concept` sparks. If similarity > threshold, an automatic `resonant_link` is forged.
- **The Result:** The `"User loves coffee"` concept node becomes a gravitational hub. Dozens of temporal episodic nodes (where the user drank coffee) become structurally tethered to it. When the concept is triggered, the system can instantly sample the tethered episodes.

## Implementation Blueprint (No New Dependencies)

1. **State Modifications (`state.py`)**:
   - `Spark.type` enum expanded: `'episodic'`, `'fossil'`, `'concept'`.
   - Ensure `resonant_links` are weighted or naturally ordered by structural affinity.

2. **Thermodynamic Refinement (`engine.py`)**:
   - `_thermodynamics` applies different decay formulas based on `Spark.type`.
   - `_reincarnate` implements "The Abyss": sever links instead of immediate deletion, unless capacity is maxed.

3. **Cognitive Dreaming (`provider.py` & `runtime.py`)**:
   - Introduce a new dream phase: `is_concept_extraction`.
   - Instruct the LLM to scan recent high-energy episodes and output pure, isolated declarative statements (Facts).
   - Feed these back into the engine to spawn `'concept'` sparks.
