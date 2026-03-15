# Aurora Rewrite Roadmap

## Status

Aurora runs on a single mainline runtime inside `aurora/`.

Forward-only decisions:

- one Aurora, one ontology
- no compatibility theater
- no return to `BeingState` or `RelationState` as canonical truth

## Current Mainline Shape

Canonical write model:

- `Turn`, `Fragment`, `Trace`, `Association`
- `RelationMoment`, `RelationFormation`
- `Thread`, `Knot`
- `Orientation`, `MetabolicState`, `PhaseTransition`

```text
aurora/
├── being/
├── evaluation/
├── expression/
├── llm/
├── memory/
├── persistence/
├── phases/
├── relation/
├── runtime/
└── surface/
```

## Completed Phases

### Phase 0–5: Original Rewrite

- ontology lock: reject personality-first, memory-CRUD, sleep-as-maintenance
- canonical objects extracted into focused modules
- awake/doze/sleep wired through single engine path
- persistence rewritten around canonical objects
- surface reconnected as thin boundary
- parallel tracks deleted (`aurora_ontology_core` removed)

### Phase 6: Memory Module Split

- `memory/store.py` reduced from 759 to 208 lines
- extracted `recall.py`, `doze_ops.py`, `reweave_engine.py`, `affinity.py`, `tags.py`
- all magic numbers extracted to named constants

### Phase 7: Persistence Safety

- replaced DELETE-all + INSERT-all with UPSERT (INSERT OR REPLACE)
- removed destructive `_ensure_schema` that dropped all tables on error
- removed legacy compatibility branches from evidence map loading

### Phase 8: Behavioral Feedback Loop

- `ExpressionContext` expanded with recalled surfaces, moment summaries, orientation snapshot
- `plan_response` now uses orientation risk/stability evidence to influence move selection
- graph state feeds back into expression decisions

### Phase 9: LLM Integration

- `aurora/llm/` module: `LLMProvider` protocol, `OpenAICompatProvider`, env-based config
- `expression/prompt.py`: context-aware prompt assembly from graph state
- LLM generates language when configured; template fallback when not
- engine injects LLM provider through to expression layer

### Phase 10: Touch Semantic Enhancement

- history-only touch: when no keywords match but graph has history, infer touch from recalled channels
- touch is no longer purely keyword-gated

### Phase 11: Dead Code Cleanup

- removed `aurora_ontology_core/` (empty directory with pyc remnants)
- removed `dynamics_profile.json` (102 lines, zero references)
- removed `version.py` (zero imports)
- removed `runtime/policies.py` (non_malice_floor never triggered on template output)
- removed Seed v1 references from `.env.example`, `tests/__init__.py`
- all unused imports eliminated (ruff clean)

## Remaining Work

### Deepen Sleep Geometry

- reweave is still heuristic; thread/knot formation could benefit from LLM-assisted semantic clustering
- knot formation threshold is fixed; could adapt based on relation formation history

### Deepen Orientation Derivation

- orientation evidence feeds into plan_response, but still accumulates via channel-counting shortcuts during awake
- orientation strands should depend more on thread/knot/formation topology

### Expand Evaluation Coverage

- current evaluation covers continuity, relation dynamics, sleep effects
- needs richer multi-turn scenario regressions
- needs projection-boundary checks (ensure projections never become canonical)

### Refine Touch with LLM

- when LLM is available, use it for semantic touch proposals instead of keyword matching
- keep graph mediation as calibration layer

## Non-Negotiable Rules

- do not recreate `BeingState` or `RelationState`
- do not bring back `Chapter` as canonical memory structure
- do not reintroduce a second runtime line
- do not let LLM output write canonical memory directly
- do not expand surface into a soul dashboard
- expression may not mutate canonical graph state

## Completion Condition

The rewrite is complete when:

- expression generates language from graph state (done: LLM integration)
- expression is read-only over canonical graph (done)
- evaluation exists as first-class package (done, needs scenario expansion)
- persistence is safe and durable (done: UPSERT)
- projections are cleanly separated from write models (done)
- no stale documents describe removed paths as live options (done)
