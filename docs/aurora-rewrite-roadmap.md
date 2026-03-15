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
- engine injects LLM provider through to expression layer

### Phase 10: Touch Semantic Enhancement

- history-only touch: when no keywords match but graph has history, infer touch from recalled channels
- touch is no longer purely keyword-gated

### Phase 11: Dead Code Cleanup

- removed `aurora_ontology_core/`, `dynamics_profile.json`, `version.py`, `runtime/policies.py`
- removed Seed v1 references from `.env.example`, `tests/__init__.py`
- all unused imports eliminated (ruff clean)

### Phase 12: Unified Cognition

- replaced split pipeline (keyword touch → rule-based move → LLM rendering) with single cognitive act
- `expression/cognition.py`: one LLM call produces touch channels, relational move, and response text together

### Phase 13: LLM Required — Delete All Heuristic Code

- LLM is now required; engine refuses to start without it
- deleted all heuristic modules: `expression/response.py`, `expression/render.py`, `expression/voice.py`, `expression/silence.py`, `expression/template_store.py`, `expression/templates.json`, `expression/prompt.py`
- deleted keyword touch: `being/touch.py`, `being/touch_lexicon.json`
- `awake.py` reduced from 303 to 204 lines; single code path through unified cognition
- `surface/api.py`: removed eager module-level app creation; `build_app` requires an engine
- test infrastructure: `StubLLM` and `ContextAwareLLM` fixtures for deterministic testing

## Remaining Work

### Deepen Sleep Geometry

- reweave is still heuristic; thread/knot formation could benefit from LLM-assisted semantic clustering
- knot formation threshold is fixed; could adapt based on relation formation history

### Deepen Orientation Derivation

- orientation evidence still accumulates via channel-counting shortcuts during awake
- orientation strands should depend more on thread/knot/formation topology

### Expand Evaluation Coverage

- current evaluation covers continuity, relation dynamics, sleep effects
- needs richer multi-turn scenario regressions
- needs projection-boundary checks (ensure projections never become canonical)

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
