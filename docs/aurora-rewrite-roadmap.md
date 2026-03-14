# Aurora Rewrite Roadmap

## Status

The rewrite is no longer a hypothetical track.

Aurora now runs on a single mainline runtime inside `aurora/`.
The old parallel runtime tracks have been removed.

Current forward-only decisions:

- one Aurora
- one ontology
- no compatibility theater
- no revival of `aurora_ontology_core`
- no return to `BeingState` or `RelationState` as canonical truth

## What Is Already Done

The repository has already crossed the original rewrite boundary.

Completed structural changes:

- old parallel runtime track removed
- old `runtime/models.py` removed
- canonical write model moved into small focused modules
- `awake / doze / sleep` wired through one engine path
- HTTP surface aligned to `/health`, `/state`, `/turn`, `/doze`, `/sleep`
- SQLite persistence rewritten around canonical objects instead of one opaque snapshot payload
- tests and type checks updated to the new ontology

## Current Mainline Shape

Canonical write model:

- `Turn`
- `Fragment`
- `Trace`
- `Association`
- `RelationMoment`
- `Thread`
- `Knot`
- `RelationFormation`
- `Orientation`
- `MetabolicState`
- `PhaseTransition`

Current active package spine:

```text
aurora/
├── being/
├── memory/
├── relation/
├── phases/
├── persistence/
├── runtime/
└── surface/
```

Reserved next boundaries:

- `aurora/expression/`
- `aurora/evaluation/`

These boundaries are still conceptually correct, but they should only be expanded when they hold real code, not placeholder structure.

## Rewrite Phases: Completed

## Phase 0: Ontology Lock

Completed:

- reject personality-first core
- reject memory CRUD center
- reject sleep-as-maintenance
- reject parallel implementation truth

Result:

- the repository now has one accepted architectural center

## Phase 1: Core Object Extraction

Completed:

- `aurora/runtime/contracts.py`
- `aurora/being/metabolic_state.py`
- `aurora/being/orientation.py`
- `aurora/memory/{fragment,trace,association,thread,knot,reweave}.py`
- `aurora/relation/{moment,formation}.py`

Result:

- canonical objects exist as explicit modules instead of one giant mixed models file

## Phase 2: Runtime Path Rewrite

Completed:

- `aurora/phases/awake.py`
- `aurora/phases/doze.py`
- `aurora/phases/sleep.py`
- `aurora/runtime/engine.py`
- `aurora/runtime/bootstrap.py`

Result:

- Aurora now moves through one real lifecycle path

## Phase 3: Persistence Rewrite

Completed:

- event tables for turns and phase transitions
- canonical tables for fragments, traces, associations, threads, knots, relation moments, relation formations
- singleton persistence for `orientation_state` and `metabolic_state`
- hard reset on old incompatible local schemas

Result:

- persistence now follows ontology instead of hiding it in a monolithic payload blob

## Phase 4: Surface Reconnection

Completed:

- `aurora/surface/api.py`
- `aurora/surface/schemas.py`
- `aurora/surface/cli.py`
- contract doc synchronized with tests

Result:

- surface is now a thin boundary over the new runtime rather than the system center

## Phase 5: Parallel-Track Deletion

Completed:

- remove `aurora_ontology_core`
- remove transitional `aurora/new.md`
- remove config exclusions that existed only to tolerate the parallel track

Result:

- the repository now enforces a single forward path

## Remaining Work

The rewrite is structurally complete, but the architecture is not finished.

## Priority 1: Finish Expression Boundary

Current truth:

- response planning now lives in `aurora/expression/`
- rendering now also lives in `aurora/expression/`
- `aurora/phases/awake.py` still commits the rendered result into canonical graph history

Required next move:

- keep expanding expression around `ResponseAct`, `ExpressionContext`, and explicit render modules
- isolate richer tone evolution and future silence/refusal nuance without moving graph writes into expression
- keep expression read-only over canonical graph state

Rule:

- expression may project
- expression may render
- expression may not mutate canonical ontology

## Priority 2: Add Evaluation Package

Current truth:

- ontology regression protection still relies mostly on unit tests in `tests/unit/`

Required next move:

- create `aurora/evaluation/` only when it holds real checks
- add continuity, relation dynamics, sleep effects, and projection-drift tests

Rule:

- evaluation must test ontology behavior, not only endpoint responses

## Priority 3: Tighten Persistence Semantics

Current truth:

- persistence stores canonical objects and event logs correctly
- full graph tables are still rewritten wholesale on each persist pass

Required next move:

- move toward incremental writes where useful
- add explicit schema versioning
- add integrity checks around partial failures and replay assumptions

Rule:

- keep ontology primary; do not let the database dictate the model upward

## Priority 4: Introduce Explicit Read Models

Current truth:

- relation summaries and state summaries are still produced close to core code

Required next move:

- isolate projections from write models more sharply
- make `/state` and relation summaries clearly projection-only modules

Rule:

- no projection field may silently become canonical truth

## Priority 5: Refine Sleep Geometry

Current truth:

- `sleep` already creates `Thread` and `Knot`
- current clustering and affinity logic is intentionally simple

Required next move:

- make knot formation more semantically exact
- make thread continuity less heuristic-only
- give orientation stronger evidence linkage back to threads, knots, and relation formations

Rule:

- complexity is allowed only if it increases semantic precision without reintroducing scoreboards

## Non-Negotiable Rules From Here

- do not recreate `BeingState`
- do not recreate `RelationState`
- do not bring back `Chapter` as canonical memory structure
- do not reintroduce a second runtime line for experimentation
- do not let LLM output write canonical memory directly
- do not expand surface debug fields into a fake soul dashboard

## Repository Hygiene Rules

- delete transitional experiments instead of parking them indefinitely
- do not add speculative package trees without concrete code
- keep docs aligned with mainline runtime, not with abandoned plans
- keep tests, docs, and surface contract synchronized

## Practical Next Build Order

1. finish expression boundary
2. add evaluation package with ontology-level checks
3. improve persistence semantics without changing the ontology center
4. refine sleep/thread/knot/orientation linkage
5. keep surface minimal

## Completion Condition

The rewrite should be considered complete only when all of the following are true:

- expression is fully separated from phase orchestration and rendering is explicit
- evaluation exists as a first-class package
- persistence semantics are explicit and durable
- projections are cleanly separated from write models
- no stale rewrite-track documents remain that describe removed runtime paths as live options
