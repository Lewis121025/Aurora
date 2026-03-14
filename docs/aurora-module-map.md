# Aurora Module Map

## Goal

This document defines the active module boundaries for the current Aurora mainline.

It is no longer a speculative rewrite tree.
It describes the single runtime path that now exists in `aurora/` and the next boundaries that should be extracted later.

## Main Rule

Aurora is built around lifecycle and graph continuity, not around surface endpoints.

So module boundaries must preserve:

- memory formation before API response
- relation history before user profiling
- internal change before explanation
- write models before projections

## Active Top-Level Layout

```text
aurora/
├── being/
├── expression/
├── memory/
├── persistence/
├── phases/
├── relation/
├── runtime/
└── surface/
```

Status notes:

- `being/`, `memory/`, `relation/`, `phases/`, `expression/`, `persistence/`, `runtime/`, and `surface/` are active
- `expression/` now owns response planning, but not the full rendering stack yet
- `evaluation/` is not yet in the tree and should only be added when it contains real ontology checks

## `aurora/being/`

Current files:

- `metabolic_state.py`
- `orientation.py`

Purpose:

- hold lifecycle control state without pretending to explain Aurora's identity
- hold long-running self/world/relation evidence without reducing it to scoreboard floats

Current canonical objects:

- `MetabolicState`
- `Orientation`

Must not become:

- a personality dashboard
- a worldview score table
- a replacement for the memory-relation graph

## `aurora/memory/`

Current files:

- `fragment.py`
- `trace.py`
- `association.py`
- `thread.py`
- `knot.py`
- `reweave.py`
- `store.py`

Purpose:

- hold the canonical memory graph
- support recall, doze decay, and sleep reweave

Current canonical objects:

- `Fragment`
- `Trace`
- `Association`
- `Thread`
- `Knot`
- `SleepMutation`

`store.py` currently owns:

- graph storage
- recall ranking
- doze decay
- sleep candidate selection
- simple clustering
- thread and knot formation

Must not become:

- a flat fact store
- a summary cache pretending to be memory
- a hidden personality engine

## `aurora/relation/`

Current files:

- `moment.py`
- `formation.py`
- `store.py`

Purpose:

- preserve relation history as lived moments and durable formations

Current canonical objects:

- `RelationMoment`
- `RelationFormation`

Important boundary:

- relation summaries may be projected from formations
- projected trust or distance values are not canonical ontology writes

Must not become:

- a user profile database
- an affection meter
- an intimacy ladder

## `aurora/phases/`

Current files:

- `awake.py`
- `doze.py`
- `sleep.py`
- `transitions.py`
- `outcomes.py`

Purpose:

- express lifecycle as real runtime phases

Current responsibilities:

- `awake.py`: ingest interaction, write graph effects, update relation history, choose immediate outward move
- `doze.py`: apply low-pressure decay and raise sleep pressure
- `sleep.py`: reweave memory graph and feed resulting changes back into orientation and metabolic state
- `transitions.py`: create explicit `PhaseTransition` records
- `outcomes.py`: keep phase outputs small and typed

Known debt:

- response planning has been extracted, but rendering is still intentionally minimal

## `aurora/runtime/`

Current files:

- `contracts.py`
- `state.py`
- `bootstrap.py`
- `clock.py`
- `engine.py`
- `policies.py`

Purpose:

- coordinate the whole Aurora instance without becoming a god-model dump

Current responsibilities:

- `contracts.py`: shared enums and small transport-like core objects
- `state.py`: runtime container for `orientation`, `metabolic`, and `transitions`
- `bootstrap.py`: initialize clean runtime state
- `clock.py`: current time source
- `engine.py`: orchestration for turn, doze, sleep, and projections
- `policies.py`: minimal runtime policy helpers

Must not become:

- a single monolithic chat handler
- a semantic dumping ground for all ontology objects

## `aurora/persistence/`

Current files:

- `migrations.py`
- `store.py`

Purpose:

- persist canonical write models and lifecycle events

Current tables:

- `turn_events`
- `phase_events`
- `fragments`
- `traces`
- `associations`
- `threads`
- `knots`
- `relation_moments`
- `relation_formations`
- `orientation_state`
- `metabolic_state`

Important boundary:

- persistence stores ontology faithfully
- persistence must not redefine ontology upward from database convenience

Known debt:

- core tables are rewritten wholesale on persist
- explicit schema versioning is not yet present

## `aurora/surface/`

Current files:

- `api.py`
- `schemas.py`
- `cli.py`

Purpose:

- expose Aurora through HTTP and CLI
- keep these surfaces thin

Current public actions:

- `GET /health`
- `GET /state`
- `POST /turn`
- `POST /doze`
- `POST /sleep`
- CLI `turn`
- CLI `doze`
- CLI `sleep`

Important boundary:

- `/state` is a projection
- `/state` is not canonical self-knowledge

## `aurora/expression/`

Current files:

- `context.py`
- `render.py`
- `response.py`
- `silence.py`
- `voice.py`

Current role:

- hold read-only expression context
- choose `ResponseAct` from current graph pressure and relation projection
- render refusal, silence, and voiced text without mutating canonical graph state

Future role:

- expand from current planning and rendering into a fuller expression stack if real complexity appears

Hard rule:

- expression may not write canonical graph state

## `aurora/evaluation/`

Current status:

- not yet implemented

Correct future role:

- protect ontology against regression
- test continuity, relation dynamics, sleep effects, and projection boundaries

Hard rule:

- evaluation must verify behavior that matters to Aurora's ontology, not only API success

## Shared Canonical Objects

Current shared object family:

- `Turn`
- `PhaseTransition`
- `Fragment`
- `Trace`
- `Association`
- `Thread`
- `Knot`
- `RelationMoment`
- `RelationFormation`
- `Orientation`
- `MetabolicState`
- `SleepMutation`

Rule:

- these are write-model or lifecycle objects
- projection objects should remain separate and secondary

## Execution Paths

## Path A: Interaction

```text
surface -> runtime.engine -> phases.awake -> memory/relation/being -> persistence
```

Meaning:

- one user turn arrives
- graph effects are written
- relation history updates
- a response is produced
- the result is persisted

## Path B: Doze

```text
surface or runtime.engine -> phases.doze -> memory/being -> persistence
```

Meaning:

- recent material softens and decays lightly
- sleep pressure can rise

## Path C: Sleep

```text
surface or runtime.engine -> phases.sleep -> memory/relation/being -> persistence
```

Meaning:

- graph regions are re-clustered and reweighted
- threads and knots form
- relation formations and orientation absorb the change

## Forbidden Regressions

- do not reintroduce `BeingState` as canonical truth
- do not reintroduce `RelationState` as canonical truth
- do not reintroduce `Chapter` as canonical memory center
- do not let `surface/` decide ontology
- do not let `persistence/` dictate ontology from schema convenience
- do not let `expression/` mutate canonical graph objects

## Next Extraction Order

1. extract expression out of `phases/awake.py`
2. finish expression rendering and silence/refusal modules
3. introduce explicit projection/read-model modules
4. add `evaluation/` once ontology checks are concrete
5. tighten persistence semantics without changing the write-model center
