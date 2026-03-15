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

All modules are active. `expression/` owns unified LLM cognition. `llm/` provides the required LLM provider. `evaluation/` contains ontology checks for continuity, relation dynamics, and sleep effects.

## `aurora/being/`

Current files:

- `metabolic_state.py`
- `orientation.py`

Purpose:

- hold lifecycle control state without pretending to explain Aurora's identity
- hold long-running self/world/relation evidence with source links back to lived moments and sleep structures

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
- `recall.py`
- `doze_ops.py`
- `reweave_engine.py`
- `affinity.py`
- `tags.py`

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

`store.py` owns graph data and CRUD. Operational logic is extracted into focused modules:

- `recall.py`: recall ranking and activation channel extraction
- `doze_ops.py`: hover and decay operations
- `reweave_engine.py`: sleep reweave orchestration, region building, thread/knot formation
- `affinity.py`: fragment affinity, overlap, and structural pressure calculations
- `tags.py`: text tag extraction

Must not become:

- a flat fact store
- a summary cache pretending to be memory
- a hidden personality engine

## `aurora/relation/`

Current files:

- `moment.py`
- `formation.py`
- `store.py`
- `decision.py`
- `projectors.py`

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

- `awake.py`: ingest interaction, run unified LLM cognition, write graph effects, update relation history
- `doze.py`: hover around active relation material, then apply low-pressure decay and raise sleep pressure
- `sleep.py`: reweave memory graph and feed resulting changes back into orientation and metabolic state
- `transitions.py`: create explicit `PhaseTransition` records
- `outcomes.py`: keep phase outputs small and typed

Known debt:

- doze resonance is still heuristic compared with the intended hovering semantics
- sleep reweave is structural and heuristic; LLM-assisted semantic clustering is the next step

## `aurora/runtime/`

Current files:

- `contracts.py`
- `state.py`
- `bootstrap.py`
- `clock.py`
- `engine.py`
- `projections.py`

Purpose:

- coordinate the whole Aurora instance without becoming a god-model dump

Current responsibilities:

- `contracts.py`: shared enums and small transport-like core objects
- `state.py`: runtime container for `orientation`, `metabolic`, and `transitions`
- `bootstrap.py`: initialize clean runtime state
- `clock.py`: current time source
- `engine.py`: orchestration for turn, doze, sleep, and projections
- `projections.py`: health and state summary projections for surface endpoints

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

Persistence uses UPSERT (INSERT OR REPLACE) for graph tables and ON CONFLICT for singletons.

Known debt:

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

## `aurora/llm/`

Current files:

- `provider.py`
- `config.py`
- `openai_compat.py`

Purpose:

- provide the required LLM integration for Aurora's cognition
- `provider.py` defines the `LLMProvider` protocol
- `openai_compat.py` implements OpenAI-compatible API calls (covers Bailian, DeepSeek, etc.)
- `config.py` loads provider configuration from environment variables

Aurora requires an LLM provider. Without one, the engine refuses to start.

## `aurora/expression/`

Current files:

- `cognition.py`
- `context.py`

Current role:

- `cognition.py`: unified LLM cognitive act — a single LLM call produces touch channels, relational move, and response text together, because a being experiences, decides, and speaks as one act
- `context.py`: read-only expression context carrying recalled surfaces, moment summaries, and orientation snapshot

Hard rule:

- expression may not write canonical graph state

## `aurora/evaluation/`

Current files:

- `continuity.py`
- `relation_dynamics.py`
- `sleep_effects.py`

Current role:

- protect ontology against regression at the package level
- test continuity, relation dynamics, and sleep effects without going through surface-only assertions

Future role:

- add projection-boundary checks
- add richer scenario fixtures

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

## Next Refinement Order

1. deepen sleep geometry with LLM-assisted semantic clustering
2. refine orientation derivation from thread/knot/formation topology
3. expand evaluation with richer scenario fixtures and projection-boundary checks
