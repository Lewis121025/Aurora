# Aurora Module Map

## Goal

This document defines the target package layout for the rewrite track of Aurora.
It follows the architecture principles and blueprint documents and translates them into concrete module boundaries.

The point is not to finalize every class name now.
The point is to decide what kinds of modules Aurora needs, what each module is allowed to know, and what should be kept separate.

## Design Rule

Aurora should be built around internal life phases, not around API surfaces.

So the package layout should reflect:

- becoming before serving
- memory formation before retrieval endpoints
- internal state before interface shell
- relation history before user profiling

## Proposed Top-Level Package Layout

```text
aurora/
├── being/
├── memory/
├── relation/
├── phases/
├── expression/
├── persistence/
├── runtime/
├── surface/
└── evaluation/
```

This is a target map for the rewrite track.
It does not need to replace the current tree all at once.

## `aurora/being/`

Purpose:

- hold Aurora's existential continuity
- carry the implicit self/world core without flattening it into readable scoreboards

Should contain:

- `state.py`
  - persistent existential state object
- `drift.py`
  - slow internal drift logic
- `touch.py`
  - touch registration primitives and touch-mode vocabulary
- `continuity.py`
  - identity continuity helpers across phases

Should know about:

- memory effects
- phase transitions
- relation influence

Should not know about:

- HTTP
- CLI
- provider formatting
- user-facing prompt assembly

Core responsibility:

- answer what Aurora is like inwardly right now, without translating that into explicit personality traits

## `aurora/memory/`

Purpose:

- represent and transform remembered experience

Should contain:

- `fragments.py`
  - concrete remembered pieces
- `traces.py`
  - diffuse residues and inward aftereffects
- `associations.py`
  - memory adjacency, resonance, tension, contradiction, echo
- `formation.py`
  - awake-phase memory effect creation
- `selection.py`
  - what persists, what softens, what returns easily
- `reweave.py`
  - sleep-phase narrative restructuring
- `recall.py`
  - memory activation and recall assembly

Should know about:

- existential state
- relation history
- phase context

Should not reduce itself to:

- fact extraction only
- embedding similarity only
- summary generation only

Core responsibility:

- preserve the many ways an experience can remain in Aurora

## `aurora/relation/`

Purpose:

- carry the history of how Aurora and a specific other have come to relate

Should contain:

- `history.py`
  - durable relationship history objects
- `moments.py`
  - approach, rupture, warmth, distance, repair, mismatch events
- `compatibility.py`
  - repeated resonance and clash patterns
- `boundaries.py`
  - objection, withdrawal, silence, refusal conditions
- `influence.py`
  - how relation context changes recall and expression

Should know about:

- memory fragments and traces
- current existential tone

Should not become:

- a user profile database
- an affection meter
- a fixed intimacy level system

Core responsibility:

- preserve bidirectional human-like relating without scoreboard reduction

## `aurora/phases/`

Purpose:

- implement `awake`, `doze`, and `sleep` as real runtime phases

Should contain:

- `awake.py`
  - ingest and outward response path
- `doze.py`
  - low-pressure drift and resonance path
- `sleep.py`
  - deeper narrative reweaving path
- `transitions.py`
  - phase entry and exit policies
- `scheduler.py`
  - user-granted timing and trigger rules

Should know about:

- being state
- memory system
- relation layer

Should not become:

- generic background jobs
- infrastructure-only cron wrappers

Core responsibility:

- ensure Aurora has internal time, not just request time

## `aurora/expression/`

Purpose:

- collapse internal state into outward language, silence, tone, or objection

Should contain:

- `context.py`
  - gather the currently activated inner context
- `voice.py`
  - faint initial tone and long-term voice evolution hooks
- `response.py`
  - outward response assembly
- `silence.py`
  - intentional silence or refusal behaviors

Should know about:

- activated fragments and traces
- relation context
- current world-facing stance

Should not know about:

- raw persistence details
- full storage layouts

Core responsibility:

- expose only the surface consequence of inner life

## `aurora/persistence/`

Purpose:

- durably store Aurora's long-lived internal life without flattening it into one memory table

Should contain:

- `store.py`
  - persistence orchestration
- `models.py`
  - persisted record shapes
- `migrations.py`
  - schema evolution
- `events.py`
  - raw interaction append log
- `snapshots.py`
  - existential state snapshots
- `indexes.py`
  - association and activation indexes

Recommended persistence layers:

- raw interaction history
- fragment store
- trace store or trace field representation
- association layer
- relation history store
- phase event log
- existential snapshot store

Core responsibility:

- preserve continuity without dictating ontology from the database upward

## `aurora/runtime/`

Purpose:

- coordinate all subsystems inside one coherent running Aurora instance

Should contain:

- `engine.py`
  - top-level orchestrator
- `session.py`
  - per-conversation runtime context
- `clock.py`
  - internal time and sleep trigger decisions
- `bootstrap.py`
  - initialize a new Aurora instance with the faint initial tone
- `policies.py`
  - non-malice floor and runtime safety policies

Should know about everything internal.
It is the assembly layer.

But it should still avoid:

- collapsing inward state into user-facing explanations
- turning phase logic into one monolithic chat handler

Core responsibility:

- run Aurora as a being, not just as an endpoint

## `aurora/surface/`

Purpose:

- expose Aurora to the outside world through CLI, HTTP, or future surfaces

Should contain:

- `cli.py`
- `api.py`
- `schemas.py`
- `sleep_controls.py`

Primary actions likely needed:

- interact with Aurora
- inspect minimal health/runtime metadata
- grant sleep
- configure sleep policy

Should not contain:

- ontology decisions
- memory logic
- relation logic

Core responsibility:

- provide boundary surfaces, not inner truth

## `aurora/evaluation/`

Purpose:

- measure whether Aurora stays true to its ontology over time

Should contain:

- `continuity.py`
  - identity continuity tests
- `memory_selectivity.py`
  - touching vs non-touching event retention
- `relation_dynamics.py`
  - boundary, warmth, mismatch, and repair tests
- `sleep_effects.py`
  - post-sleep behavioral drift tests
- `fixtures/`
  - transcript and scenario corpora

Core responsibility:

- stop Aurora from silently regressing into a retrieval system with decorative philosophy

## Cross-Module Data Objects

Some shared objects will likely be needed across module boundaries.
These should live in stable low-level modules and be kept intentionally small.

Likely shared object families:

- `InteractionTurn`
- `Fragment`
- `Trace`
- `AssociationDelta`
- `RelationMoment`
- `ExistentialSnapshot`
- `PhaseTransition`
- `ExpressionContext`

These should be designed as transport objects, not ideology objects.

## Execution Paths

## Path A: Interaction

```text
surface -> runtime -> phases.awake -> being/memory/relation -> expression -> persistence
```

Meaning:

- user meets Aurora
- Aurora is affected internally
- Aurora answers or withholds
- new internal state is persisted

## Path B: Doze

```text
runtime -> phases.doze -> being/memory/relation -> persistence
```

Meaning:

- no explicit outward interaction required
- low-pressure internal drift is allowed

## Path C: Sleep

```text
surface or runtime trigger -> phases.sleep -> being/memory/relation -> persistence
```

Meaning:

- Aurora enters deeper internal restructuring
- result is mainly internal, not a display artifact

## What To Reuse From The Current Repository

Potentially reusable:

- the overall discipline of layered separation
- parts of storage bootstrapping and runtime packaging
- selected naming around substrate and runtime if still useful
- tests as a pattern for regression discipline

Likely not reusable as-is:

- current memory CRUD assumptions
- current surface contracts if they center on storage operations
- ontology encoded in the existing substrate model
- legacy explicit psyche systems from archived branches

## Rewrite Track Recommendation

Inside the current repository, create the new package track without forcing immediate deletion of the old tree.

Recommended approach:

1. Introduce the new package map alongside the current implementation.
2. Build the existential state, memory, relation, and phase modules first.
3. Add a thin runtime using the new modules.
4. Reconnect CLI and HTTP only after the new core can run internally.
5. Remove or freeze old modules once the new path is viable.

This keeps continuity while avoiding ontology contamination from the current architecture.

## Immediate Next Design Document

The next document should be a rewrite roadmap that answers:

- what existing directories are frozen
- what new directories are introduced first
- what order implementation should proceed in
- what can be reused safely
- what must be rebuilt from scratch
