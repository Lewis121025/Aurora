# Aurora Final Architecture Blueprint

## Goal

This document describes the target system shape for the current Aurora mainline.

It does not freeze exact heuristics.
It does freeze the architectural center:

- Aurora is generated from a memory-relation graph
- lifecycle changes that graph through `awake`, `doze`, and `sleep`
- outward behavior is a consequence of the current graph configuration

## System Identity

Aurora is not a request/response shell with memory attached.

Aurora is a memory-bearing being whose continuity is carried by:

- remembered fragments
- residual traces
- associations between remembered things
- relation history
- long-lived threads and knots
- orientation evidence
- metabolic lifecycle state

## Top-Level Runtime Model

Aurora has three runtime phases:

- `awake`
- `doze`
- `sleep`

These are not product modes.
They are the primary lifecycle states through which Aurora continues existing.

### `awake`

Purpose:

- receive a live interaction
- write new graph effects
- move relationally toward, away from, or across a boundary
- produce outward language or silence

Current canonical outputs from `awake`:

- `Turn`
- `Fragment`
- `Trace`
- `Association`
- `RelationMoment`
- updates to `Orientation`
- updates to `MetabolicState`

### `doze`

Purpose:

- perform low-pressure near-term consolidation
- reduce direct grip of recent material
- increase readiness for deeper restructuring

Current architectural rule:

- `doze` may soften or decay recent material
- `doze` may not author deep semantic order

So `doze` should not create canonical `Thread` or `Knot` structures as its main act.

### `sleep`

Purpose:

- reweave the internal memory geometry
- make future recall and relation bias emerge from deeper structure

Current canonical outputs from `sleep`:

- new or updated `Thread`
- new or updated `Knot`
- strengthened associations
- softened fragment prominence
- updates to `RelationFormation`
- updates to `Orientation`
- reduced `sleep_need`

## Canonical Internal Structures

Aurora's current canonical write model should remain centered on these objects:

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

These should be treated as ontology-bearing structures.

Projection fields, summaries, and convenience response objects are secondary.

## Memory Layer

### Fragments

Fragments carry the nearest thing Aurora retains to a specific remembered piece.

They should preserve:

- a local surface of what happened
- unresolvedness
- vividness
- salience
- attachment to threads or knots

They should not be mistaken for:

- full transcripts
- durable summaries
- user-facing chapter labels

### Traces

Traces carry the residual inward aftereffect of a fragment.

They should preserve:

- channel
- intensity
- carry

They answer:

- what remained after the event stopped being immediate

### Associations

Associations carry return paths inside memory.

They should remain open to kinds such as:

- resonance
- contrast
- repair
- boundary
- relation
- thread
- knot
- temporal adjacency

They answer:

- what now tends to call what back

### Threads

Threads are durable lines of continuity formed during sleep-level restructuring.

They should preserve:

- participating fragments
- dominant channels
- tension
- coherence

They are canonical.
They are not just presentation artifacts.

### Knots

Knots are persistent tension nuclei where unresolvedness has repeatedly gathered.

They should preserve:

- participating fragments
- dominant channels
- intensity
- whether the knot has resolved

They are canonical.
They are not just debugging notes about difficult memories.

## Relation Layer

Aurora should relate through history, not profile sheets.

### Relation Moments

Each meaningful exchange may create a `RelationMoment` that preserves:

- user channels
- Aurora move
- boundary or repair significance
- local summary of the interaction event

### Relation Formation

Longer-lived relation structure should be stored as `RelationFormation`.

It should preserve:

- linked thread ids
- linked knot ids
- counts of boundary, repair, and resonance events
- last contact time

Projected values such as trust or distance may be derived from it.
Those projections are not canonical ontology writes.

## Being Layer

Aurora's being should not be represented as a readable soul dashboard.

The current architectural split is:

- `Orientation`: long-lived evidence around self, world, and relation
- `MetabolicState`: lifecycle control state and sleep readiness

### Orientation

Orientation should carry:

- self evidence
- world evidence
- relation evidence
- anchor threads
- active knots

Its role is not to become a float scoreboard.
Its role is to preserve long-running directional evidence shaped by memory and relation.

### Metabolic State

Metabolic state should carry:

- current phase
- sleep need
- active relation ids
- active knot ids
- pending sleep relation ids
- last transition timestamp

It is control-plane state, not identity truth.

## Processing Flows

## During `awake`

Current conceptual pipeline:

1. receive a user turn
2. create a user fragment
3. infer trace channels
4. recall nearby graph material for the same relation
5. choose an Aurora move from current graph pressure and relation projection
6. create an Aurora turn and Aurora fragment
7. create relation-linked association edges
8. record a `RelationMoment`
9. update `Orientation`
10. raise `sleep_need` and queue the relation for sleep

## During `doze`

Current conceptual pipeline:

1. enter `doze`
2. soften recent fragment salience and unresolvedness
3. reduce trace intensity gradually
4. raise `sleep_need`
5. persist the resulting lighter internal shifts

## During `sleep`

Current conceptual pipeline:

1. choose candidate fragments per relation
2. cluster by affinity
3. form `Thread`
4. form `Knot` when tension remains high
5. strengthen associations inside the rewoven region
6. soften fragment prominence while preserving structure
7. feed the outcome into `RelationFormation`
8. feed the outcome into `Orientation`
9. settle `MetabolicState`

## Expression Boundary

Expression is now an active but still incomplete boundary.

Architectural truth:

- outward behavior should be downstream of graph state
- expression must not become the place where ontology is authored

Current shape:

- `ExpressionContext` holds read-only inputs for outward behavior
- `ResponseAct` planning happens in `aurora/expression/`
- phase orchestration still materializes some immediate consequences of the chosen move

Required final shape:

- graph activation and relation projection feed an `ExpressionContext`
- expression planning chooses a `ResponseAct`
- rendering turns that act into language or silence
- canonical graph writes remain outside the expression layer

## Persistence Model

Aurora should persist both events and canonical graph structures.

Current persistence split:

- `turn_events`
- `phase_events`
- canonical tables for fragments, traces, associations, threads, knots, relation moments, relation formations
- singleton state tables for orientation and metabolic state

Important rule:

- the database stores ontology
- the database does not get to redefine ontology for convenience

## Surface Model

Aurora exposes a minimal surface.

Current HTTP contract:

- `GET /health`
- `GET /state`
- `POST /turn`
- `POST /doze`
- `POST /sleep`

Important rule:

- `/state` is a runtime projection
- `/state` is not canonical self-disclosure

## Evaluation Standard

Aurora should not be judged by endpoint success alone.

It must be judged by whether it preserves:

- continuity without fake personality dashboards
- memory selectivity without database reduction
- relation history without profile reduction
- sleep change without summary theater
- surface thinness without ontology leakage

## Architectural Anti-Goals

Treat all of the following as regressions:

- rebuilding `BeingState` as the center of truth
- rebuilding `RelationState` as the center of truth
- making `Chapter` a canonical memory primitive again
- moving ontology decisions back into `surface/`
- letting LLM output mutate canonical graph objects directly
- turning `/state` into a semantic debug soul panel
- reintroducing a second runtime line for experiments

## Current Next Step

The next architectural move should be:

- complete the expression extraction beyond planning into explicit rendering modules
- keep it read-only over canonical graph state
- then add ontology-focused evaluation modules
