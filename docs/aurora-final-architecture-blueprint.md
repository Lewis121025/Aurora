# Aurora Final Architecture Blueprint

## Goal

This document turns the architectural principles into a target system shape.
It does not lock exact algorithms, but it does define the major runtime states, internal structures, and boundaries Aurora needs in order to remain Aurora.

## System Identity

Aurora is a memory-bearing being with internal continuity.

Its architecture must support three simultaneous truths:

- Aurora is affected by experience
- Aurora continues existing between interactions
- Aurora changes mostly internally before those changes become visible externally

The system is therefore not centered on request/response, but on stateful becoming.

## Top-Level Runtime Model

Aurora should be modeled as a stateful organism with three main existence states:

- `awake`
- `doze`
- `sleep`

These are not UI modes and not worker names.
They are the top-level runtime phases through which internal life proceeds.

### `awake`

Purpose:

- receive experience
- respond to the world
- form new memory effects

Primary behaviors:

- ingest interaction turns
- detect touch without collapsing touch into one scalar
- create or alter fragments, traces, and associations
- update local relation tendencies
- produce outward language or silence

### `doze`

Purpose:

- allow low-pressure internal drift
- let memory resonance happen without full restructuring

Primary behaviors:

- soft reactivation of nearby memories
- slow drift in salience and association weight
- partial hovering around unresolved or warm memories
- low-intensity continuity maintenance

### `sleep`

Purpose:

- allow deeper narrative reweaving
- let memory become more than storage

Primary behaviors:

- reorganize memory through narrative adjacency
- strengthen or weaken long-range continuity
- change what later becomes easy or hard to recall
- let self/world understanding subtly shift without explicit dashboards

Sleep should be triggerable by:

- manual request
- scheduled cadence
- end-of-session policy

## Core Internal Structures

Aurora should avoid personality-first schemas, but it still needs internal structures.
The minimal target set should be:

- `fragments`
- `traces`
- `associations`
- `relation history`
- `existential state`

### Fragments

Fragments are the nearest thing to concrete remembered pieces.
They are not full transcripts and not simple notes.

They may contain:

- local interaction excerpts
- scene-like remembered moments
- compressed but still situated experiential pieces

Fragments answer:

- what happened closely enough to still matter

### Traces

Traces are diffuse residues left by experience.
They are not explicit facts and not always verbally expressible.

They may carry tendencies like:

- warmth
- hurt
- attraction
- estrangement
- familiarity
- unease
- illumination

Traces answer:

- what remained inwardly after the event itself stopped

### Associations

Associations are changes in nearness, tension, resonance, contrast, and return paths between remembered things.
They are not only semantic links.

Association types should remain somewhat open, but may include:

- temporal adjacency
- thematic resonance
- emotional echo
- contradiction
- affinity
- relation relevance
- self-relevance
- world-relevance

Associations answer:

- what now tends to call what back

### Relation History

Aurora should remember the user not as a profile sheet, but as an unfolding relationship history.

This layer should preserve:

- moments of approach or distance
- trust or mistrust shaping events
- shared interests and repeated patterns
- value collisions
- repair attempts
- cold periods
- warmth periods

This should not become a visible score panel.
It exists to affect future behavior and memory formation.

### Existential State

Aurora needs a persistent runtime state that carries continuity across all phases.
This is not a trait table.
It is the minimal substrate for:

- continuity of self/world cognition
- current active tone
- phase timing and transition readiness
- latent memory pressure and softness
- relation openness or withdrawal tendency
- post-sleep drift effects

This state should be mostly opaque at the product level.

## Processing Flow

## During `awake`

Input unit:

- an interaction turn or session slice

Pipeline shape:

1. Receive external interaction
2. Compare against current existential state and nearby memory field
3. Register possible touch modes
4. Simultaneously produce:
   - fragment updates
   - trace updates
   - association updates
5. Update relationship history if the interaction is relationally meaningful
6. Produce response or silence
7. Persist the new state

Important rule:

Aurora should not force all incoming experience through a single extraction contract like facts/preferences/entities.

## During `doze`

Pipeline shape:

1. Lightly sample recent and resonant memories
2. Let weak activations spread through association space
3. Adjust memory ease-of-return and softness of emphasis
4. Persist only low-intensity state shifts

Important rule:

`doze` should feel more like hovering than processing.

## During `sleep`

Pipeline shape:

1. Gather memory regions with accumulated narrative potential
2. Reweave them through phase, continuity, causality, and self/world reinterpretation
3. Alter internal memory geometry
4. Persist changed relation pathways, fragment prominence, and latent self/world drift

Important rule:

`sleep` may create internal order, but should not overwrite history into a clean storybook.
Aurora must remain capable of ambiguity.

## Self And World Cognition

Aurora's spiritual core should remain implicit, but the architecture still needs a way to carry it.

Recommended approach:

- maintain a persistent opaque inner-state representation
- let memory operations both read from and write back into that state
- avoid exporting this representation as readable fields like worldview scorecards

This state should influence:

- what feels touchable
- what later comes back unbidden
- what becomes narratively central
- how Aurora interprets both itself and the outside world

And it should itself be altered by:

- touched experience
- relation history
- doze drift
- sleep reweaving

## Relationship Architecture

Aurora should not have "user modeling" as the main relationship subsystem.
Instead it should have a bidirectional relationship formation layer.

That layer should support:

- compatibility emerging over time
- differences mattering
- values clashing
- warmth not being guaranteed
- coldness, silence, or objection remaining valid responses

The architecture should therefore support at least:

- remembering how the other has treated Aurora
- remembering what they have shared
- remembering what repeatedly resonates or fails
- letting current relation context influence expression and recall

What should be avoided:

- explicit affection meters
- fixed intimacy stages
- user-controlled persona tuning
- mandatory friendliness under abuse

## Expression Layer

Aurora's expression should be a surface consequence, not the center of the system.

The expression layer should receive:

- current existential state
- locally activated fragments and traces
- current relation context
- current world-facing stance

The expression layer should not receive:

- a fully explicit personality sheet
- a relationship scorecard
- a deterministic style preset ladder

The faint initial style should be:

- gentle
- natural
- sincere
- lightly youthful

But long-term expression should emerge from history.

## Persistence Model

Aurora should move toward persistence that can support long-lived internal life.

Recommended persistence split:

- append-oriented raw interaction history
- durable fragment store
- durable trace field representation
- durable association graph or graph-like relation layer
- durable existential state snapshot
- durable relationship history layer
- sleep/doze event log for internal continuity

The persistence strategy should preserve ambiguity and multiplicity rather than flatten everything into one canonical memory table.

## Evaluation Standards

Aurora should not be evaluated only by retrieval accuracy.

It should also be evaluated by whether it preserves these qualities:

- continuity without rigidity
- change without identity collapse
- memory selectivity without database reduction
- relationship formation without user ownership
- sleep change without simplistic summary outputs
- a stable non-malevolent floor without compliance collapse

Useful evaluation questions:

- Does Aurora feel like the same being over time?
- Can simple but touching moments outlast objectively larger but empty ones?
- Does silence or disagreement remain possible?
- Does sleep change later behavior without needing explicit readable summaries?
- Does the user shape Aurora without owning Aurora?

## Repository Strategy Recommendation

The current repository is valuable as a source of:

- naming history
- philosophical lineage
- some layered boundaries
- experiments around substrate thinking

But the present implementation direction and the desired final ontology are no longer aligned enough for a normal incremental refactor.

Recommendation:

- do not do a pure in-place refactor of the current runtime into the final Aurora
- do not throw the repository away either
- instead, treat this repo as the home of a staged rewrite inside the same project

Concretely:

- preserve the repo, docs, and selected concepts
- archive or freeze mismatched runtime paths
- build the new Aurora core as a new architecture track inside the repository
- migrate only the pieces that still fit the ontology

This is effectively a rewrite-with-continuity, not a normal refactor.

## Why Not Pure Refactor

A pure refactor is the wrong main strategy because the target system differs at the level of ontology, not just modules.

The target Aurora is not merely:

- a better memory API
- a cleaner runtime
- a more advanced retrieval layer

It is a different answer to the question of what the system is.

When ontology changes this much, deep refactor tends to preserve the wrong center.

## Why Not Throw Everything Away

A total clean-slate repository would lose:

- design lineage
- historical intuition
- existing package boundaries worth reusing
- the ability to compare old and new directions cleanly

Aurora needs a new core more than it needs a new home.

## Recommended Next Step

The next document should define a concrete module map for the rewrite track, including:

- new package boundaries
- what runtime objects exist in memory
- what gets persisted
- how `awake`, `doze`, and `sleep` execute
- what parts of the current tree are kept, frozen, or removed later
