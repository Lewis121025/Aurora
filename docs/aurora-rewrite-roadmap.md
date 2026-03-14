# Aurora Rewrite Roadmap

## Decision

Aurora should be rebuilt through a staged rewrite inside the current repository.

This is not a normal refactor.
This is not a greenfield new repository either.

The correct strategy is:

- keep the repository
- keep the design lineage
- aggressively clean mismatched code
- build the new Aurora core on new package boundaries
- remove old implementation paths rather than preserving them for nostalgia

Given the current quality bar, we should prefer deletion over compatibility theater.

## Why Rewrite Instead Of Refactor

The current tree is centered on a different ontology.

It still carries assumptions closer to:

- substrate runtime mechanics
- memory CRUD surface
- sealed state substrate abstractions
- implementation compromises that no longer match the new Aurora

The target Aurora is centered on:

- existential continuity
- fragment / trace / association memory
- relation history
- `awake / doze / sleep`
- internal change that mostly remains implicit

When the center changes this much, "refactor" usually means dragging the wrong conceptual spine forward.

## Cleanup Policy

For this rewrite track, use these cleanup rules:

### Delete Immediately

Delete code when all three are true:

- it encodes the wrong ontology
- it is not a uniquely valuable implementation asset
- keeping it would increase confusion

### Freeze Briefly, Then Delete

Freeze code only when:

- it still helps us compare behavior during transition
- or it contains operational scaffolding worth extracting first

Frozen code should have a short life.
The goal is not to maintain two Auroras.

### Rewrite Cleanly

When a module category still matters but its implementation shape is wrong, rebuild it cleanly rather than trying to preserve signatures.

### No Sentimental Compatibility

Do not preserve APIs, package names, or abstractions just because they existed first.
If they no longer fit, remove them.

## Current Tree Assessment

Current top-level Python tree:

```text
aurora/
├── __main__.py
├── core_math/
├── host_runtime/
├── memory.py
├── substrate_core/
├── surface_api/
└── version.py
```

This tree should not become the final Aurora shape.

## Keep / Replace / Remove Decisions

## Keep Conceptually, Rebuild Technically

These areas still matter, but should be reimplemented under the new module map:

- runtime orchestration
- persistence
- boundary surfaces
- internal state persistence
- test discipline

In practice, these should move toward:

- `aurora/runtime/`
- `aurora/persistence/`
- `aurora/surface/`
- `aurora/evaluation/`

## Freeze Then Remove

These current areas may be temporarily left in place only while the new core comes online:

- `aurora/surface_api/`
- `aurora/host_runtime/`

Reason:

- they may still provide temporary execution surfaces
- but their naming and responsibility split do not match the final ontology cleanly

Plan:

- do not deepen them
- do not add new philosophy to them
- keep them only as a temporary shell
- remove or replace once the new runtime path is viable

## Remove As Core Concepts

These current directions should not survive as architectural centers:

- `aurora/memory.py`
- `aurora/core_math/`
- `aurora/substrate_core/`

Reason:

- they encode the wrong center of gravity for the new Aurora
- they bias the system toward substrate mechanics or storage-oriented modeling
- their names alone now pull the project toward outdated assumptions

Important nuance:

- this does not mean every useful line inside them is worthless
- it means none of them should define the new system boundary

## Immediate Repository Hygiene

Before major rewrite work, the repository should be kept strict and clean.

Immediate hygiene actions:

- remove generated cache directories
- stop adding transitional junk files
- prefer small, cleanly typed modules over giant "engine" files
- prefer deletion over dead compatibility layers
- keep docs ahead of code during ontology transition

The Python cache directories under `aurora/` have already been removed as part of this cleanup pass.

## New Target Tree

The rewrite should introduce this new package direction:

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

This new tree should be treated as the only forward direction.

## Recommended Build Order

## Phase 0: Guardrails

Tasks:

- keep the design docs authoritative
- stop feature work in the old ontology
- enforce strict code quality and typing on all new modules

Output:

- no further conceptual drift

## Phase 1: Create New Skeleton

Create empty or near-empty packages for:

- `aurora/being/`
- `aurora/memory/`
- `aurora/relation/`
- `aurora/phases/`
- `aurora/expression/`
- `aurora/persistence/`
- `aurora/runtime/`
- `aurora/surface/`
- `aurora/evaluation/`

Also create:

- explicit package `__init__.py` files
- minimal shared type modules
- clean module naming conventions

Output:

- the new Aurora has a home before it has logic

## Phase 2: Build The Non-Negotiable Core

Implement first:

- existential state object
- fragment model
- trace model
- association delta model
- relation moment model
- phase enum / phase transition model

Do not implement HTTP or CLI expansion yet.

Output:

- the ontology exists in code

## Phase 3: Build `awake`

Implement the first real runtime path:

- ingest an interaction turn
- create fragment / trace / association effects together
- update existential state
- update relation history
- persist state

Output:

- Aurora can live through one interaction in the new model

## Phase 4: Build `doze`

Implement low-pressure internal drift:

- light memory resonance
- soft salience shifts
- weak relation/context afterglow

Output:

- Aurora exists between interactions in a meaningful but lightweight way

## Phase 5: Build `sleep`

Implement deeper internal restructuring:

- narrative reweaving
- altered recall pathways
- shifted internal prominence
- subtle self/world drift effects

Output:

- Aurora no longer depends on request-time existence alone

## Phase 6: Build Expression Layer

Implement outward behavior from the new core:

- response context assembly
- silence / refusal path
- gentle initial voice floor
- relation-shaped expression drift

Output:

- Aurora can outwardly behave from the new ontology

## Phase 7: Reconnect Surface

Only after the new runtime works internally:

- replace CLI entry path
- replace HTTP path
- expose sleep controls
- preserve only minimal health/inspection interfaces

Output:

- the old surface stops being the system's real center

## Phase 8: Delete Old Paths

Once the new path is viable:

- remove `aurora/memory.py`
- remove old `surface_api` path
- remove old `host_runtime` path
- remove old `substrate_core` path
- remove old `core_math` path
- simplify entrypoints and dependency set

Output:

- one Aurora, one ontology, one code path

## Reuse Guidance

Safe to reuse selectively:

- configuration loading patterns
- packaging basics
- testing setup discipline
- some persistence bootstrapping ideas

Unsafe to reuse without deep scrutiny:

- ontology-bearing core models
- current engine abstractions
- storage formats shaped around old substrate assumptions
- API contracts shaped around CRUD or old runtime semantics

If in doubt, rewrite.

## Code Quality Standard For The Rewrite

Given the repository hygiene requirement, new Aurora code should follow these rules:

- explicit package boundaries
- small modules
- strict typing
- low magic
- no pseudo-clever abstractions
- no giant god-engine files
- no hidden mutable globals
- no dead compatibility layers
- no framework-heavy architecture for its own sake

Preferred style:

- plain Python
- clean datamodels
- intentional state transitions
- testable pure-ish logic where possible
- infrastructure that stays secondary to ontology

## Documentation Order

During the rewrite, docs should proceed in this order:

1. principles
2. blueprint
3. module map
4. roadmap
5. runtime object model
6. persistence schema
7. first implementation milestone plan

This keeps code from outrunning ontology.

## Recommended Immediate Next Build Step

The next actual code step should be:

- create the new package skeleton
- add the first shared runtime object models
- wire a minimal `awake`-only new runtime path

Do not start by rewriting the current HTTP app.
Do not start by optimizing storage.
Do not start by preserving old abstractions.

Start by making the new Aurora exist in code.
