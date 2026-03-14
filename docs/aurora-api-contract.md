# Aurora API Contract

This document defines the active HTTP contract for the current Aurora runtime.

Base behavior:

- all endpoints are JSON
- successful responses return HTTP `200`
- schema changes require updating this file and unit tests

## Endpoints

## `GET /health`

Purpose:

- minimal runtime health and counters

Response:

```json
{
  "status": "ok",
  "phase": "awake",
  "turns": 0,
  "transitions": 0
}
```

Fields:

- `status`: string
- `phase`: one of `awake | doze | sleep`
- `turns`: persisted user turn count
- `transitions`: persisted phase transition count

## `GET /state`

Purpose:

- expose a runtime projection instead of canonical ontology writes

Response:

```json
{
  "phase": "awake",
  "sleep_need": 0.0,
  "active_relation_ids": [],
  "pending_sleep_relation_ids": [],
  "active_knot_ids": [],
  "anchor_thread_ids": [],
  "turns": 0,
  "memory_fragments": 0,
  "memory_traces": 0,
  "memory_associations": 0,
  "memory_threads": 0,
  "memory_knots": 0,
  "relation_formations": 0,
  "relation_moments": 0,
  "sleep_cycles": 0,
  "transitions": 0
}
```

## `POST /turn`

Purpose:

- run one awake interaction turn

Request:

```json
{
  "session_id": "default",
  "text": "I learned something important today"
}
```

Response:

```json
{
  "turn_id": "turn_xxx",
  "response_text": "I am staying with what is present, without flattening it.",
  "aurora_move": "witness",
  "dominant_channels": ["recognition", "coherence"]
}
```

Rules:

- `aurora_move` is one of `approach | withhold | boundary | repair | silence | witness`
- `dominant_channels` is an array of channel names

## `POST /doze`

Purpose:

- run doze consolidation pass

Request body:

- empty

Response:

```json
{
  "phase": "doze",
  "transition_id": "pt_xxx"
}
```

## `POST /sleep`

Purpose:

- run sleep reweave pass

Request body:

- empty

Response:

```json
{
  "phase": "sleep",
  "transition_id": "pt_xxx"
}
```

## Contract Rules

- `phase` values always map to `awake | doze | sleep`
- `transition_id` is always returned for phase endpoints
- `/state` fields are projections; canonical ontology writes are internal
