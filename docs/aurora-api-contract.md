# Aurora API Contract

This document defines the active HTTP contract for the current Aurora runtime.

Base behavior:

- all endpoints are JSON
- successful responses return HTTP `200`
- request/response schemas are stable unless this file changes

## Endpoints

## `GET /v1/health`

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
- `turns`: persisted interaction turn count
- `transitions`: persisted phase transition count

## `GET /v1/state`

Purpose:

- operational runtime state summary
- no explicit personality/inner-core explanation

Response:

```json
{
  "phase": "awake",
  "updated_at": 0.0,
  "self_view": 0.05,
  "world_view": 0.1,
  "openness": 0.7,
  "turns": 0,
  "memory_fragments": 0,
  "memory_traces": 0,
  "memory_associations": 0,
  "avg_salience": 0.0,
  "avg_narrative_weight": 0.0,
  "narrative_pressure": 0.0,
  "sleep_cycles": 0,
  "last_reweave_delta": 0.0,
  "relation_moments": 0,
  "relation_tone": "neutral",
  "relation_strength": 1.0,
  "transitions": 0
}
```

## `POST /v1/turn`

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
  "response_text": "I am here. ...",
  "touch_modes": ["insight"]
}
```

## `POST /v1/doze`

Purpose:

- apply a doze phase transition

Request body:

- empty

Response:

```json
{
  "phase": "doze",
  "transition_id": "pt_xxx"
}
```

## `POST /v1/sleep`

Purpose:

- apply a sleep phase transition
- run conservative reweave

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
- `touch_modes` is always an array of strings
- `/v1/state` exposes only operational counters/continuous values
- any schema change requires updating this file and relevant tests
