from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from aurora.core_math.state import ArrivalState, isoformat_utc, parse_utc


def hours_between(later: datetime, earlier: datetime) -> float:
    return float((later - earlier).total_seconds() / 3600.0)


def advance_arrival(arrival: ArrivalState, when: datetime) -> float:
    last = parse_utc(arrival.last_event_time)
    dt_hours = max(0.0, hours_between(when, last))
    if dt_hours <= 0.0:
        return 0.0
    decay = float(np.exp(-arrival.decay_per_hour * dt_hours))
    arrival.internal_drive *= decay
    arrival.no_contact_hours += dt_hours
    arrival.last_event_time = isoformat_utc(when)
    return dt_hours


def observe_user_contact(arrival: ArrivalState, error: float) -> None:
    arrival.no_contact_hours = 0.0
    arrival.internal_drive = float(np.clip(arrival.internal_drive + 0.65 * error, 0.0, 8.0))


def observe_internal_action(arrival: ArrivalState, mass: float) -> None:
    arrival.internal_drive = float(np.clip(arrival.internal_drive + 0.3 * mass, 0.0, 8.0))


def no_contact_surprisal(arrival: ArrivalState) -> float:
    return float(arrival.base_rate * max(0.0, arrival.no_contact_hours))


def internal_intensity(arrival: ArrivalState, delta_hours: float) -> float:
    decayed_drive = arrival.internal_drive * float(np.exp(-arrival.decay_per_hour * delta_hours))
    surprise = arrival.base_rate * max(0.0, arrival.no_contact_hours + delta_hours)
    return float(max(1e-4, arrival.base_rate + 0.12 * surprise + 0.08 * decayed_drive))


def sample_next_wake(
    arrival: ArrivalState,
    now: datetime,
    rng: np.random.Generator,
    horizon_hours: float = 72.0,
) -> str:
    horizon = now + timedelta(hours=horizon_hours)
    current = now
    
    int_now = internal_intensity(arrival, 0.0)
    int_horizon = internal_intensity(arrival, hours_between(horizon, now))
    upper_bound = max(int_now, int_horizon)
    
    while current < horizon:
        upper = upper_bound
        wait_hours = rng.exponential(1.0 / max(upper, 1e-6))
        current = current + timedelta(hours=float(wait_hours))
        if current >= horizon:
            break
        actual = internal_intensity(arrival, hours_between(current, now))
        if rng.random() <= actual / upper:
            return isoformat_utc(current)
    return isoformat_utc(horizon)
