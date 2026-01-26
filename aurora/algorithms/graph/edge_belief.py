"""
AURORA Edge Belief
===================

Probabilistic edge strength using Beta posterior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from aurora.utils.time_utils import now_ts


@dataclass
class EdgeBelief:
    """Edge helpfulness posterior: Beta(a,b).

    Models the probability that traversing this edge is helpful,
    using a conjugate Beta-Bernoulli model.

    Attributes:
        edge_type: Type of relationship this edge represents
        a: Alpha parameter (successes + 1)
        b: Beta parameter (failures + 1)
        use_count: Number of times this edge has been used
        last_used_ts: Timestamp of last use
    """

    edge_type: str
    a: float = 1.0
    b: float = 1.0
    use_count: int = 0
    last_used_ts: float = field(default_factory=now_ts)

    def mean(self) -> float:
        """Get expected helpfulness (Beta posterior mean).

        Returns:
            Expected probability of helpfulness in (0, 1)
        """
        return self.a / (self.a + self.b)

    def update(self, success: bool) -> None:
        """Update belief based on outcome.

        Args:
            success: Whether traversing this edge was helpful
        """
        self.use_count += 1
        self.last_used_ts = now_ts()
        if success:
            self.a += 1.0
        else:
            self.b += 1.0

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "edge_type": self.edge_type,
            "a": self.a,
            "b": self.b,
            "use_count": self.use_count,
            "last_used_ts": self.last_used_ts,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "EdgeBelief":
        """Reconstruct from state dict."""
        return cls(
            edge_type=d["edge_type"],
            a=d["a"],
            b=d["b"],
            use_count=d["use_count"],
            last_used_ts=d["last_used_ts"],
        )
