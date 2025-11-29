from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class GRSCState:
    """
    GRSCState represents the internal state of the C7 core:

    G: Grounding
    R: Relevance / Reward
    S: Stability
    C: Coherence

    In this v0.1 implementation, these are simple floats expected
    to stay within [0, 1].
    """
    g: float
    r: float
    s: float
    c: float

    def clamp(self) -> None:
        """Clamp all values to the [0, 1] range."""
        self.g = max(0.0, min(1.0, self.g))
        self.r = max(0.0, min(1.0, self.r))
        self.s = max(0.0, min(1.0, self.s))
        self.c = max(0.0, min(1.0, self.c))

    def as_dict(self) -> Dict[str, float]:
        """Return a dictionary representation of the state."""
        return asdict(self)
