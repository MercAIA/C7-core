from typing import Dict

from .config import (
    G_UPDATE_RATE,
    R_UPDATE_RATE,
    S_UPDATE_RATE,
    C_UPDATE_RATE,
)
from .grsc_state import GRSCState
from .observer import Observation


class DecoherenceEngine:
    """
    DecoherenceEngine updates the GRSCState based on an Observation.

    This v0.1 implementation is simple and deterministic:
    - map observation features into small deltas for G, R, S, C
    - apply those deltas with separate update rates.
    """

    def step(self, state: GRSCState, obs: Observation) -> GRSCState:
        deltas = self._features_to_deltas(obs.features)
        new_state = GRSCState(
            g=state.g + G_UPDATE_RATE * deltas["dg"],
            r=state.r + R_UPDATE_RATE * deltas["dr"],
            s=state.s + S_UPDATE_RATE * deltas["ds"],
            c=state.c + C_UPDATE_RATE * deltas["dc"],
        )
        new_state.clamp()
        return new_state

    def _features_to_deltas(self, features: Dict[str, float]) -> Dict[str, float]:
        length_norm = features.get("length_norm", 0.0)
        has_question = features.get("has_question", 0.0)
        has_exclamation = features.get("has_exclamation", 0.0)

        # Example interpretation in v0.1:
        # - longer input slightly increases G and C
        # - questions push R up and S slightly down
        # - exclamation marks shake stability (S) and reward (R)
        dg = 0.3 * length_norm
        dr = 0.4 * has_question + 0.2 * has_exclamation
        ds = -0.2 * has_question - 0.3 * has_exclamation
        dc = 0.2 * length_norm - 0.1 * has_exclamation

        return {"dg": dg, "dr": dr, "ds": ds, "dc": dc}
