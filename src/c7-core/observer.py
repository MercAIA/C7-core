from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Observation:
    """
    A simple observation container used by the C7 Core.

    In v0.1 we keep it very simple:
    - raw_input: the raw token / event (mode-agnostic)
    - features: a shallow feature dictionary extracted from raw_input
    """
    raw_input: Any
    features: Dict[str, float]


class Observer:
    """
    Observer takes raw input (e.g. text, event, token)
    and converts it into an Observation with a shallow feature map.

    This implementation is intentionally minimal and modality-agnostic.
    """

    def observe(self, raw_input: Any) -> Observation:
        """
        Build an Observation from raw input.

        For v0.1 we:
        - if input is a string, use its length and punctuation as features
        - otherwise fall back to neutral features.
        """
        features: Dict[str, float] = {}

        if isinstance(raw_input, str):
            length = len(raw_input)
            features["length_norm"] = min(1.0, length / 100.0)
            features["has_question"] = 1.0 if "?" in raw_input else 0.0
            features["has_exclamation"] = 1.0 if "!" in raw_input else 0.0
        else:
            # For non-string, just mark as neutral
            features["length_norm"] = 0.0
            features["has_question"] = 0.0
            features["has_exclamation"] = 0.0

        return Observation(raw_input=raw_input, features=features)
