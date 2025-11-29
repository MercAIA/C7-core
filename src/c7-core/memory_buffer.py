from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

from .grsc_state import GRSCState
from .observer import Observation
from .config import MEMORY_BUFFER_SIZE


@dataclass
class MemoryItem:
    """
    A single memory item: (state_before, observation, state_after)
    """
    state_before: GRSCState
    observation: Observation
    state_after: GRSCState


class MemoryBuffer:
    """
    A fixed-size memory buffer for C7 Core transitions.
    """

    def __init__(self, capacity: int = MEMORY_BUFFER_SIZE) -> None:
        self._buffer: Deque[MemoryItem] = deque(maxlen=capacity)

    def add(self, item: MemoryItem) -> None:
        self._buffer.append(item)

    def to_list(self) -> List[MemoryItem]:
        return list(self._buffer)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._buffer)

    def summary(self) -> Tuple[int, float]:
        """
        Return:
        - number of items
        - mean coherence value over stored states_after
        """
        if not self._buffer:
            return 0, 0.0

        total_c = sum(item.state_after.c for item in self._buffer)
        return len(self._buffer), total_c / len(self._buffer)
