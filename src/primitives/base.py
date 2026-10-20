from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class DistributionPrimitive(ABC):
    """
    Base execution contract for all distribution primitives.
    """

    @abstractmethod
    def generate(
        self,
        rng: np.random.Generator,
        size: int,
        params: Dict[str, Any],
    ) -> np.ndarray:
        pass

class RelationshipPrimitive(ABC):
    """
    Base execution contract for all relationship primitives.
    """

    @abstractmethod
    def generate(
        self,
        input: Dict[str, Any],
        params: Dict[str, Any],
    ) -> np.ndarray:
        pass