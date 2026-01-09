from abc import ABC, abstractmethod
import numpy as np
from typing import List

class AbstractVideoProcessor(ABC):

    @abstractmethod
    def process(self, frames: List[np.ndarray], fps: float = 1e-6) -> List[np.ndarray]:
        pass
        