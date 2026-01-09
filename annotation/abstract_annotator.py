from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class AbstractAnnotator(ABC):

    @abstractmethod
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        pass
        