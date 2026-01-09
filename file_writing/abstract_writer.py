from abc import ABC, abstractmethod
from typing import Any

class AbstractWriter(ABC):
    @abstractmethod
    def write(self, filename: str, data: Any) -> None:
        pass
        
    @abstractmethod
    def _make_serializable(self, obj: Any) -> Any:
        pass