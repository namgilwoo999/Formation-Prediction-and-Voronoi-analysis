from abc import ABC, abstractmethod
from ultralytics import YOLO
import torch
from typing import Any, Dict, List
from ultralytics.engine.results import Results
import numpy as np

class AbstractTracker(ABC):

    def __init__(self, model_path: str, conf: float = 0.1) -> None:
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf  # 신뢰도 임계값 설정
        self.cur_frame = 0  # 현재 프레임 카운터 초기화

    @abstractmethod
    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        
        pass
        
    @abstractmethod
    def track(self, detection: Results) -> dict:
        
        pass
