from .abstract_annotator import AbstractAnnotator

import cv2
import numpy as np
from typing import Dict

class KeypointsAnnotator(AbstractAnnotator):

    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        
         
        frame = frame.copy()

        for kp_id, (x, y) in tracks.items():
            # 각 키포인트에 대해 반지름 5, 초록색(0, 255, 0)으로 원 그리기
            cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            # 점 옆에 키포인트 ID 표시
            cv2.putText(frame, str(kp_id), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame
