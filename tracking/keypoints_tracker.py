from tracking.abstract_tracker import AbstractTracker

import cv2
import supervision as sv
from typing import List
from ultralytics.engine.results import Results
import numpy as np

class KeypointsTracker(AbstractTracker):
    

    def __init__(self, model_path: str, conf: float = 0.1, kp_conf: float = 0.7) -> None:
        
        super().__init__(model_path, conf)  # Call the Tracker base class constructor
        self.kp_conf = kp_conf  # Keypoint Confidence Threshold
        self.tracks = []  # Initialize tracks list
        self.cur_frame = 0  # Frame counter initialization
        self.original_size = (1920, 1080)  # Original resolution (1920x1080)
        self.scale_x = self.original_size[0] / 1280
        self.scale_y = self.original_size[1] / 1280

    def detect(self, frames: List[np.ndarray]) -> List[Results]:
        
        # 각 프레임의 탐지 전 대비 조정
        contrast_adjusted_frames = [self._preprocess_frame(frame) for frame in frames]

        # YOLOv8의 배치 예측 방법 사용
        detections = self.model.predict(contrast_adjusted_frames, conf=self.conf)
        return detections

    def track(self, detection: Results) -> dict:
        
        detection = sv.KeyPoints.from_ultralytics(detection)
        
        # 확인 
        if not detection:
            return {}

        # xy 좌표, 신뢰도, 키포인트 개수 추출
        xy = detection.xy[0]  # Shape: (32, 2), assuming there are 32 keypoints
        confidence = detection.confidence[0]  # Shape: (32,), confidence values

        # 임계값보다 높은 신뢰도를 가진 키포인트 맵 생성
        filtered_keypoints = {
            i: (coords[0] * self.scale_x, coords[1] * self.scale_y)  # i is the key (index), (x, y) are the values
            for i, (coords, conf) in enumerate(zip(xy, confidence))
            if conf > self.kp_conf
            and 0 <= coords[0] <= 1280  # 확인 if x is within bounds
            and 0 <= coords[1] <= 1280  # 확인 if y is within bounds
        }

        self.tracks.append(detection)
        self.cur_frame += 1

        return filtered_keypoints

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        
        # 대비 조정
        frame = self._adjust_contrast(frame)
        
        # 프레임을 1280x1280으로 리사이즈
        resized_frame = cv2.resize(frame, (1280, 1280))

        return resized_frame
    
    def _adjust_contrast(self, frame: np.ndarray) -> np.ndarray:
        
        # 프레임이 컬러(3채널)인지 확인. 그렇다면 히스토그램 평활화를 위해 그레이스케일로 변환
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # YUV 색공간으로 변환
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            
            # Y 채널(휘도)에 히스토그램 평활화 적용
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            
            # BGR 형식으로 역변환
            frame_equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # 프레임이 이미 그레이스케일이면 히스토그램 평활화 직접 적용
            frame_equalized = cv2.equalizeHist(frame)

        return frame_equalized
