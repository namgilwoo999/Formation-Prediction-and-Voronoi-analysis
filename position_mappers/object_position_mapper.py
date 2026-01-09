from .abstract_mapper import AbstractMapper
from .homography import get_homography, apply_homography, HomographySmoother
from utils.bbox_utils import get_feet_pos

import numpy as np

class ObjectPositionMapper(AbstractMapper):
    

    def __init__(self, top_down_keypoints: np.ndarray, alpha: float = 0.9) -> None:
        
        super().__init__()
        self.top_down_keypoints = top_down_keypoints
        self.homography_smoother = HomographySmoother(alpha=alpha)

    def map(self, detection: dict) -> dict:
        
        detection = detection.copy()
        
        keypoints = detection['keypoints']
        object_data = detection['object']

        if not keypoints or not object_data:
            return detection

        # keypoints 수 체크(최소 4개)
        if len(keypoints) < 4:
            print(f"Homography skipped: insufficient keypoints ({len(keypoints)})")
            detection['homography_skipped'] = True
            return detection

        H = get_homography(keypoints, self.top_down_keypoints)
        
        if H is None:
            print("Homography skipped: cv2.findHomography returned None")
            detection['homography_skipped'] = True
            return detection
        
        smoothed_H = self.homography_smoother.smooth(H)  # 호모그래피 행렬에 스무딩 적용

        for _, object_info in object_data.items():
            for _, track_info in object_info.items():
                bbox = track_info['bbox']
                feet_pos = get_feet_pos(bbox)  # 발 위치 가져오기
                projected_pos = apply_homography(feet_pos, smoothed_H)  # 스무딩된 호모그래피 함수 적용
                track_info['projection'] = projected_pos  # 추적 정보에 투영 좌표 추가

        return detection