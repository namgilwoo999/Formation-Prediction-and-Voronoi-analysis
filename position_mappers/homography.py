import cv2
import numpy as np
from typing import Tuple, List

class HomographySmoother:
    def __init__(self, alpha: float = 0.9):
        
        self.alpha = alpha
        self.smoothed_H = None

    def smooth(self, current_H: np.ndarray) -> np.ndarray:
        
        if self.smoothed_H is None:
            # 첫 번째 호모그래피 행렬로 초기화
            self.smoothed_H = current_H
        else:
            # 지수 평활화
            self.smoothed_H = self.alpha * current_H + (1 - self.alpha) * self.smoothed_H

        return self.smoothed_H

def get_homography(keypoints: dict, top_down_keypoints: np.ndarray) -> np.ndarray:
    
    kps: List[Tuple[float, float]] = []
    proj_kps: List[Tuple[float, float]] = []

    for key in keypoints.keys():
        kps.append(keypoints[key])
        proj_kps.append(top_down_keypoints[key])

    if len(kps) < 4 or len(proj_kps) < 4:
        print(f"[Warning] Not enough keypoints for homography! (detected={len(kps)})")
        return None

    def _compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
        
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        h, _ = cv2.findHomography(src_points, dst_points)

        return h.astype(np.float32)

    H = _compute_homography(np.array(kps), np.array(proj_kps))

    return H


def apply_homography(pos: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    
    x, y = pos
    pos_homogeneous = np.array([x, y, 1])
    projected_pos = np.dot(H, pos_homogeneous)
    projected_pos /= projected_pos[2]

    return projected_pos[0], projected_pos[1]