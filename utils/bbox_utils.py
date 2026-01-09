from typing import Tuple
import math

def is_valid_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return False
    for coord in bbox:
        if not isinstance(coord, (int, float)) or math.isnan(coord):
            return False
    return True

def get_bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    if not is_valid_bbox(bbox):
        print(f"[WARN] Invalid bbox: {bbox}")
        return None
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox: Tuple[float, float, float, float]) -> float:
    """
    바운딩 박스의 너비를 계산합니다.

    Args:
        bbox (Tuple[float, float, float, float]): (x1, y1, x2, y2)로 정의된 바운딩 박스

    Returns:
        float: 바운딩 박스의 너비
    """
    x1, _, x2, _ = bbox
    return x2 - x1

def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    두 점 사이의 유클리드 거리를 계산합니다.

    Args:
        p1 (Tuple[float, float]): 첫 번째 점 (x1, y1)
        p2 (Tuple[float, float]): 두 번째 점 (x2, y2)

    Returns:
        float: 두 점 사이의 거리
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def point_coord_diff(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """
    두 점 사이의 좌표 차이를 계산합니다.

    Args:
        p1 (Tuple[float, float]): 첫 번째 점 (x1, y1)
        p2 (Tuple[float, float]): 두 번째 점 (x2, y2)

    Returns:
        Tuple[float, float]: 두 점 사이의 차이 (dx, dy)
    """
    return p1[0] - p2[0], p1[1] - p2[1]

def get_feet_pos(bbox: Tuple[float, float, float, float]) -> Tuple[float, int]:
    """
    바운딩 박스로부터 발 위치를 계산합니다.

    Args:
        bbox (Tuple[float, float, float, float]): (x1, y1, x2, y2)로 정의된 바운딩 박스

    Returns:
        Tuple[float, int]: 발 위치 (feet_x, feet_y), feet_y는 정수로 반올림됨
    """
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, int(y2)

def measure_distance(p1, p2):
    if p1 is None or p2 is None:
        print(f"[WARN] Invalid points for distance: {p1}, {p2}")
        return math.inf
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
