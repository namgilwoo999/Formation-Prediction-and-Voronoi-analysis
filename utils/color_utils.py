from typing import Tuple

def is_color_dark(color: Tuple[int, int, int]) -> bool:
    """
    주어진 RGB 색상이 어두운지 밝은지 휘도 공식을 사용하여 확인합니다.

    Args:
        color (Tuple[int, int, int]): (R, G, B) 튜플로 표현된 RGB 색상

    Returns:
        bool: 색상이 어두우면 True, 밝으면 False
    """
    luminance = (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])  # Corrected to use the right color channels
    return luminance < 128


def rgb_bgr_converter(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    RGB 형식의 색상을 BGR 형식으로 변환하거나 그 반대로 변환합니다.

    Args:
        color (Tuple[int, int, int]): (R, G, B) 튜플로 표현된 색상

    Returns:
        Tuple[int, int, int]: (B, G, R) 튜플로 변환된 BGR 형식의 색상
    """
    return (color[2], color[1], color[0])
