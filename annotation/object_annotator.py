from .abstract_annotator import AbstractAnnotator
from utils import get_bbox_width, get_bbox_center, is_color_dark, rgb_bgr_converter

import cv2
import numpy as np
from typing import Dict, Tuple

class ObjectAnnotator(AbstractAnnotator):
    

    def __init__(self, ball_annotation_color: Tuple[int, int, int] = (48, 48, 190), 
                 referee_annotation_color: Tuple[int, int, int] = (40, 40, 40)) -> None:
        self.ball_annotation_color = ball_annotation_color
        self.referee_annotation_color = referee_annotation_color
        super().__init__()
        
    
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:

        frame = frame.copy()

        # 추적된 객체들을 반복
        for track in tracks:
            for track_id, item in tracks[track].items():
                # TeamAssigner가 붙여준 team_color 을 우선 사용
                # (info['team_color']는 [R,G,B] 리스트이니, BGR 튜플로 변환)
                if 'team_color' in item:
                    color = rgb_bgr_converter(item['team_color'])
                else:
                    # 볼·심판 등은 기존 default
                    color = (0, 255, 255) if track=='ball' else (255,255,0)

                # 객체 타입에 따라 주석 추가
                if track == 'ball':
                    frame = self.draw_triangle(frame, item['bbox'], self.ball_annotation_color)
                elif track == 'referee':
                    frame = self.draw_ellipse(frame, item['bbox'], self.referee_annotation_color, track_id, -1, track)
                else:
                    frame = self.draw_ellipse(frame, item['bbox'], color, track_id, -1, track)

                    # 선수가 공을 가지고 있으면 삼각형으로 표시
                    if 'has_ball' in item and item['has_ball']:
                        frame = self.draw_triangle(frame, item['bbox'], color)

        return frame
    

    def draw_triangle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                      color: Tuple[int, int, int]) -> np.ndarray:

        # 공 색상의 밝기에 따라 삼각형 윤곽선 색상 조정
        color2 = (255, 255, 255) if is_color_dark(color) else (0, 0, 0)

        # 삼각형의 x, y 위치 가져오기
        y = int(bbox[1])
        x, _ = get_bbox_center(bbox)
        x = int(x)

        # 삼각형 포인트 정의
        points = np.array([
            [x, y],
            [x - 8, y - 18],
            [x + 8, y - 18]
        ])

        # 채워진 삼각형 그리기
        cv2.drawContours(frame, [points], 0, color, cv2.FILLED)
        # 삼각형 윤곽선 그리기
        cv2.drawContours(frame, [points], 0, color2, 1)

        return frame
    
    def _draw_double_ellipse(self, frame: np.ndarray, x: int, y: int, w: int, color: Tuple[int, int, int]) -> None:

        size_decrement = 5  # 두 번째 타원의 크기 축소
        # 두 개의 동심 타원 그리기
        for i in range(2):
            cv2.ellipse(frame, center=(x, y), axes=(w - i * size_decrement, 20 - i * size_decrement),
                        angle=0, startAngle=-30, endAngle=240, color=color, thickness=2, lineType=cv2.LINE_AA)
            
    
    def _draw_dashed_ellipse(self, frame: np.ndarray, x: int, y: int, w: int, color: Tuple[int, int, int]) -> None:
        
        dash_length = 15  # 각 대시의 길이
        total_angle = 270  # 커버할 총 각도

        # 대시와 간격을 교대로 하여 점선 그리기
        for angle in range(-30, total_angle, dash_length * 2):
            cv2.ellipse(frame, center=(x, y), axes=(w, 20), angle=0,
                        startAngle=angle, endAngle=angle + dash_length, color=color, thickness=2, lineType=cv2.LINE_AA)

       

    def draw_ellipse(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                     color: Tuple[int, int, int], track_id: int, _speed: float, obj_cls: str = 'player') -> np.ndarray:
        
        # 기본 색상의 밝기에 따라 텍스트 및 ID 색상 조정
        color2 = (255, 255, 255) if is_color_dark(color) else (0, 0, 0)

        # 타원의 위치와 크기 가져오기
        y = int(bbox[3])
        x, _ = get_bbox_center(bbox)
        x = int(x)
        w = int(get_bbox_width(bbox))

        # 객체 클래스에 따라 타원 스타일 결정
        if obj_cls == 'referee':
            self._draw_dashed_ellipse(frame, x, y, w, color)
        elif obj_cls == 'goalkeeper':
            self._draw_double_ellipse(frame, x, y, w, color)
        else:
            # 선수용 표준 타원
            cv2.ellipse(frame, center=(x, y), axes=(w, 20), angle=0, startAngle=-30, endAngle=240, color=color,
                        thickness=2, lineType=cv2.LINE_AA)

        # 객체 ID를 표시할 작은 사각형을 타원 아래에 그리기
        y = int(bbox[3]) + 10
        h, w = 10, 20
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), color, cv2.FILLED)

        # 트랙 ID 표시
        x1 = x - len(str(track_id)) * 5
        cv2.putText(frame, text=f"{track_id}", org=(x1, y + h // 2), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                    color=color2, thickness=2)

        return frame
