from .abstract_annotator import AbstractAnnotator
from utils import is_color_dark, rgb_bgr_converter

import cv2, os
import numpy as np
from scipy.spatial import Voronoi
from typing import Dict
from utils.voronoi_temporal import TemporalSmoother, owner_map_to_team_probs, team_probs_to_owner_map
from utils.voronoi_tti import compute_owner_map_tti

def rasterize_owner_map(H, W, regions, region_team_ids):
    """
    regions: List[np.ndarray (Ni,2) int/float]  # Voronoi 다각형 꼭짓점 리스트
    region_team_ids: List[int]  # 각 폴리곤에 해당하는 팀ID(0 또는 1)
    return: (H,W) np.int8  # owner_map
    """
    owner = np.full((H, W), -1, dtype=np.int8)
    for poly, tid in zip(regions, region_team_ids):
        if poly is None or len(poly) < 3:  # 안전장치
            continue
        cv2.fillPoly(owner, [np.asarray(poly, dtype=np.int32)], color=int(tid))
    return owner

class ProjectionAnnotator(AbstractAnnotator):
    def __init__(self):
        super().__init__()
        self._color2team = {}
        self._next_team_id = 0
        self._prev_xy = {}
        self._prev_frame = {}
        self._vel_xy = {}
        self.fps = 25.0
        self.save_owner_map = True
        self.output_dir = "runs/soccer_owner_maps"
        self._frame_ctr = 0

    def _draw_outline(self, frame: np.ndarray, pos: tuple, shape: str = 'circle', size: int = 10, is_dark: bool = True) -> None:
        
        outline_color = (255, 255, 255) if is_dark else (0, 0, 0)

        if shape == 'circle':
            cv2.circle(frame, (int(pos[0]), int(pos[1])), radius=size + 2, color=outline_color, thickness=2)
        elif shape == 'square':
            top_left = (int(pos[0]) - (size + 2), int(pos[1]) - (size + 2))
            bottom_right = (int(pos[0]) + (size + 2), int(pos[1]) + (size + 2))
            cv2.rectangle(frame, top_left, bottom_right, color=outline_color, thickness=2)
        elif shape == 'dashed_circle':
            dash_length, gap_length = 30, 30
            for i in range(0, 360, dash_length + gap_length):
                start_angle_rad, end_angle_rad = np.radians(i), np.radians(i + dash_length)
                start_x = int(pos[0]) + int((size + 2) * np.cos(start_angle_rad))
                start_y = int(pos[1]) + int((size + 2) * np.sin(start_angle_rad))
                end_x = int(pos[0]) + int((size + 2) * np.cos(end_angle_rad))
                end_y = int(pos[1]) + int((size + 2) * np.sin(end_angle_rad))
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color=(0, 0, 0), thickness=2)
        elif shape == 'plus':
            cv2.line(frame, (int(pos[0]) - size, int(pos[1])), (int(pos[0]) + size, int(pos[1])), color=outline_color, thickness=10)
            cv2.line(frame, (int(pos[0]), int(pos[1]) - size), (int(pos[0]), int(pos[1]) + size), color=outline_color, thickness=10)


    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        if not hasattr(self, "save_owner_map"):
            import os
            self.save_owner_map = True
            self.output_dir = getattr(self, "output_dir", "runs/soccer_owner_maps")
            os.makedirs(self.output_dir, exist_ok=True)
            self._frame_ctr = 0

            self._color2team = {}
            self._next_team_id = 0

            self._prev_xy     = {}
            self._prev_frame  = {}
            self._vel_xy      = {}
            self.fps = getattr(self, "fps", 25.0)
        

        frame = frame.copy()
        frame = self._draw_voronoi(frame, tracks)

        # 선수/볼 드로잉
        for class_name, track_data in tracks.items():
            if class_name == 'ball':
                # 볼
                for track_id, track_info in track_data.items():
                    if 'projection' not in track_info:
                        continue
                    x, y = track_info['projection'][:2]
                    cv2.circle(frame, (int(x), int(y)), radius=5, color=(0,165,255), thickness=-1)
            else:
                # 선수/골키퍼
                for track_id, track_info in track_data.items():
                    if 'projection' not in track_info:
                        continue
                    x, y = track_info['projection'][:2]
                    color = rgb_bgr_converter(track_info.get('team_color', (255,255,255)))
                    cv2.circle(frame, (int(x), int(y)), radius=10, color=color, thickness=-1)

        # 프레임 카운터 증가
        self._frame_ctr += 1
        return frame


    def _draw_voronoi(self, image: np.ndarray, tracks: Dict) -> np.ndarray:
        height, width = image.shape[:2]
        overlay = image.copy()

        # 1) 선수 포인트/색/팀ID/트랙ID 수집
        points = []
        colors = []
        team_ids_for_points = []
        track_ids_for_points = []

        for class_name in ['player', 'goalkeeper']:
            track_data = tracks.get(class_name, {})
            for track_id, track_info in track_data.items():
                if 'projection' not in track_info:
                    continue
                x, y = track_info['projection'][:2]
                points.append([float(x), float(y)])

                # 팀 색상(없으면 회색)
                bgr = rgb_bgr_converter(track_info.get('team_color', (128,128,128)))
                colors.append(bgr)

                # 팀ID 매핑(이미 있으면 재사용, 없으면 0/1 순차 부여)
                if bgr not in self._color2team and self._next_team_id < 2:
                    self._color2team[bgr] = self._next_team_id
                    self._next_team_id += 1
                team_ids_for_points.append(self._color2team.get(bgr, 0))

                # 트랙ID 기록(TTI 속도 계산용)
                track_ids_for_points.append(int(track_id))

        # 2) 바운더리 포인트 추가(무한 면 폐쇄용)
        boundary_margin = 1000
        boundary_points = [
            [-boundary_margin, -boundary_margin], [width // 2, -boundary_margin],
            [width + boundary_margin, -boundary_margin], [-boundary_margin, height // 2],
            [width + boundary_margin, height // 2], [-boundary_margin, height + boundary_margin],
            [width // 2, height + boundary_margin], [width + boundary_margin, height + boundary_margin]
        ]
        boundary_color = (128, 128, 128)
        num_players = len(points)
        points.extend(boundary_points)
        colors.extend([boundary_color] * len(boundary_points))

        # 3) Voronoi 계산 & 화면 드로잉 + owner_map 수집
        if len(points) > 2:
            pts_np = np.asarray(points, np.float32)
            vor = Voronoi(pts_np)

            # 소유맵 생성용 버퍼
            regions_for_owner = []
            region_team_ids   = []

            for region_index, region in enumerate(vor.point_region):
                region_vertices = vor.regions[region]
                if -1 in region_vertices or not region_vertices:
                    continue

                polygon_pts = [vor.vertices[i] for i in region_vertices]
                polygon = np.array(polygon_pts, np.int32).reshape((-1, 1, 2))

                color = colors[region_index] if region_index < len(colors) else boundary_color
                cv2.polylines(overlay, [polygon], isClosed=True, color=color, thickness=2)
                cv2.fillPoly(overlay, [polygon], color=color)

                # 선수 포인트(=경계 이전)만 소유맵에 반영
                if region_index < num_players and color != boundary_color:
                    team_id = int(team_ids_for_points[region_index])
                    regions_for_owner.append(np.asarray(polygon).reshape(-1, 2))
                    region_team_ids.append(team_id)

            # 4) owner_map(static / t / tti) 생성 & 저장
            if self.save_owner_map and regions_for_owner:
                owner_map_static = rasterize_owner_map(height, width, regions_for_owner, region_team_ids)
                np.save(f"{self.output_dir}/owner_{self._frame_ctr:06d}_static.npy", owner_map_static)

                # 4-1) t-Voronoi (EMA 스무딩)
                probs = owner_map_to_team_probs(owner_map_static, num_teams=2)
                if not hasattr(self, "_t_smoother"):
                    self._t_smoother = TemporalSmoother(alpha=0.6, num_teams=2)
                probs_s = self._t_smoother.smooth_team_probs(probs)
                owner_map_t = team_probs_to_owner_map(probs_s)
                np.save(f"{self.output_dir}/owner_{self._frame_ctr:06d}_t.npy", owner_map_t)

                # 4-2) TTI-Voronoi (속도 추정 → 도달시간 기반)
                #      속도는 (Δxy / Δframe)에 EMA로 살짝 평활해서 사용
                v_list = []
                for i in range(num_players):
                    tid = track_ids_for_points[i]
                    xy = np.asarray(points[i], np.float32)
                    if tid in self._prev_xy and tid in self._prev_frame:
                        dt = max(1, self._frame_ctr - self._prev_frame[tid])
                        v = (xy - self._prev_xy[tid]) / float(dt)
                    else:
                        v = np.zeros(2, np.float32)
                    # 소프트 EMA
                    if tid in self._vel_xy:
                        v = 0.6 * v + 0.4 * self._vel_xy[tid]
                    self._vel_xy[tid] = v
                    self._prev_xy[tid] = xy
                    self._prev_frame[tid] = self._frame_ctr
                    v_list.append(v)

                if num_players >= 2:
                    owner_map_tti = compute_owner_map_tti(
                        pts_xy=np.asarray(points[:num_players], np.float32),
                        v_xy=np.vstack(v_list).astype(np.float32),
                        team_ids=np.asarray(team_ids_for_points, np.int32),
                        H=height, W=width,
                        tau=0.7, vperp_scale=0.5, kappa_turn=0.15, fps=self.fps, downsample=2
                    )
                    np.save(f"{self.output_dir}/owner_{self._frame_ctr:06d}_tti.npy", owner_map_tti)

        # 드로잉 블렌딩
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        return image



    