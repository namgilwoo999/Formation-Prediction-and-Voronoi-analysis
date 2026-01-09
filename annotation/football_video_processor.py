from .abstract_annotator import AbstractAnnotator
from .abstract_video_processor import AbstractVideoProcessor
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from .frame_number_annotator import FrameNumberAnnotator
from file_writing import TracksJsonWriter
from tracking import KeypointsTracker
from tracking.tracker import Tracker
from tracking.team_consistency_checker import TeamConsistencyChecker
from ball_to_player_assignment.player_ball_assigner import PlayerBallAssigner
from utils import rgb_bgr_converter
from team_assigner.team_assigner import TeamAssigner

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

def batch_iterable(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
            
class FootballVideoProcessor(AbstractAnnotator, AbstractVideoProcessor):

    def __init__(self, obj_tracker: Tracker, kp_tracker: KeypointsTracker, 
                 team_assigner: TeamAssigner, ball_to_player_assigner: PlayerBallAssigner, 
                 top_down_keypoints: np.ndarray, field_img_path: str, 
                 save_tracks_dir: Optional[str] = None, draw_frame_num: bool = True) -> None:

        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.team_assigner = team_assigner
        self.ball_to_player_assigner = ball_to_player_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        self.draw_frame_num = draw_frame_num
        
        # 팀 일관성 체커 초기화 (초반 감지 강화)
        self.team_checker = TeamConsistencyChecker(
            history_window=30,           # 30프레임 히스토리
            confidence_threshold=0.7,    # 70% 신뢰도
            color_change_threshold=40    # 색상 변화 임계값 (낮을수록 민감)
        )
        
        if self.draw_frame_num:
            self.frame_num_annotator = FrameNumberAnnotator() 

        if save_tracks_dir:
            self.save_tracks_dir = save_tracks_dir
            self.writer = TracksJsonWriter(save_tracks_dir)
        
        field_image = cv2.imread(field_img_path)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)
        
        self.frame_num = 0
        self.field_image = field_image

    def process(self, frame_gen, fps: float = 1e-6, kp_batch_size=256, buffer_size=300):
        """
        버퍼 기반 스트리밍 처리 방식 (개선된 버전 - 트래킹 연속성 유지 + 프레임 중복 방지)
        
        Args:
            frame_gen: 프레임 제너레이터
            fps: 비디오 FPS
            kp_batch_size: 키포인트 배치 크기
            buffer_size: 객체 트래킹 버퍼 크기 (기본 300 프레임)
        """
        self.cur_fps = max(fps, 1e-6)
        self.frame_num = 0
        first_team_assigned = False
        
        # 프레임 버퍼
        frame_buffer = []
        processed_count = 0
        already_yielded = 0  # 이미 yield한 프레임 수 추적
        
        print(f"[INFO] 버퍼 기반 처리 시작 (최대 버퍼 크기: {buffer_size} 프레임)")
        
        def process_buffer(frames, is_final=False):
            #버퍼 처리 함수
            if not frames:
                return
                
            buffer_info = f"최종 버퍼" if is_final else f"버퍼"
            print(f"[INFO] {buffer_info} 처리 중... ({len(frames)} 프레임)")
            
            # 1. 버퍼의 모든 프레임에 대해 객체 트래킹 수행 (연속성 보장)
            all_obj_tracks = self.obj_tracker.get_object_tracks(frames)
            
            # 2. 키포인트는 배치로 처리
            all_kp_tracks = []
            for kp_batch in batch_iterable(frames, kp_batch_size):
                kp_detections_batch = self.kp_tracker.detect(kp_batch)
                for kp_detection in kp_detections_batch:
                    kp_track = self.kp_tracker.track(kp_detection)
                    all_kp_tracks.append(kp_track)
            
            # 3. 각 프레임에 대해 어노테이션 수행
            for idx in range(len(frames)):
                frame = frames[idx]
                
                # 객체 트랙 추출
                obj_tracks = {
                    'player':     all_obj_tracks['players'][idx],
                    'referee':    all_obj_tracks['referees'][idx],
                    'goalkeeper': all_obj_tracks['goalkeepers'][idx],
                    'ball':       all_obj_tracks['ball'][idx]
                }
                
                # 키포인트 트랙
                kp_tracks = all_kp_tracks[idx] if idx < len(all_kp_tracks) else {}
                
                # 팀 컬러 할당 (첫 프레임에만)
                nonlocal first_team_assigned
                if not first_team_assigned and obj_tracks['player']:
                    self.team_assigner.assign_team_color(frame, obj_tracks['player'])
                    first_team_assigned = True
                
                # 팀 ID 및 색상 할당
                for pid, info in obj_tracks['player'].items():
                    tid = self.team_assigner.get_player_team(frame, info['bbox'], pid)
                    info['team_id'] = tid
                    info['team_color'] = self.team_assigner.team_colors[tid].tolist()
                
                for gid, info in obj_tracks['goalkeeper'].items():
                    tid = self.team_assigner.get_player_team(frame, info['bbox'], gid)
                    info['team_id'] = tid
                    info['team_color'] = self.team_assigner.team_colors[tid].tolist()
                
                # 팀 일관성 체크 및 수정 (프레임 이미지 전달)
                obj_tracks['player'] = self.team_checker.update(obj_tracks['player'], frame)
                # goalkeeper도 체크
                obj_tracks['goalkeeper'] = self.team_checker.update(obj_tracks['goalkeeper'], frame)
                
                # 팀 색상 재할당 (수정된 경우)
                for pid, info in obj_tracks['player'].items():
                    info['team_color'] = self.team_assigner.team_colors[info['team_id']].tolist()
                for gid, info in obj_tracks['goalkeeper'].items():
                    info['team_color'] = self.team_assigner.team_colors[info['team_id']].tolist()
                
                # 트랙 통합
                all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}
                all_tracks = self.obj_mapper.map(all_tracks)
                
                # 볼 소유자 할당
                ball_items = all_tracks['object']['ball']
                if ball_items:
                    _, ball_info = next(iter(ball_items.items()))
                    ball_bbox = ball_info['bbox']
                    assigned_pid = self.ball_to_player_assigner.assign_ball_to_player(
                        all_tracks['object']['player'],
                        ball_bbox
                    )
                    for pid, info in all_tracks['object']['player'].items():
                        info['has_ball'] = (pid == assigned_pid)
                else:
                    for pid, info in all_tracks['object']['player'].items():
                        info['has_ball'] = False
                
                # 트랙 저장
                if self.save_tracks_dir:
                    self._save_tracks(all_tracks)
                
                # 어노테이션
                annotated_frame = self.annotate(frame, all_tracks)
                yield annotated_frame
                
                self.frame_num += 1
                nonlocal processed_count
                processed_count += 1
        
        # 메인 처리 루프
        for frame in frame_gen:
            if frame is None:
                break
            frame_buffer.append(frame)
            
            # 버퍼가 가득 찼을 때 처리
            if len(frame_buffer) >= buffer_size:
                # 처리 (오버랩된 프레임은 건너뛰고 yield)
                frames_to_yield = 0
                for annotated_frame in process_buffer(frame_buffer):
                    # 오버랩으로 인한 중복 프레임은 건너뛰기
                    if frames_to_yield >= already_yielded:
                        yield annotated_frame
                    frames_to_yield += 1
                
                # 트래킹 연속성을 위해 작은 오버랩 유지 (10 프레임)
                overlap = 10
                already_yielded = overlap  # 다음 버퍼에서 건너뛸 프레임 수
                frame_buffer = frame_buffer[-overlap:] if overlap > 0 else []
                print(f"[INFO] {processed_count} 프레임 처리 완료")
        
        # 제너레이터 종료 후 남은 프레임 처리
        if frame_buffer:
            frames_to_yield = 0
            for annotated_frame in process_buffer(frame_buffer, is_final=True):
                # 마지막 버퍼에서도 오버랩 프레임 건너뛰기
                if frames_to_yield >= already_yielded:
                    yield annotated_frame
                frames_to_yield += 1
            print(f"[INFO] 총 {processed_count} 프레임 처리 완료")
    
    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        #프레임에 어노테이션 추가
        
        # 프레임 번호 표시
        if self.draw_frame_num:
            frame = self.frame_num_annotator.annotate(frame, {'frame_num': self.frame_num})
        
        # 키포인트 및 객체 어노테이션
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        
        # 필드 투영 어노테이션 (field_image에 적용)
        projection_frame = self.projection_annotator.annotate(self.field_image.copy(), tracks['object'])
        
        # 메인 프레임과 투영 프레임 결합
        combined_frame = self._combine_frame_projection(frame, projection_frame)
        
        return combined_frame
    
    def _combine_frame_projection(self, frame: np.ndarray, projection_frame: np.ndarray) -> np.ndarray:
        #메인 프레임과 투영 프레임을 결합
        
        # 프레임 크기
        h_frame, w_frame, _ = frame.shape
        canvas_width, canvas_height = w_frame, h_frame
        h_proj, w_proj, _ = projection_frame.shape
        
        # 투영 프레임을 70% 크기로 조정
        scale_proj = 0.7
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(projection_frame, (new_w_proj, new_h_proj))
        
        # 캔버스 생성
        combined = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 메인 프레임 복사
        combined[:h_frame, :w_frame] = frame
        
        # 투영 프레임을 중앙 하단에 배치
        x_offset = (canvas_width - new_w_proj) // 2
        y_offset = canvas_height - new_h_proj - 25
        
        # 경계 체크
        if y_offset >= 0 and x_offset >= 0:
            combined[y_offset:y_offset+new_h_proj, x_offset:x_offset+new_w_proj] = projection_resized
        
        return combined
    
    def _save_tracks(self, all_tracks: Dict) -> None:
        #트랙 정보를 JSON 파일로 저장
        if hasattr(self, 'writer'):
            # 객체 트랙과 키포인트 트랙을 각각 저장
            if 'object' in all_tracks:
                self.writer.write(self.writer.get_object_tracks_path(), all_tracks['object'])
            if 'keypoints' in all_tracks:
                self.writer.write(self.writer.get_keypoints_tracks_path(), all_tracks['keypoints'])
