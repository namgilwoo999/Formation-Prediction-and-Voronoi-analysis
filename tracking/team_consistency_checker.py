"""
팀 색상 일관성 기반 ID 재할당 모듈 (초반 감지 강화)
선수가 겹쳤을 때 ID가 스왑되는 문제를 해결합니다.
"""
import numpy as np
from collections import defaultdict
import cv2

class TeamConsistencyChecker:
    """
    선수의 팀 할당과 색상이 갑자기 바뀌는 것을 감지하고 수정합니다.
    """
    
    def __init__(self, history_window=30, confidence_threshold=0.7, color_change_threshold=50):
        """
        Args:
            history_window: 히스토리를 몇 프레임 동안 유지할지
            confidence_threshold: 팀 변경을 확정하기 위한 임계값
            color_change_threshold: 색상 변화 임계값 (BGR 거리)
        """
        self.history_window = history_window
        self.confidence_threshold = confidence_threshold
        self.color_change_threshold = color_change_threshold
        
        # player_id -> [team_id_history]
        self.team_history = defaultdict(list)
        
        # player_id -> dominant_team_id
        self.player_teams = {}
        
        # player_id -> initial_color (BGR)
        self.player_colors = {}
        
        # player_id -> frame_first_seen
        self.first_seen_frame = {}
        
        self.current_frame = 0
    
    def _get_player_color(self, frame, bbox):
        """선수의 대표 색상 추출 (상반신 영역)"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # 경계 체크
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(x1+1, min(x2, w))
        y1 = max(0, min(y1, h-1))
        y2 = max(y1+1, min(y2, h))
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        # 상반신만 (유니폼)
        top_half = roi[:roi.shape[0]//2, :]
        if top_half.size == 0:
            return None
        
        # 평균 색상
        mean_color = cv2.mean(top_half)[:3]
        return np.array(mean_color)
    
    def update(self, player_tracks, frame=None):
        """
        현재 프레임의 선수 트랙 정보를 받아서 팀 히스토리를 업데이트
        
        Args:
            player_tracks: {player_id: {'team_id': 0 or 1, 'bbox': [...], ...}}
            frame: 현재 프레임 이미지 (색상 체크용, 선택사항)
        
        Returns:
            corrected_tracks: 수정된 트랙 정보
        """
        self.current_frame += 1
        corrected_tracks = {}
        
        for player_id, info in player_tracks.items():
            current_team = info.get('team_id')
            bbox = info.get('bbox')
            
            if current_team is None:
                corrected_tracks[player_id] = info
                continue
            
            # 처음 보는 선수면 색상 저장
            if player_id not in self.first_seen_frame:
                self.first_seen_frame[player_id] = self.current_frame
                if frame is not None and bbox is not None:
                    color = self._get_player_color(frame, bbox)
                    if color is not None:
                        self.player_colors[player_id] = color
                        print(f"[COLOR_INIT] Player {player_id} at frame {self.current_frame}: team={current_team}, color={color}")
            
            # 히스토리에 추가
            self.team_history[player_id].append(current_team)
            
            # 윈도우 크기 제한
            if len(self.team_history[player_id]) > self.history_window:
                self.team_history[player_id].pop(0)
            
            # 초반 5프레임 이내: 색상 기반 검증
            frames_since_first_seen = self.current_frame - self.first_seen_frame.get(player_id, 0)
            if frames_since_first_seen <= 5 and frame is not None and bbox is not None:
                if player_id in self.player_colors:
                    current_color = self._get_player_color(frame, bbox)
                    if current_color is not None:
                        initial_color = self.player_colors[player_id]
                        color_distance = np.linalg.norm(current_color - initial_color)
                        
                        # 색상이 크게 바뀌었으면 팀도 의심
                        if color_distance > self.color_change_threshold:
                            # 팀을 반대로 뒤집기
                            corrected_team = 1 - current_team
                            print(f"[COLOR_FIX] Player {player_id} at frame {self.current_frame}: "
                                  f"team {current_team}→{corrected_team} (color dist: {color_distance:.1f})")
                            info = info.copy()
                            info['team_id'] = corrected_team
                            current_team = corrected_team
                            
                            # 색상도 업데이트
                            self.player_colors[player_id] = current_color
            
            # 지배적인 팀 계산 (3프레임 이상)
            history = self.team_history[player_id]
            if len(history) >= 3:
                team_counts = {0: 0, 1: 0}
                for t in history:
                    if t in team_counts:
                        team_counts[t] += 1
                
                total = sum(team_counts.values())
                if total > 0:
                    dominant_team = max(team_counts, key=team_counts.get)
                    confidence = team_counts[dominant_team] / total
                    
                    # 지배적인 팀이 확실하면 저장
                    if confidence >= self.confidence_threshold:
                        self.player_teams[player_id] = dominant_team
            
            # 수정: 현재 팀이 지배적인 팀과 다르면 교정 (초반 제외)
            if player_id in self.player_teams and frames_since_first_seen > 5:
                expected_team = self.player_teams[player_id]
                if current_team != expected_team:
                    print(f"[TEAM_FIX] Player {player_id} at frame {self.current_frame}: "
                          f"team {current_team}→{expected_team}")
                    info = info.copy()
                    info['team_id'] = expected_team
            
            corrected_tracks[player_id] = info
        
        return corrected_tracks
    
    def detect_swap(self, player_tracks):
        """
        두 선수의 ID가 스왑되었는지 감지
        
        Returns:
            스왑해야 하는 (id1, id2) 튜플 리스트
        """
        swaps = []
        player_ids = list(player_tracks.keys())
        
        for i, pid1 in enumerate(player_ids):
            for pid2 in player_ids[i+1:]:
                # 두 선수 모두 팀 히스토리가 충분한지 확인
                if (pid1 in self.player_teams and 
                    pid2 in self.player_teams and
                    pid1 in player_tracks and 
                    pid2 in player_tracks):
                    
                    expected_team1 = self.player_teams[pid1]
                    expected_team2 = self.player_teams[pid2]
                    current_team1 = player_tracks[pid1].get('team_id')
                    current_team2 = player_tracks[pid2].get('team_id')
                    
                    # 크로스 체크: A의 현재 팀 = B의 예상 팀, B의 현재 팀 = A의 예상 팀
                    if (current_team1 == expected_team2 and 
                        current_team2 == expected_team1 and
                        expected_team1 != expected_team2):
                        
                        swaps.append((pid1, pid2))
                        print(f"[SWAP_DETECTED] Players {pid1} ↔ {pid2} at frame {self.current_frame}")
        
        return swaps
    
    def clear_history(self, player_id):
        """특정 선수의 히스토리를 초기화 (재등장 시)"""
        if player_id in self.team_history:
            del self.team_history[player_id]
        if player_id in self.player_teams:
            del self.player_teams[player_id]
        if player_id in self.player_colors:
            del self.player_colors[player_id]
        if player_id in self.first_seen_frame:
            del self.first_seen_frame[player_id]
