
import cv2
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self, team0_name: str = None, team1_name: str = None):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        self.team1_name = team0_name
        self.team2_name = team1_name

    def _mask_grass(self, hsv: np.ndarray) -> np.ndarray:
        """
        잔디(초록색) 영역을 마스크합니다.
        hsv: H∈[0,180], S∈[0,255], V∈[0,255]
        """
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])
        grass = cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_not(grass)

    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray | None:
        """
        bbox 영역의 상반부를 잘라 HSV→잔디 마스크→LAB 픽셀 샘플링→KMeans(2)→
        채도(chroma) 높은 클러스터의 BGR 중심을 반환합니다.
        LAB 색상 공간의 a, b 채널만 사용하여 조명 변화에 강건합니다.
        """
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # 상반부(유니폼 저지 영역)만 사용
        top = roi[: roi.shape[0] // 2]

        # HSV로 변환 & 잔디 제거
        hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
        mask = self._mask_grass(hsv)

        # 마스크된 영역의 BGR 픽셀만 샘플링
        pixels = top[mask > 0]
        if len(pixels) < 30:
            return None

        # LAB로 변환 후 a, b 채널만 사용 (조명 불변성)
        lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        ab_pixels = lab_pixels[:, 1:]  # a, b 채널만 (L 채널 제외)

        # a, b 공간에서 2개 군집
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(ab_pixels)
        centers_ab = kmeans.cluster_centers_  # shape (2, 2)

        # 채도 개념: sqrt(a^2 + b^2)가 큰 것 선택 (원점에서 거리 = chroma)
        chroma = np.sqrt(centers_ab[:, 0]**2 + centers_ab[:, 1]**2)
        chosen = np.argmax(chroma)

        # 선택된 클러스터의 평균 LAB 계산 (L 포함)
        labels = kmeans.labels_
        chosen_lab = lab_pixels[labels == chosen].mean(axis=0)

        # BGR로 역변환
        chosen_bgr = cv2.cvtColor(
            chosen_lab.reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_LAB2BGR
        ).reshape(3).astype(float)

        return chosen_bgr

    def assign_team_color(self, frame: np.ndarray, detections: dict):
        """
        한 프레임의 모든 선수 bbox를 돌며 get_player_color 호출 →  
        수집된 색상 리스트로 다시 2-cluster KMeans →  
        두 팀의 대표 색(self.team_colors) 저장
        """
        self.player_team_dict.clear()
        samples = []
        for pid, det in detections.items():
            c = self.get_player_color(frame, det["bbox"])
            if c is not None:
                samples.append(c)

        if len(samples) < 2:
            # 충분치 않으면 빨강/파랑 기본
            self.team_colors = {
            0: np.array([0, 0, 255], dtype=float),
            1: np.array([255, 0, 0], dtype=float)
        }
            self.kmeans = None
            return

        X = np.vstack(samples)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        c1, c2 = self.kmeans.cluster_centers_
        # 클러스터 인덱스(0,1)를 그대로 팀 ID로 사용
        self.team_colors = {0: c1, 1: c2}

    def get_player_team(self, frame: np.ndarray, bbox: list, player_id: int) -> int:
        """
        player_id에 캐시된 팀이 있으면 반환,  
        없으면 get_player_color로 픽셀 → self.kmeans.predict → 1 or 2 반환 후 캐싱
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            team = 0
        else:
            c = self.get_player_color(frame, bbox)
            if c is None:
                team = 0
            else:
                team = int(self.kmeans.predict(c.reshape(1, -1))[0])

        self.player_team_dict[player_id] = team
        return team
