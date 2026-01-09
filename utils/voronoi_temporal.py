import numpy as np

class TemporalSmoother:
    """팀 확률맵(또는 one-hot 소유맵)을 EMA로 시간 스무딩합니다."""
    def __init__(self, alpha: float = 0.6, num_teams: int = 2):
        self.alpha = float(alpha)
        self.prev = None
        self.k = num_teams

    def smooth_team_probs(self, team_probs: np.ndarray) -> np.ndarray:
        """
        team_probs: (K,H,W) 팀별 확률맵(또는 0/1 one-hot)
        return: (K,H,W) EMA 결과
        """
        x = team_probs.astype(np.float32)
        if self.prev is None:
            self.prev = x
        else:
            self.prev = self.alpha * x + (1.0 - self.alpha) * self.prev
        return self.prev

def owner_map_to_team_probs(owner_map: np.ndarray, num_teams: int = 2) -> np.ndarray:
    """(H,W) owner_map(팀ID; -1은 미할당) → (K,H,W) one-hot 확률."""
    H, W = owner_map.shape
    probs = np.zeros((num_teams, H, W), dtype=np.float32)
    for k in range(num_teams):
        probs[k] = (owner_map == k).astype(np.float32)
    # 너무 하드하면 불안정하니 eps로 살짝 소프트닝
    eps = 1e-6
    probs = (probs + eps) / (probs.sum(axis=0, keepdims=True) + eps * num_teams)
    return probs

def team_probs_to_owner_map(team_probs: np.ndarray) -> np.ndarray:
    """(K,H,W) 확률 → (H,W) argmax 팀ID"""
    return np.argmax(team_probs, axis=0).astype(np.int8)
