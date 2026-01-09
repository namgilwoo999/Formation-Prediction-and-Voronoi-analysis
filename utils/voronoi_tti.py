import numpy as np

def _grid(H, W):
    ys, xs = np.mgrid[0:H, 0:W]
    return xs.astype(np.float32), ys.astype(np.float32)

def compute_owner_map_tti(
    pts_xy: np.ndarray,          # (N,2) [x,y] minimap pixels
    v_xy: np.ndarray,            # (N,2) velocity in px/frame
    team_ids: np.ndarray,        # (N,) in {0,1}
    H: int, W: int,
    tau: float = 0.7,            # look-ahead horizon [seconds]
    vperp_scale: float = 0.5,    # lateral speed ratio (v_perp = v * scale)
    kappa_turn: float = 0.15,    # turning/behind penalty
    fps: float = 25.0,           # video fps
    downsample: int = 2          # compute on smaller grid for speed
) -> np.ndarray:
    """
    간단한 TTI 근사: 각 픽셀까지의 도달 '시간'을 근사하여 argmin 팀으로 소유 결정.
    downsample>1이면 작은 그리드에서 계산 후 업샘플합니다.
    """
    assert pts_xy.shape[0] == v_xy.shape[0] == team_ids.shape[0]
    h, w = H // downsample, W // downsample
    xs, ys = _grid(h, w); xs *= downsample; ys *= downsample

    N = pts_xy.shape[0]
    eps = 1e-6
    speeds = np.linalg.norm(v_xy, axis=1) + eps
    headings = np.arctan2(v_xy[:,1], v_xy[:,0])  # [-pi,pi]

    owner_time = np.full((N, h, w), np.inf, dtype=np.float32)
    for i in range(N):
        px, py = pts_xy[i]
        dx = xs - px
        dy = ys - py
        c, s = np.cos(-headings[i]), np.sin(-headings[i])
        dxp = c*dx - s*dy
        dyp = s*dx + c*dy

        v_par  = speeds[i]
        v_perp = speeds[i] * vperp_scale + eps
        a = (v_par  * tau * fps) + eps  # heading 방향 유효 반경
        b = (v_perp * tau * fps) + eps  # 측면 유효 반경

        dist = np.sqrt((dxp/a)**2 + (dyp/b)**2).astype(np.float32)
        # 뒤쪽(heading 반대)일수록 추가 페널티
        turn_pen = np.where(dxp >= 0, 1.0, 1.0 + kappa_turn*np.minimum(1.0, (-dxp)/(a+1e-3))).astype(np.float32)
        owner_time[i] = dist * turn_pen

    t0 = np.min(owner_time[team_ids == 0], axis=0) if np.any(team_ids == 0) else np.full((h,w), np.inf, np.float32)
    t1 = np.min(owner_time[team_ids == 1], axis=0) if np.any(team_ids == 1) else np.full((h,w), np.inf, np.float32)
    owner_small = np.where(t0 <= t1, 0, 1).astype(np.int8)

    owner_map = np.repeat(np.repeat(owner_small, downsample, axis=0), downsample, axis=1)
    return owner_map[:H, :W]
