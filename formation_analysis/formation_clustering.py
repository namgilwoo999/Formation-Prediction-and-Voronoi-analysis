import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from formation_analysis.classify_formation import compute_similarity
import os
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import Voronoi
from matplotlib.patches import Rectangle, Arc,Polygon
import shapely.geometry as sg
from shapely.geometry import Polygon as SPolygon, box
import json
from pathlib import Path

def compute_distance_matrix(formations):
    """
    포메이션 간 쌍별 거리(1 - 유사도) 행렬을 계산합니다.

    Args:
        formations (list of np.ndarray): 포메이션 리스트: (10, 2) 포메이션 요약 리스트

    Returns:
        np.ndarray: (N, N) 거리 행렬
    """
    n = len(formations)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim, _ = compute_similarity(formations[i], formations[j])
                dist_mat[i, j] = 1 - sim
    return dist_mat

def cluster_formations(formations, method='ward', n_clusters=3):
    """
    포메이션에 대해 계층적 클러스터링을 수행합니다.

    Args:
        formations (list of np.ndarray): 포메이션 리스트
        method (str): 연결 방법
        n_clusters (int): 원하는 클러스터 수

    Returns:
        labels (np.ndarray): 클러스터 레이블
    """
    flat_forms = [f.flatten() for f in formations]
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(flat_forms)
    return labels

def plot_dendrogram(formations, labels=None):
    """
    덴드로그램을 통해 계층적 클러스터링을 시각화합니다.

    Args:
        formations (list of np.ndarray): 포메이션 리스트
        labels (list of str): 선택적 레이블
    """
    flat_forms = [f.flatten() for f in formations]
    Z = linkage(flat_forms, method='ward')
    plt.figure(figsize=(12, 4))
    dendrogram(Z, labels=labels)
    plt.title("Dendrogram of Formation Clusters")
    plt.xlabel("Segment")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

def plot_pca_clusters(formations, cluster_labels):
    """
    PCA 투영 및 클러스터 시각화를 수행합니다.

    Args:
        formations (list of np.ndarray): 포메이션 리스트
        cluster_labels (list or np.ndarray): 클러스터 레이블
    """
    flat_forms = np.array([f.flatten() for f in formations])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(flat_forms)

    plt.figure(figsize=(8, 6))
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        plt.scatter(reduced[mask, 0], reduced[mask, 1], label=f"Cluster {cluster_id+1}")

    plt.title("Formation Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def cluster_formations_by_similarity(stub_dir: str, team_id: int, save_path: str = None):
    """
    세그먼트별 포지션 평균 데이터를 기반으로 포메이션 클러스터링 및 시각화

    Args:
        stub_dir (str): 세그먼트 결과가 저장된 디렉토리
        team_id (int): 팀 번호
        save_path (str): 저장 경로
    """
    # 1. 모든 segment의 formation summary 로드
    summaries = []
    labels = []

    for i in range(1, 20):  # 최대 20 segment까지 탐색
        npy_path = os.path.join(stub_dir, f"team{team_id}_segment{i}_form_summary.npy")
        if not os.path.exists(npy_path):
            continue
        summary = np.load(npy_path).reshape(-1)
        summaries.append(summary)
        labels.append(f"S{i}")

    if not summaries:
        print(f"[WARNING] 팀 {team_id}의 클러스터링을 위한 요약 정보가 없습니다.")
        return
    
    # 세그먼트가 1개만 있으면 클러스터링 불가능
    if len(summaries) < 2:
        print(f"[WARNING] 팀 {team_id}의 세그먼트가 1개뿐이어서 클러스터링을 수행할 수 없습니다.")
        return

    # 2. 클러스터링 수행
    summaries = np.array(summaries)
    dist_mat = pairwise_distances(summaries, metric="euclidean")

    Z = linkage(dist_mat, method="ward")

    # 3. 덴드로그램 시각화
    plt.figure(figsize=(10, 4))
    if len(Z) > 0:
        color_thresh = 0.7 * max(Z[:, 2])
        dendrogram(Z, labels=labels, color_threshold=color_thresh)
    else:
        dendrogram(Z, labels=labels)
        
    plt.title(f"Team {team_id} Formation Clustering")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[INFO] 클러스터링 시각화 저장 완료 → {save_path}")
    else:
        plt.show()

def load_formation_config(config_path="formation_config.json"):
    """포메이션 설정 파일 로드"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('formation_groups', {})
    return {}

def normalize_formation(formation, config):
    """변형 포메이션을 base 포메이션으로 정규화"""
    formation_clean = formation.replace('-', '')
    
    for family_name, family_data in config.items():
        base = family_data.get('base')
        variants = family_data.get('variants', [])
        if formation_clean in variants and base:
            return base
    return formation_clean

def get_formation_display(formation, config):
    """포메이션 표시 형식 반환"""
    formation_clean = formation.replace('-', '')
    base = normalize_formation(formation, config)
    
    if formation_clean == base:
        # base 포메이션인 경우
        for family_name, family_data in config.items():
            if family_data.get('base') == formation_clean:
                return f"{formation} (기본)"
        return formation  # 설정에 없는 포메이션
    else:
        # 파생 포메이션인 경우
        return f"{formation} ({base}계열)"

def plot_formation_scatter(
    avg_positions0: np.ndarray,
    roles0: list,
    avg_positions1: np.ndarray,
    roles1: list,
    segment_id: int,
    frame_range: list,
    top_formations0: list = None,
    top_formations1: list = None,
    save_path: str = None,
    ground_truth0: str = None,
    ground_truth1: str = None
):
    """포메이션 정규화를 반영한 개선된 시각화 함수"""

    print(f"[DEBUG] plot_formation_scatter started")

    # 포메이션 설정 로드
    print(f"[DEBUG] Loading formation config...")
    config = load_formation_config()
    print(f"[DEBUG] Config loaded")
    
    # team1 좌표 좌우 반전
    print(f"[DEBUG] Flipping team1 coordinates...")
    avg_positions1 = avg_positions1.copy()
    avg_positions1[:, 0] = 1 - avg_positions1[:, 0]

    def generate_boundary_points(field_margin=1.0):
        return np.array([
            [-field_margin, -field_margin],
            [ 0.5, -field_margin],
            [ 1 + field_margin, -field_margin],
            [-field_margin,  0.5],
            [ 1 + field_margin,  0.5],
            [-field_margin, 1 + field_margin],
            [ 0.5, 1 + field_margin],
            [ 1 + field_margin, 1 + field_margin]
        ])

    print(f"[DEBUG] Creating matplotlib figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    print(f"[DEBUG] Figure created")

    # 1) 선수 및 경계 포인트 준비
    print(f"[DEBUG] Preparing Voronoi points...")
    boundary_points = generate_boundary_points(field_margin=1.0)
    vor_points = np.vstack([avg_positions0, avg_positions1])
    vor_all_points = np.vstack([vor_points, boundary_points])

    n0, n1 = len(avg_positions0), len(avg_positions1)
    team_colors = ['dodgerblue'] * n0 + ['orangered'] * n1
    print(f"[DEBUG] n0={n0}, n1={n1}, total points={len(vor_all_points)}")

    print(f"[DEBUG] Computing Voronoi diagram...")
    try:
        vor = Voronoi(vor_all_points)
        print(f"[DEBUG] Voronoi computed successfully")
    except Exception as e:
        print(f"[ERROR] Voronoi 생성 실패 (segment {segment_id}): {e}")
        return

    # 2) 피치 그리기
    def draw_pitch(ax):
        ax.set_facecolor('green')
        ax.plot([0, 1], [0, 0], color='white', lw=2)
        ax.plot([1, 1], [0, 1], color='white', lw=2)
        ax.plot([1, 0], [1, 1], color='white', lw=2)
        ax.plot([0, 0], [1, 0], color='white', lw=2)
        ax.axvline(0.5, color='white', linestyle='-', linewidth=2)
        ax.add_patch(plt.Circle((0.5, 0.5), 0.1, color='white', fill=False, linewidth=2))
        ax.plot(0.5, 0.5, 'wo', markersize=3)
        for x in [0, 1]:
            penalty_x = x if x == 0 else x - 0.15
            goal_x = x if x == 0 else x - 0.05
            spot_x = 0.11 if x == 0 else 0.89
            ax.add_patch(Rectangle((penalty_x, 0.3), 0.15, 0.4, fill=False, color='white', linewidth=2))
            ax.add_patch(Rectangle((goal_x, 0.4), 0.05, 0.2, fill=False, color='white', linewidth=2))
            ax.plot(spot_x, 0.5, 'wo', markersize=2)
            arc = Arc((spot_x, 0.5), 0.2, 0.2,
                      theta1=310 if x == 0 else 130,
                      theta2=50 if x == 0 else 230,
                      color='white', linewidth=2)
            ax.add_patch(arc)

    draw_pitch(ax)

    # 3) Voronoi 셀 시각화 및 점유율 계산
    # 경기장 경계 정의 (0~1 정규화 기준)
    field_polygon = box(0, 0, 1, 1)

    team_area = [0.0, 0.0]
    for idx in range(len(vor_points)):
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        polygon = [vor.vertices[i] for i in region]
        if len(polygon) < 3:
            continue
        color = team_colors[idx]
        ax.fill(*zip(*polygon), facecolor=color, alpha=0.38, edgecolor=None, zorder=0)

        cell = SPolygon(polygon)
        if cell.is_valid and not cell.is_empty:
            clipped_cell = cell.intersection(field_polygon)
            if clipped_cell.is_valid and not clipped_cell.is_empty:
                team_id = 0 if idx < n0 else 1
                team_area[team_id] += clipped_cell.area

    # 4) 선수 마커 및 라벨
    def plot_team(avg_pos, roles, color, marker):
        for (x, y), role in zip(avg_pos, roles):
            ax.scatter(x, y, color=color, edgecolors='k', s=120, marker=marker, zorder=2)
            ax.text(x + 0.008, y + 0.008, role, fontsize=9, zorder=3, color='white', weight='bold')

    plot_team(avg_positions0, roles0, 'dodgerblue', 'o')
    plot_team(avg_positions1, roles1, 'orangered', '^')

    # 5) 헤더/타이틀
    ax.text(0.04, 0.975, "Team 0 →", color='dodgerblue', transform=ax.transAxes, fontsize=12, weight='bold')
    ax.text(0.80, 0.975, "← Team 1", color='orangered', transform=ax.transAxes, fontsize=12, weight='bold')
    ax.set_title(f"Voronoi Analysis - Segment {segment_id} (Frames {frame_range[0]}-{frame_range[1]})", 
                fontsize=13, pad=10, fontweight='bold')

    # 6) 포메이션 표시 (정규화 반영)
    def format_formation_list(formations, ground_truth=None):
        """포메이션 리스트를 정규화 정보와 함께 포맷팅"""
        if not formations:
            return "No data"
        
        lines = []
        for i, (name, score) in enumerate(formations[:3]):  # 상위 3개만
            # 포메이션 표시 형식 가져오기
            display = get_formation_display(name, config)
            
            # 실제값과 일치하는지 확인 (정규화된 비교)
            is_match = False
            if ground_truth:
                gt_norm = normalize_formation(ground_truth, config)
                pred_norm = normalize_formation(name, config)
                is_match = (gt_norm == pred_norm)
            
            # 매치 표시
            match_mark = " ✓" if is_match else ""
            lines.append(f"{i+1}. {display} ({score:.2f}){match_mark}")
        
        return "\n".join(lines)
    
    # 포메이션 정보 표시
    y_pos = -0.25  # 시작 위치
    
    # 팀 0 포메이션
    if top_formations0:
        formation_text0 = format_formation_list(top_formations0, ground_truth0)
        ax.text(0.02, y_pos, f"Team0 예측:\n{formation_text0}", 
                color='dodgerblue', transform=ax.transAxes,
                fontsize=10, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
    
    if ground_truth0:
        gt_display0 = get_formation_display(ground_truth0, config)
        ax.text(0.02, y_pos - 0.15, f"Team0 실제: {gt_display0}", 
                color='darkblue', transform=ax.transAxes,
                fontsize=10, ha='left', va='top', weight='bold')
    
    # 팀 1 포메이션
    if top_formations1:
        formation_text1 = format_formation_list(top_formations1, ground_truth1)
        ax.text(0.75, y_pos, f"Team1 예측:\n{formation_text1}", 
                color='orangered', transform=ax.transAxes,
                fontsize=10, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
    
    if ground_truth1:
        gt_display1 = get_formation_display(ground_truth1, config)
        ax.text(0.75, y_pos - 0.15, f"Team1 실제: {gt_display1}", 
                color='darkred', transform=ax.transAxes,
                fontsize=10, ha='left', va='top', weight='bold')

    # 7) 점유율 표시
    total = sum(team_area)
    pct0 = team_area[0] / total * 100 if total > 0 else 0
    pct1 = team_area[1] / total * 100 if total > 0 else 0
    
    # 점유율 바 그래프 추가
    bar_y = -0.05
    bar_height = 0.02
    ax.add_patch(Rectangle((0.3, bar_y), pct0*0.004, bar_height, 
                           transform=ax.transAxes, facecolor='dodgerblue', alpha=0.7))
    ax.add_patch(Rectangle((0.3 + pct0*0.004, bar_y), pct1*0.004, bar_height, 
                           transform=ax.transAxes, facecolor='orangered', alpha=0.7))
    
    ax.text(0.02, bar_y + 0.01, f"Team0: {pct0:.1f}%", color='dodgerblue', 
            transform=ax.transAxes, fontsize=11, weight='bold', va='center')
    ax.text(0.70, bar_y + 0.01, f"Team1: {pct1:.1f}%", color='orangered', 
            transform=ax.transAxes, fontsize=11, weight='bold', va='center')

    # 8) 정확도 요약 (정규화된 비교 결과)
    accuracy_text = ""
    if ground_truth0 and top_formations0:
        gt_norm0 = normalize_formation(ground_truth0, config)
        pred_norm0 = normalize_formation(top_formations0[0][0], config)
        match0 = "✓" if gt_norm0 == pred_norm0 else "✗"
        accuracy_text += f"T0: {match0} "
    
    if ground_truth1 and top_formations1:
        gt_norm1 = normalize_formation(ground_truth1, config)
        pred_norm1 = normalize_formation(top_formations1[0][0], config)
        match1 = "✓" if gt_norm1 == pred_norm1 else "✗"
        accuracy_text += f"T1: {match1}"
    
    if accuracy_text:
        ax.text(0.5, -0.35, f"정규화 매칭: {accuracy_text}", 
                transform=ax.transAxes, fontsize=10, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            plt.savefig(save_path, dpi=50)  # 메모리 절약을 위해 DPI 더 감소
            print(f"[INFO] Saved enhanced Voronoi plot to {save_path}")
        except MemoryError:
            print(f"[WARNING] Memory error saving {save_path}, trying with lower DPI...")
            plt.savefig(save_path, dpi=30)
            print(f"[INFO] Saved with reduced quality to {save_path}")
    plt.close(fig)  # 명시적으로 figure 메모리 해제
