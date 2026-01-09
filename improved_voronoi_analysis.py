"""
Voronoi 분석 및 시각화
Team0/Team1 표시 및 팀 이름 인자 지원
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter
import scipy.stats as st
import argparse

# 고품질 출력 설정
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.style.use('seaborn-v0_8-whitegrid')

# 경로 설정
OWNER_MAP_DIR = Path(r"E:\soccer\football_analysis\runs\soccer_owner_maps")

# 팀 색상
TEAM_COLORS_PAPER = {
    0: '#FF6B6B',  # 팀 0: 빨간색 계열
    1: '#4ECDC4',  # 팀 1: 청록색 계열
    -1: '#F7F7F7'  # 미할당: 연한 회색
}

def parse_args():
    #명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Enhanced Voronoi Analysis')
    parser.add_argument('--team0', type=str, default='Team0',
                        help='Name of Team 0 (e.g., Ajax)')
    parser.add_argument('--team1', type=str, default='Team1',
                        help='Name of Team 1 (e.g., Bayern_Munchen)')
    return parser.parse_args()

def load_owner_maps(directory):
    #Static과 TTI owner map 로딩 함수
    maps = {'static': [], 'tti': []}
    files = sorted(directory.glob("*.npy"))
    
    for file in files:
        parts = file.stem.split('_')
        if len(parts) >= 3:
            frame_num = int(parts[1])
            map_type = parts[2]
            
            if map_type == 't':
                continue
                
            if map_type in maps:
                data = np.load(file)
                maps[map_type].append((frame_num, data))
    
    # 정렬
    for key in maps:
        maps[key].sort(key=lambda x: x[0])
        print(f"Loaded {len(maps[key])} {key} maps")
    
    return maps

def calculate_possession_stats(maps):
    # 정적 및 TTI 점유율 통계 계산
    stats = {}
    
    for map_type in ['static', 'tti']:
        possessions = []
        for frame_num, owner_map in maps[map_type]:
            valid_pixels = owner_map >= 0
            total_valid = np.sum(valid_pixels)
            
            if total_valid > 0:
                team0_pixels = np.sum(owner_map == 0)
                team1_pixels = np.sum(owner_map == 1)
                
                team0_pct = (team0_pixels / total_valid) * 100
                team1_pct = (team1_pixels / total_valid) * 100
                
                possessions.append({
                    'frame': frame_num,
                    'team0': team0_pct,
                    'team1': team1_pct
                })
        
        df = pd.DataFrame(possessions)
        
        # 이동 평균 적용 (스무딩)
        window = 25  # 1초 (25fps 기준)
        df['team0_smooth'] = df['team0'].rolling(window, center=True, min_periods=1).mean()
        df['team1_smooth'] = df['team1'].rolling(window, center=True, min_periods=1).mean()
        
        stats[map_type] = df
    
    return stats

def plot_comparison_figure(maps, frame_idx=100, team_names=None):
    # 논문용 비교 그림 생성 (정적 vs TTI)
    if team_names is None:
        team_names = ['Team0', 'Team1']
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['static', 'tti']
    titles = ['Static Voronoi', 'TTI-Voronoi']
    
    for idx, (method, title) in enumerate(zip(methods, titles)):
        if frame_idx < len(maps[method]):
            _, owner_map = maps[method][frame_idx]
            
            # 스무딩 적용 (시각적 개선)
            owner_map_smooth = gaussian_filter(owner_map.astype(float), sigma=1.0)
            
            # 커스텀 컬러맵
            colors = [TEAM_COLORS_PAPER[-1], TEAM_COLORS_PAPER[0], TEAM_COLORS_PAPER[1]]
            cmap = ListedColormap(colors)
            
            im = axes[idx].imshow(owner_map_smooth, cmap=cmap, vmin=-1, vmax=1)
            axes[idx].set_title(title, fontweight='bold')
            axes[idx].axis('off')
            
            # 스케일바 추가
            if idx == 0:
                # 10m 스케일바 (가정: 1픽셀 = 0.1m)
                scale_length = 100  # 100 pixels = 10m
                axes[idx].plot([50, 50+scale_length], [owner_map.shape[0]-30, owner_map.shape[0]-30], 
                             'k-', linewidth=3)
                axes[idx].text(50+scale_length//2, owner_map.shape[0]-40, '10m', 
                             ha='center', fontsize=9)
    
    plt.suptitle(f'Static vs TTI Voronoi Comparison (Frame {frame_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    plt.savefig('voronoi_comparison.png', bbox_inches='tight', dpi=300)
    print(f"Saved: voronoi_comparison.png")
    return fig

def plot_temporal_possession(stats, team_names=None):
    # 시간에 따른 점유율 변화 그래프 (정적 vs TTI)
    if team_names is None:
        team_names = ['Team0', 'Team1']
        
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    methods = ['static', 'tti']
    titles = ['Static Voronoi', 'TTI-Voronoi']
    
    for idx, (method, title) in enumerate(zip(methods, titles)):
        df = stats[method]
        
        # 원본 데이터 (연한 색)
        axes[idx].fill_between(df['frame'], 0, df['team0'], 
                              color=TEAM_COLORS_PAPER[0], alpha=0.3, label=f'{team_names[0]} (raw)')
        axes[idx].fill_between(df['frame'], df['team0'], 100, 
                              color=TEAM_COLORS_PAPER[1], alpha=0.3, label=f'{team_names[1]} (raw)')
        
        # 스무딩된 데이터 (진한 선)
        axes[idx].plot(df['frame'], df['team0_smooth'], 
                      color=TEAM_COLORS_PAPER[0], linewidth=2, label=team_names[0])
        axes[idx].plot(df['frame'], df['team1_smooth'], 
                      color=TEAM_COLORS_PAPER[1], linewidth=2, label=team_names[1])
        
        # 50% 기준선
        axes[idx].axhline(50, color='gray', linestyle='--', alpha=0.5)
        
        # 평균값 표시
        mean0 = df['team0'].mean()
        mean1 = df['team1'].mean()
        std0 = df['team0'].std()
        std1 = df['team1'].std()
        
        axes[idx].set_title(f'{title}: {team_names[0]}={mean0:.1f}%±{std0:.1f}%, {team_names[1]}={mean1:.1f}%±{std1:.1f}%',
                           fontweight='bold')
        axes[idx].set_ylabel('Spatial Possession (%)')
        axes[idx].set_ylim([0, 100])
        axes[idx].grid(True, alpha=0.3)
        
        if idx == 0:
            axes[idx].legend(loc='upper right', framealpha=0.9)
    
    axes[-1].set_xlabel('Frame Number')
    plt.suptitle('Static vs TTI: Temporal Evolution of Spatial Possession', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    plt.savefig('temporal_possession.png', bbox_inches='tight', dpi=300)
    print(f"Saved: temporal_possession.png")
    return fig

def plot_heatmaps(maps, team_names=None):
    # 논문용 명확한 히트맵과 영역 분석
    if team_names is None:
        team_names = ['Team0', 'Team1']
        
    fig = plt.figure(figsize=(14, 12))
    
    methods = ['static', 'tti']
    titles = ['Static Voronoi', 'TTI-Voronoi']
    
    # 히트맵용 누적 맵 생성
    accumulated_maps = {}
    
    for method in methods:
        if not maps[method]:
            continue
            
        height, width = maps[method][0][1].shape
        team0_accumulator = np.zeros((height, width))
        team1_accumulator = np.zeros((height, width))
        
        for frame_num, owner_map in maps[method]:
            team0_accumulator += (owner_map == 0).astype(float)
            team1_accumulator += (owner_map == 1).astype(float)
        
        # 정규화 (0-100%)
        total = len(maps[method])
        team0_accumulator = (team0_accumulator / total) * 100
        team1_accumulator = (team1_accumulator / total) * 100
        
        # 팀1이 더 많이 점유한 곳 - 팀0이 더 많이 점유한 곳
        dominance_map = team1_accumulator - team0_accumulator
        
        accumulated_maps[method] = dominance_map
    
    # 2x2 서브플롯: 상단에 히트맵, 하단에 영역별 분석
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.8], hspace=0.3, wspace=0.3)
    
    # 상단: 히트맵
    for col, (method, title) in enumerate(zip(methods, titles)):
        ax = fig.add_subplot(gs[0, col])
        
        dominance_map = accumulated_maps[method]
        
        # 가우시안 스무딩 적용
        dominance_smooth = gaussian_filter(dominance_map, sigma=2.0)
        
        # 등고선 표시
        im = ax.imshow(dominance_smooth, cmap='RdBu_r', vmin=-50, vmax=50)
        # ax.invert_xaxis() x,y 반전
        # 50% 경계선 (균등 점유)
        contours = ax.contour(dominance_smooth, levels=[0], colors='black', linewidths=2)
        ax.clabel(contours, inline=True, fmt='50-50', fontsize=8)
        
        ax.set_title(f'{title}\n{team_names[0]},{team_names[1]}', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Dominance (%)', fontsize=10)
        cbar.set_ticks([-50, -25, 0, 25, 50])
        cbar.set_ticklabels([f'{team_names[0]}\n100%', '', 'Equal', '', f'{team_names[1]}\n100%'])
    
    # 하단: 영역별 점유율 분석
    for col, (method, title) in enumerate(zip(methods, titles)):
        ax = fig.add_subplot(gs[1, col])
        
        # 영역 분할: 왼쪽/중앙/오른쪽
        dominance_map = accumulated_maps[method]
        height, width = dominance_map.shape
        
        left_third = width // 3
        right_third = 2 * width // 3
        
        # 각 영역별 점유율 계산
        zones = ['Left', 'Center', 'Right']
        team0_zones = []
        team1_zones = []
        
        for i, zone in enumerate(zones):
            if zone == 'Left':
                zone_data = dominance_map[:, :left_third]
            elif zone == 'Center':
                zone_data = dominance_map[:, left_third:right_third]
            else:
                zone_data = dominance_map[:, right_third:]
            
            # 평균 dominance 계산
            avg_dominance = np.mean(zone_data)
            team1_pct = 50 + avg_dominance  # 팀1 점유율
            team0_pct = 50 - avg_dominance  # 팀0 점유율
            
            team0_zones.append(team0_pct)
            team1_zones.append(team1_pct)
        
        # 막대 그래프
        x = np.arange(len(zones))
        width = 0.35
        
        rects0 = ax.bar(x - width/2, team0_zones, width, label=team_names[0], color=TEAM_COLORS_PAPER[0])
        rects1 = ax.bar(x + width/2, team1_zones, width, label=team_names[1], color=TEAM_COLORS_PAPER[1])
        
        # 값 표시
        for rect in rects0:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
        
        for rect in rects1:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
        
        # 50% 기준선
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Field Zone')
        ax.set_ylabel('Possession (%)')
        ax.set_title(f'{title} - Zone Control', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(zones)
        ax.set_ylim([0, 100])
        
        # 전체 통계 계산
        overall_team0 = np.mean(team0_zones)
        overall_team1 = np.mean(team1_zones)
        
        # 전체 텍스트를 상단에 배치
        overall_text = f'Overall: {team_names[0]}={overall_team0:.1f}%, {team_names[1]}={overall_team1:.1f}%'
        ax.text(0.98, 0.98, overall_text,
               transform=ax.transAxes, fontsize=9, fontweight='bold',
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 범례를 자동으로 최적 위치에 배치 (빈 공간 찾기)
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
    
    plt.suptitle('Spatial Possession Analysis: Static vs TTI', fontsize=16, fontweight='bold')
    
    # 저장
    plt.savefig('occupation_heatmaps.png', bbox_inches='tight', dpi=300)
    print(f"Saved: occupation_heatmaps.png")
    return fig

def plot_average_comparison(stats, team_names=None):
    # 평균 점유율 비교 막대 그래프
    if team_names is None:
        team_names = ['Team0', 'Team1']
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 평균값 계산
    static_team0 = stats['static']['team0'].mean()
    static_team1 = stats['static']['team1'].mean()
    tti_team0 = stats['tti']['team0'].mean()
    tti_team1 = stats['tti']['team1'].mean()
    
    # 막대 그래프 설정
    methods = ['STATIC', 'TTI']
    x = np.arange(len(methods))
    width = 0.35
    
    team0_values = [static_team0, tti_team0]
    team1_values = [static_team1, tti_team1]
    
    rects0 = ax.bar(x - width/2, team0_values, width, 
                    label=team_names[0], color=TEAM_COLORS_PAPER[0])
    rects1 = ax.bar(x + width/2, team1_values, width, 
                    label=team_names[1], color=TEAM_COLORS_PAPER[1])
    
    # 값 표시
    for rect, val in zip(rects0, team0_values):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for rect, val in zip(rects1, team1_values):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 50% 기준선
    ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% (Equal)')
    
    ax.set_ylabel('Spatial Possession (%)', fontsize=12)
    ax.set_title('Average Spatial Possession Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 80])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    plt.savefig('average_comparison.png', bbox_inches='tight', dpi=300)
    print(f"Saved: average_comparison.png")
    return fig
'''
def perform_statistical_tests(stats_data):
    #통계적 검정 수행
    results = {
        'static': {},
        'tti': {},
        'comparison': {}
    }
    
    # 각 방법별 통계
    for method in ['static', 'tti']:
        df = stats_data[method]
        
        # 기술 통계
        results[method]['mean_team0'] = df['team0'].mean()
        results[method]['mean_team1'] = df['team1'].mean()
        results[method]['std_team0'] = df['team0'].std()
        results[method]['std_team1'] = df['team1'].std()
        results[method]['median_team0'] = df['team0'].median()
        results[method]['median_team1'] = df['team1'].median()
        
        # 정상성 검정 (ADF test)
        from statsmodels.stats.stattools import jarque_bera
        from statsmodels.tsa.stattools import adfuller
        
        adf_result = adfuller(df['team0'])
        results[method]['stationarity_pvalue'] = adf_result[1]
        
        # 자기상관 검정
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        lb_result = acorr_ljungbox(df['team0'], lags=10, return_df=True)
        results[method]['autocorr_pvalue'] = lb_result['lb_pvalue'].min()
    
    # 방법 간 비교
    static_team0 = stats_data['static']['team0']
    tti_team0 = stats_data['tti']['team0']
    
    # 평균 차이
    results['comparison']['mean_diff'] = tti_team0.mean() - static_team0.mean()
    
    # 95% 신뢰구간
    diff = tti_team0 - static_team0
    results['comparison']['ci_lower'] = diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))
    results['comparison']['ci_upper'] = diff.mean() + 1.96 * diff.std() / np.sqrt(len(diff))
    
    # 대응표본 t-검정
    t_stat, p_val = stats.ttest_rel(tti_team0, static_team0)
    results['comparison']['ttest_statistic'] = t_stat
    results['comparison']['ttest_pvalue'] = p_val
    
    # 효과 크기 (Cohen's d)
    cohens_d = (tti_team0.mean() - static_team0.mean()) / np.sqrt((tti_team0.var() + static_team0.var()) / 2)
    results['comparison']['cohens_d'] = cohens_d
    
    # 비모수 검정 (윌콕슨 부호순위 검정)
    from scipy.stats import wilcoxon
    
    wilcox_stat, wilcox_p = wilcoxon(tti_team0, static_team0)
    results['comparison']['wilcoxon_statistic'] = wilcox_stat
    results['comparison']['wilcoxon_pvalue'] = wilcox_p
    
    return results
'''
def main():
    # 명령줄 인자 파싱
    args = parse_args()
    team_names = [f'Team0 ({args.team0})', f'Team1 ({args.team1})']
    
    print("Loading data...")
    maps = load_owner_maps(OWNER_MAP_DIR)
    
    if not maps['static'] and not maps['tti']:
        print("No owner maps found. Please run main.py first to generate owner maps.")
        return
    
    print("Calculating possession statistics...")
    stats = calculate_possession_stats(maps)
    
    print("Generating visualizations...")
    
    # 시각화 생성
    plot_comparison_figure(maps, frame_idx=100, team_names=team_names)
    plot_temporal_possession(stats, team_names=team_names)
    plot_heatmaps(maps, team_names=team_names)
    plot_average_comparison(stats, team_names=team_names)
    
    '''
    # 통계 분석
    test_results = perform_statistical_tests(stats)
    '''
    # 결과 출력
    print("\n" + "="*60)
    print("ENHANCED STATISTICAL ANALYSIS RESULTS")
    print("="*60)
    
    print("\n1. DESCRIPTIVE STATISTICS:")
    print("-"*40)
    '''
    for method in ['STATIC', 'TTI']:
        method_lower = method.lower()
        r = test_results[method_lower]
        print(f"\n{method}:")
        print(f"  {team_names[0]}: {r['mean_team0']:.2f}% ± {r['std_team0']:.2f}% (median: {r['median_team0']:.2f}%)")
        print(f"  {team_names[1]}: {r['mean_team1']:.2f}% ± {r['std_team1']:.2f}% (median: {r['median_team1']:.2f}%)")
        print(f"  Stationarity p-value: {r['stationarity_pvalue']:.4f}")
        print(f"  Autocorrelation p-value: {r['autocorr_pvalue']:.4f}")
    
    print("\n2. COMPARISON BETWEEN METHODS:")
    print("-"*40)
    
    comp = test_results['comparison']
    print(f"  Mean Difference (TTI - Static): {comp['mean_diff']:.2f}%")
    print(f"  95% CI of Difference: [{comp['ci_lower']:.2f}%, {comp['ci_upper']:.2f}%]")
    
    print(f"  Paired t-test:")
    print(f"    t-statistic: {comp['ttest_statistic']:.4f}")
    print(f"    p-value: {comp['ttest_pvalue']:.6f}")
    
    if comp['ttest_pvalue'] < 0.001:
        print(f"    Result: SIGNIFICANT")
    elif comp['ttest_pvalue'] < 0.05:
        print(f"    Result: SIGNIFICANT")
    else:
        print(f"    Result: Not significant")
    
    # 효과 크기 해석
    d = abs(comp['cohens_d'])
    if d < 0.2:
        effect = "Negligible"
    elif d < 0.5:
        effect = "Small"
    elif d < 0.8:
        effect = "Medium"
    else:
        effect = "Large"
    
    print(f"  Effect Size (Cohen's d): {comp['cohens_d']:.3f} ({effect})")
    
    print(f"  Non-parametric Test (Wilcoxon):")
    print(f"    statistic: {comp['wilcoxon_statistic']:.2f}")
    print(f"    p-value: {comp['wilcoxon_pvalue']:.6f}")
    
    print("\n3. PRACTICAL SIGNIFICANCE:")
    print("-"*40)
    
    # 팀 간 불균형 정도
    static_imbalance = abs(test_results['static']['mean_team0'] - test_results['static']['mean_team1'])
    tti_imbalance = abs(test_results['tti']['mean_team0'] - test_results['tti']['mean_team1'])
    
    print(f"  Team Imbalance (Static): {static_imbalance:.2f}%")
    print(f"  Team Imbalance (TTI): {tti_imbalance:.2f}%")
    print(f"  Imbalance Reduction: {static_imbalance - tti_imbalance:.2f}%")
    print(f"  Relative Improvement: {(1 - tti_imbalance/static_imbalance)*100:.1f}%")
    
    # 결론
    if comp['ttest_pvalue'] < 0.001:
        print("\n  → The difference between methods is HIGHLY SIGNIFICANT (p < 0.001)")
        print("  → TTI-Voronoi provides a more balanced assessment of spatial possession")
    elif comp['ttest_pvalue'] < 0.05:
        print("\n  → The difference between methods is SIGNIFICANT (p < 0.05)")
        print("  → TTI-Voronoi shows meaningful improvements over Static Voronoi")
    else:
        print("\n  → No significant difference between methods")
    '''
    # CSV 저장 (팀 이름 포함)
    for method in ['static', 'tti']:
        df = stats[method].copy()
        # 컬럼명에 팀 이름 추가
        df.columns = ['frame', 
                      f'team0_{args.team0}', 
                      f'team1_{args.team1}',
                      f'team0_{args.team0}_smooth',
                      f'team1_{args.team1}_smooth']
        df.to_csv(f'possession_stats_{method}.csv', index=False)
        print(f"Saved: possession_stats_{method}.csv")
    
    print("\n Analysis complete!")

if __name__ == "__main__":
    main()