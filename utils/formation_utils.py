"""
포메이션 유틸리티
- 템플릿을 실제 선수 분포에 맞게 조정
"""

import json
import numpy as np
import os
from collections import defaultdict

from formation_analysis.classify_formation import FormationClassification
from formation_analysis.load_formation import load_formations


def normalize_formation_name(formation: str) -> str:
    """
    포메이션 이름 정규화
    예: "4-3-3" -> "433", "4-2-3-1" -> "4231"
    """
    if formation is None:
        return None
    return formation.replace('-', '').replace(' ', '')


def get_formation_template_positions(formation_name: str, field_width=527.0, field_height=351.0):
    """
    포메이션 템플릿의 실제 필드 좌표 반환
    
    Args:
        formation_name: 포메이션 이름 (예: "433", "4231")
        field_width: 필드 너비 (기본: 527.0)
        field_height: 필드 높이 (기본: 351.0)
    
    Returns:
        positions: (10, 2) shape의 실제 필드 좌표
        roles: 각 위치의 역할 이름 리스트
    """
    forms, form_names, forwards, defenders = load_formations()
    
    # 정규화된 포메이션 이름 찾기
    normalized = normalize_formation_name(formation_name)
    
    if normalized not in form_names:
        print(f"[WARNING] 포메이션 '{formation_name}' (정규화: '{normalized}')를 찾을 수 없습니다.")
        print(f"[WARNING] 사용 가능한 포메이션: {form_names}")
        print(f"[WARNING] 기본값 '433' 사용")
        normalized = "433"
    
    # 해당 포메이션 찾기
    idx = form_names.index(normalized)
    formation_template = forms[idx]  # (10, 3): [x, y, role]
    
    # 정규화된 좌표(0-10)를 실제 필드 좌표로 변환
    positions = []
    roles = []
    
    for i in range(10):
        x_norm = float(formation_template[i][0])  # 0-10
        y_norm = float(formation_template[i][1])  # 0-10
        role = formation_template[i][2]
        
        # 정규화된 좌표를 실제 좌표로 변환
        x_real = (x_norm / 10.0) * field_width
        y_real = (y_norm / 10.0) * field_height
        
        positions.append([x_real, y_real])
        roles.append(role)
    
    positions = np.array(positions)
    
    return positions, roles


def adjust_template_to_actual_distribution(template_positions, known_positions_dict):
    """
    템플릿을 실제 선수 분포에 맞게 조정 (NEW!)
    
    Args:
        template_positions: (10, 2) - 원본 템플릿 위치
        known_positions_dict: {player_id: [x, y]} - 실제로 알려진 선수들의 위치
    
    Returns:
        adjusted_template: (10, 2) - 조정된 템플릿 위치
    """
    if not known_positions_dict or len(known_positions_dict) == 0:
        # 알려진 위치가 없으면 원본 템플릿 반환
        return template_positions
    
    known_positions = np.array(list(known_positions_dict.values()))
    
    # 1. 실제 선수들의 통계
    actual_center = np.mean(known_positions, axis=0)
    actual_std = np.std(known_positions, axis=0)
    
    # 2. 템플릿의 통계
    template_center = np.mean(template_positions, axis=0)
    template_std = np.std(template_positions, axis=0)
    
    # 3. 템플릿을 실제 분포에 맞게 조정
    # (template - template_center) / template_std * actual_std + actual_center
    adjusted = (template_positions - template_center) / (template_std + 1e-6) * actual_std + actual_center
    
    return adjusted


def find_closest_template_position(known_positions_dict, template_positions, formation_name=None, verbose=False):
    """
    알려진 위치들과 가장 가까운 템플릿 위치 찾기 (개선됨)
    
    Args:
        known_positions_dict: {player_id: [x, y]} - 알려진 선수들
        template_positions: (10, 2) - 템플릿 위치
        formation_name: 포메이션 이름 (디버깅용)
        verbose: 디버깅 메시지 출력
    
    Returns:
        best_position: [x, y] - 가장 적절한 위치
    """
    if not known_positions_dict:
        # 알려진 위치가 없으면 템플릿 중 첫 번째 (보통 수비수)
        return template_positions[0]
    
    # 템플릿을 실제 분포에 맞게 조정
    adjusted_template = adjust_template_to_actual_distribution(template_positions, known_positions_dict)
    
    if verbose and formation_name:
        print(f"    [DEBUG] 템플릿 조정: 중심 이동 및 스케일 조정 완료")
        print(f"    [DEBUG] 원본 템플릿 범위: x=[{template_positions[:, 0].min():.1f}, {template_positions[:, 0].max():.1f}], "
              f"y=[{template_positions[:, 1].min():.1f}, {template_positions[:, 1].max():.1f}]")
        print(f"    [DEBUG] 조정 템플릿 범위: x=[{adjusted_template[:, 0].min():.1f}, {adjusted_template[:, 0].max():.1f}], "
              f"y=[{adjusted_template[:, 1].min():.1f}, {adjusted_template[:, 1].max():.1f}]")
    
    # 알려진 위치들의 중심 계산
    known_coords = np.array(list(known_positions_dict.values()))
    center = np.mean(known_coords, axis=0)
    
    # 사용되지 않은 템플릿 위치 찾기
    used_positions = set()
    for pos in known_coords:
        # 각 알려진 위치에 가장 가까운 조정된 템플릿 찾기
        distances = np.linalg.norm(adjusted_template - pos, axis=1)
        closest_idx = np.argmin(distances)
        used_positions.add(closest_idx)
    
    # 사용되지 않은 템플릿 중 중심에서 가장 가까운 것
    unused_indices = [i for i in range(len(adjusted_template)) if i not in used_positions]
    
    if unused_indices:
        distances_from_center = np.linalg.norm(adjusted_template[unused_indices] - center, axis=1)
        best_unused = unused_indices[np.argmin(distances_from_center)]
        best_position = adjusted_template[best_unused]
        
        if verbose:
            print(f"    [DEBUG] 사용 가능한 템플릿 인덱스: {unused_indices}")
            print(f"    [DEBUG] 선택된 인덱스: {best_unused}, 위치: [{best_position[0]:.1f}, {best_position[1]:.1f}]")
        
        return best_position
    else:
        # 모두 사용됨 -> 중심에서 가장 가까운 템플릿
        distances = np.linalg.norm(adjusted_template - center, axis=1)
        best_idx = np.argmin(distances)
        return adjusted_template[best_idx]


def extract_team_positions_from_json(json_path, team_id: int, formation_name: str = None, 
                                     use_stable_tracking=True, verbose=True, flip_team1_coordinates=True): # flip 반전 불필요시 False
    """
    팀 포지션 추출 (개선된 안정적 추적 + 포메이션 기반 초기화 + 분포 조정)
    
    개선점:
    1. 첫 유효 프레임(10명 이상)에서 10명의 ID를 고정
    2. 이후 프레임에서는 같은 ID만 추적
    3. ID가 사라지면 마지막으로 알려진 위치를 유지
    4. 초기 위치가 없으면 포메이션 템플릿 사용
    5. 템플릿을 실제 선수 분포에 맞게 조정 (NEW!)
    
    Args:
        json_path: object_tracks.json 경로
        team_id: 팀 ID (0 or 1)
        formation_name: 포메이션 이름 (예: "4-3-3", "4-2-3-1")
        use_stable_tracking: True면 개선된 안정적 추적 사용
        verbose: 디버깅 메시지 출력 여부
        flip_team1_coordinates: team_id==1일 때 X축 반전 여부 (기본: True)
    
    Returns:
        pos: (num_frames, 10, 2) shape의 포지션 배열
        player_ids: 고정된 10명의 player ID 리스트
    """
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not use_stable_tracking:
        # 기존 방식 (하위 호환성)
        return _extract_legacy(data, team_id, json_path, flip_team1_coordinates)
    
    if verbose:
        print(f"\n[INFO] 팀{team_id} 안정적인 포지션 추출 시작...")
        if formation_name:
            print(f"[INFO] 지정된 포메이션: {formation_name}")
    
    # 포메이션 템플릿 로드
    template_positions = None
    template_roles = None
    if formation_name:
        try:
            template_positions, template_roles = get_formation_template_positions(formation_name)
            if verbose:
                print(f"[INFO] 포메이션 템플릿 로드 완료: {normalize_formation_name(formation_name)}")
                print(f"[INFO] 템플릿 역할: {template_roles}")
        except Exception as e:
            print(f"[WARNING] 포메이션 템플릿 로드 실패: {e}")
            print(f"[WARNING] 기본 초기화 방식 사용")
    
    # Step 1: 첫 유효 프레임에서 10명의 ID 선택
    fixed_player_ids = None
    first_valid_frame_idx = -1
    
    for frame_idx, frame in enumerate(data):
        frame_players = [
            (int(pid), info["projection"])
            for pid, info in frame.get("player", {}).items()
            if info.get("team_id") == team_id and "projection" in info
        ]

        if len(frame_players) >= 7:  # 최소 7명 이상이면 ID 고정 시작
            # ID로 정렬하여 일관성 확보
            frame_players = sorted(frame_players, key=lambda x: x[0])

            # 실제 감지된 선수 ID (최대 10명)
            real_ids = [p[0] for p in frame_players[:10]]
            num_real = len(real_ids)

            # 10명 미만이면 가상 ID로 채우기
            if num_real < 10:
                num_virtual = 10 - num_real
                virtual_ids = [-i for i in range(1, num_virtual + 1)]
                fixed_player_ids = real_ids + virtual_ids

                if verbose:
                    print(f"[INFO] 첫 유효 프레임: {frame_idx}")
                    print(f"[INFO] 실제 감지: {num_real}명, 가상 ID: {num_virtual}명")
                    print(f"[INFO] 고정된 ID: {fixed_player_ids}")
            else:
                # 10명 이상 감지됨
                fixed_player_ids = real_ids

                if verbose:
                    print(f"[INFO] 첫 유효 프레임: {frame_idx}")
                    print(f"[INFO] 고정된 10명의 ID: {fixed_player_ids}")

            first_valid_frame_idx = frame_idx
            break

    if fixed_player_ids is None:
        if verbose:
            print(f"[WARNING] 팀{team_id}에서 7명 이상의 선수가 있는 프레임을 찾지 못했습니다.")
        return np.array([]), []
    
    # Step 2: 각 프레임에서 고정된 ID만 추적
    team_positions = []
    last_known_positions = {}  # pid -> [x, y]
    missing_counts = defaultdict(int)  # pid -> 연속 누락 횟수
    template_used_count = 0  # 템플릿 사용 횟수 (디버깅용)
    
    for frame_idx, frame in enumerate(data):
        frame_dict = {}
        
        # 현재 프레임에서 선수 정보 추출
        for pid, info in frame.get("player", {}).items():
            if info.get("team_id") == team_id and "projection" in info:
                frame_dict[int(pid)] = info["projection"]
        
        # 고정된 10명의 위치 수집
        current_positions = []
        
        for pid in fixed_player_ids:
            if pid in frame_dict:
                # 선수가 감지됨 - 현재 위치 사용
                pos = frame_dict[pid]
                current_positions.append(pos)
                last_known_positions[pid] = pos
                missing_counts[pid] = 0  # 누락 카운트 리셋
                
            elif pid in last_known_positions:
                # 선수가 감지되지 않음 - 마지막 위치 사용
                pos = last_known_positions[pid]
                current_positions.append(pos)
                missing_counts[pid] += 1
                
                # 10프레임 이상 연속으로 누락되면 경고
                if missing_counts[pid] == 10 and verbose:
                    print(f"[WARNING] 프레임 {frame_idx}: 선수 ID {pid}가 10프레임째 감지되지 않음 (마지막 위치 사용 중)")
                
            else:
                # 첫 프레임부터 감지되지 않음 - 포메이션 템플릿 또는 기본 위치
                if template_positions is not None:
                    # 템플릿을 실제 분포에 맞게 조정하여 사용
                    pos = find_closest_template_position(
                        last_known_positions, 
                        template_positions,
                        formation_name=formation_name,
                        verbose=(verbose and frame_idx == first_valid_frame_idx)
                    )
                    pos = pos.tolist()
                    template_used_count += 1
                    
                    if verbose and frame_idx == first_valid_frame_idx:
                        print(f"[INFO] 선수 ID {pid}: 포메이션 템플릿 위치 사용 (분포 조정됨) [{pos[0]:.1f}, {pos[1]:.1f}]")
                else:
                    # 템플릿 없으면 필드 중앙 (기존 방식)
                    pos = [263.0, 176.0]
                    
                    if verbose and frame_idx == first_valid_frame_idx:
                        print(f"[WARNING] 선수 ID {pid}: 초기 위치 없음, 필드 중앙 사용")
                
                current_positions.append(pos)
                last_known_positions[pid] = pos
        
        team_positions.append(current_positions)
    
    # Step 3: numpy 배열로 변환
    pos = np.array(team_positions)  # (num_frames, 10, 2)
    
    if verbose:
        print(f"\n[INFO] 팀{team_id} 포지션 추출 완료")
        print(f"- 총 프레임: {len(pos)}")
        print(f"- 포지션 shape: {pos.shape}")
        print(f"- 고정 ID: {fixed_player_ids}")
        if template_used_count > 0:
            print(f"- 템플릿 사용 횟수: {template_used_count}회")
        
        # 각 선수의 감지율 출력
        print(f"\n[INFO] 선수별 감지율:")
        for pid in fixed_player_ids:
            detected = sum(1 for f in data if int(pid) in [int(p) for p in f.get("player", {}).keys() 
                                                            if f.get("player", {})[p].get("team_id") == team_id])
            rate = detected / len(data) * 100
            status = "Check" if rate >= 80 else "!" if rate >= 50 else "X"
            print(f"  {status} 선수 {pid}: {detected}/{len(data)} 프레임 ({rate:.1f}%)")
    
    # Step 4: 필드 반전 (기존 로직 유지)
    if not flip_team1_coordinates:
        pass  # 좌표 반전 안 함
    else:
        if team_id == 1:
            field_width = 527.0  # 필드 너비 (top_down_keypoints 기준)
            pos[:, :, 0] = field_width - pos[:, :, 0]
    
    return pos, fixed_player_ids


def _extract_legacy(data, team_id, json_path, flip_team1_coordinates=True):

    # 기존 방식
    team_frames = []

    for frame in data:
        frame_players = [
            (int(pid), info["projection"])
            for pid, info in frame.get("player", {}).items()
            if info.get("team_id") == team_id and "projection" in info
        ]
        if len(frame_players) >= 10:
            frame_players = sorted(frame_players, key=lambda x: x[0])
            coords = [p[1] for p in frame_players[:10]]
            ids = [p[0] for p in frame_players[:10]]
            team_frames.append((coords,ids))

    if not team_frames:
        return np.array([]), []
    
    pos = np.array([f[0] for f in team_frames])
    player_ids = team_frames[0][1]

    if not flip_team1_coordinates:
        pass  # 좌표 반전 안 함
    else:
        if team_id == 1:
            field_width = np.max(pos[:, :, 0])
            pos[:, :, 0] = field_width - pos[:, :, 0]

    return pos, player_ids


def assign_formation_from_tracks(json_path: str, save_dir: str, 
                                 formation0: str = None, formation1: str = None, flip_team1_coordinates=True):
    """
    트래킹 데이터로부터 포메이션 할당 (개선된 안정적 추적 + 분포 조정)
    
    Args:
        json_path: object_tracks.json 경로
        save_dir: 결과 저장 디렉토리
        formation0: 팀0의 포메이션 (예: "4-3-3")
        formation1: 팀1의 포메이션 (예: "4-2-3-1")
        flip_team1_coordinates: team_id==1일 때 X축 반전 여부 (기본: True)
    """
    
    formations = {0: formation0, 1: formation1}
    
    for team_id in [0, 1]:
        print(f"\n{'='*80}")
        print(f"[INFO] Processing team {team_id} formation...")
        print(f"{'='*80}")
        
        # 개선된 안정적 추적 사용 (포메이션 템플릿 + 분포 조정)
        pos, player_ids = extract_team_positions_from_json(
            json_path, 
            team_id, 
            formation_name=formations[team_id],
            use_stable_tracking=True, 
            verbose=True,
            flip_team1_coordinates=flip_team1_coordinates
        )
        
        if len(pos) == 0:
            print(f"[WARNING] No valid player positions found for team {team_id}")
            continue
        
        # 저장 (npy + json)
        npy_path = os.path.join(save_dir, f"team{team_id}_pos.npy")
        json_path_out = os.path.join(save_dir, f"team{team_id}_pos.json")
        np.save(npy_path, pos)
        
        json_dict = {
            "player_ids": player_ids,
            "positions": pos.tolist(),
            "formation": formations[team_id]
        }
        with open(json_path_out, 'w') as f:
            json.dump(json_dict, f, indent=2)
        
        # 디버깅용 출력
        print(f"\n[DEBUG] Team {team_id} 결과:")
        print(f"- pos shape: {pos.shape}")
        print(f"- player_ids: {player_ids}")
        print(f"- formation: {formations[team_id]}")
        print(f"- 평균 위치 범위: x=[{pos[:,:,0].min():.1f}, {pos[:,:,0].max():.1f}], "
              f"y=[{pos[:,:,1].min():.1f}, {pos[:,:,1].max():.1f}]")
        
        # 안전성 검사
        if np.isnan(pos).any() or np.isinf(pos).any():
            print(f"[ERROR] Team {team_id}: pos contains NaN or inf. Skipping...")
            continue
        
        if pos.shape[1] != 10:
            print(f"[ERROR] Team {team_id}: Expected 10 players, got {pos.shape[1]}. Skipping...")
            continue
        
        # 포메이션 분류
        try:
            fc = FormationClassification(pos, player_ids)
            fc.compute_form_summary()

            # 시각화 저장(임시 비활성화)
            plot_path = os.path.join(save_dir, f"team{team_id}_overall_formation.png")
            fc.visualize_form_summary(title=f"Team{team_id} Overall Formation", save_path=plot_path)
            print(f"[INFO] Saved team {team_id} overall formation visualization to {plot_path}")

            # JSON 파일 저장 (FormationConfusionAnalyzer가 사용)
            overall_json_path = os.path.join(save_dir, f"team{team_id}_overall_formation.json")
            overall_data = {
                'top_matches': fc.get_top_k_formations(k=10),
                'roles': fc.roles,
                'player_ids': player_ids,
                'specified_formation': formations[team_id]
            }
            with open(overall_json_path, 'w') as f:
                json.dump(overall_data, f, indent=2)
            print(f"[INFO] Saved team {team_id} overall formation data to {overall_json_path}")

            # 평균 포지션 저장 (Voronoi 사용)
            avg_pos_path = os.path.join(save_dir, f"team{team_id}_overall_avg_pos.npy")
            pos_mean = np.mean(pos, axis=0)  # (10, 2) shape
            np.save(avg_pos_path, pos_mean)
            print(f"[INFO] Saved team {team_id} overall average position to {avg_pos_path}")

        except Exception as e:
            print(f"\n[ERROR] Team {team_id} 포메이션 분류 중 에러 발생!")
            print(f"[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"[WARNING] Skipping team {team_id} formation analysis...\n")
            continue
        
        print(f"\n{'='*80}\n")
