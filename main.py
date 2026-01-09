import matplotlib
matplotlib.use('Agg')

from utils.video_utils import read_video_generator, save_video
from tracking.tracker import Tracker
from tracking import KeypointsTracker 
from team_assigner.team_assigner import TeamAssigner
from ball_to_player_assignment.player_ball_assigner import PlayerBallAssigner
from annotation import FootballVideoProcessor
import argparse, os, numpy as np
import cv2, json
import gc
from utils.formation_utils import assign_formation_from_tracks
from formation_analysis.formation_clustering import *
from formation_analysis.formation_confusion_matrix import FormationConfusionAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description='Soccer Video Analysis Pipeline')
    parser.add_argument('input_videos',   type=str, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default='output.mp4',
                        help='Path to save annotated output video')
    parser.add_argument('--team0', type=str, required=True,
                        help='첫 번째 팀 이름')
    parser.add_argument('--team1', type=str, required=True,
                        help='두 번째 팀 이름')
    parser.add_argument('--formation0', type=str, default=None,
                        help='팀0의 실제 포메이션 (예: 4-3-3)')
    parser.add_argument('--formation1', type=str, default=None,
                        help='팀1의 실제 포메이션 (예: 4-2-3-1)')
    parser.add_argument('--stub_dir',     type=str, default='stubs',
                        help='Directory to cache intermediate results')
    parser.add_argument('--interactive', action='store_true',
                        help='클릭 인터랙티브 미리보기 모드 (끝내려면 q)')
    return parser.parse_args()

def main():
    args = parse_args()
    # 축구 분석 프로젝트 사용 방법을 설명하는 주요 기능입니다. 모델 로드, 팀 할당, 객체 및 선수 추적, 동영상 처리 과정을 안내합니다.
    
    # 0) TeamAssigner 생성 시 팀 이름 전달
    team_assigner = TeamAssigner(team0_name=args.team0, team1_name=args.team1)

    # 1) Tracker 모델
    obj_tracker = Tracker(
        model_path='models/best.pt',
        project_dir='.',                          
        video_name=os.path.basename('input_videos/Kaggle_DFL_3.mp4')
    )

    # 2. keypoint 모델
    # conf, kp_conf 변경
    kp_tracker = KeypointsTracker(
        model_path='models/court_keypoint_detector.pt',
        conf=0.3,      # 필드 감지 threshold
        kp_conf=0.5,   # 키포인트 신뢰 threshold
    )
    
    # 3. 팀 할당
    ball_to_player_assigner = PlayerBallAssigner()

    # 5. 축구 경기장의 상단 뷰를 위한 키포인트 정의 (왼쪽에서 오른쪽, 위에서 아래 순서)
    # 필드의 원근 변환에 사용됨
    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],             # 0-5 (left goal line)
        [32, 122], [32, 229],                                                # 6-7 (left goal box corners)
        [64, 176],                                                           # 8 (left penalty dot)
        [96, 57], [96, 122], [96, 229], [96, 293],                           # 9-12 (left penalty box)
        [263, 0], [263, 122], [263, 229], [263, 351],                        # 13-16 (halfway line)
        [431, 57], [431, 122], [431, 229], [431, 293],                       # 17-20 (right penalty box)
        [463, 176],                                                          # 21 (right penalty dot)
        [495, 122], [495, 229],                                              # 22-23 (right goal box corners)
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351], # 24-29 (right goal line)
        [210, 176], [317, 176]                                               # 30-31 (center circle leftmost and rightmost points)
    ])

    # 6. 비디오 프로세서 초기화
    # 이 프로세서는 분석에 필요한 모든 작업을 처리합니다.
    processor = FootballVideoProcessor(obj_tracker=obj_tracker,                                  # Created ObjectTracker object
                                       kp_tracker=kp_tracker,                                    # Created KeypointsTracker object
                                       team_assigner=team_assigner,                              # Created TeamAssigner object
                                       ball_to_player_assigner=ball_to_player_assigner,          # Created BallToPlayerAssigner object
                                       top_down_keypoints=top_down_keypoints,                    # Created Top-Down keypoints numpy array
                                       field_img_path='input_videos/field_2d_v2.png', # Top-Down field image path
                                       save_tracks_dir=args.stub_dir,                 # 추적 정보 저장 디렉토리
                                       draw_frame_num=True                            # 프레임 번호 표시 여부 
                                                                                      # 출력 비디오
                                       )
    
    # 7) 비디오 저장
    # 7-1) 프레임 및 FPS 읽기
    cap = cv2.VideoCapture(args.input_videos)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"[INFO] Total frames in video: {total_frames}")
    print(f"[INFO] Video FPS: {fps}")
    
    # 비디오가 너무 짧은 경우 경고
    if total_frames < 10:
        print(f"[WARNING] Video has only {total_frames} frames, which may be too short for analysis")
    
    frame_gen = read_video_generator(args.input_videos)
    '''
    # 선수의 크롭된 이미지 저장
    os.makedirs("output_videos/crops", exist_ok=True)

    frame0 = frames[0]
    H, W = frame0.shape[:2]

    tracks_f0 = player_tracks[0]        

    for track_id, det in tracks_f0.items():
        x1, y1, x2, y2 = map(int, det["bbox"])

        # 프레임 경계로 클램프(음수/초과 방지)
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H))

        if x2 > x1 and y2 > y1:
            crop = frame0[y1:y2, x1:x2]
            cv2.imwrite(f"output_videos/crops/player_{track_id}_f0.jpg", crop)
            break  # 한 장만 저장/여러명 저장시 지우기
    '''
    # 7-2) FootballVideoProcessor로 어노테이션/트래킹 실행
    annotated_frame_gen = processor.process(frame_gen, fps=fps, kp_batch_size=16, buffer_size=300)

    ann = getattr(processor, "projection_annotator", getattr(processor, "annotator", None))
    if ann is not None:
        ann.fps = fps
        ann.save_owner_map = True
        ann.output_dir = "runs/soccer_owner_maps"

    # 7-3) 결과 비디오 저장
    success = save_video(annotated_frame_gen,args.output_video,fps)
    
    if not success:
        print(f"[ERROR] Failed to save video to {args.output_video}")
        print("[INFO] Possible solutions:")
        print("1.Check if the input video is valid and not corrupted")
        print("2.Try with a longer video (minimum 10 frames recommended)")
        print("3.Check if there are any processing errors in the logs")
        return
    
    print(f"[INFO] Successfully saved to {args.output_video} (FPS={fps})")
    
    # 성공적으로 비디오가 저장된 경우에만 후처리 진행
    
    # 7-4) 포메이션 자동 할당 실행 (포메이션 템플릿 사용)
    assign_formation_from_tracks(
        json_path=os.path.join(args.stub_dir, "object_tracks.json"),
        save_dir=args.stub_dir,
        formation0=args.formation0,  # 팀0 포메이션
        formation1=args.formation1   # 팀1 포메이션
    )
    
    # 7-5) 전체 영상 평균 포지션으로 Voronoi 생성
    print("\n[INFO] Creating Voronoi diagram for overall formation...")
    try:
        # 전체 평균 포지션 파일 경로
        avg0_path = os.path.join(args.stub_dir, "team0_overall_avg_pos.npy")
        avg1_path = os.path.join(args.stub_dir, "team1_overall_avg_pos.npy")
        meta0_path = os.path.join(args.stub_dir, "team0_pos.json")
        meta1_path = os.path.join(args.stub_dir, "team1_pos.json")

        if all(map(os.path.exists, [avg0_path, avg1_path, meta0_path, meta1_path])):
            # 평균 포지션 로드
            avg0 = np.load(avg0_path)
            avg1 = np.load(avg1_path)
            
            with open(meta0_path) as f0, open(meta1_path) as f1:
                meta0 = json.load(f0)
                meta1 = json.load(f1)

            # overall_formation.json에서 roles 정보 가져오기
            overall0_path = os.path.join(args.stub_dir, "team0_overall_formation.json")
            overall1_path = os.path.join(args.stub_dir, "team1_overall_formation.json")
            
            if os.path.exists(overall0_path) and os.path.exists(overall1_path):
                with open(overall0_path) as f0, open(overall1_path) as f1:
                    overall0 = json.load(f0)
                    overall1 = json.load(f1)
                
                roles0 = overall0.get('roles', ['P'] * len(avg0))
                roles1 = overall1.get('roles', ['P'] * len(avg1))
                
                # 실제 포메이션
                ground_truth0 = args.formation0.replace('-', '') if args.formation0 else None
                ground_truth1 = args.formation1.replace('-', '') if args.formation1 else None
                
                # 상위 포메이션
                top_formations0 = overall0.get('top_matches', [])
                top_formations1 = overall1.get('top_matches', [])
                
                # 프레임 범위 (전체 영상)
                frame_range = (0, len(meta0.get('player_ids', [])) * 100)
                
                # Voronoi 시각화
                combined_save_path = os.path.join(args.stub_dir, "overall_formation_voronoi.png")
                
                print(f"[INFO] Creating Voronoi for overall formation...")
                print(f"Team0: {len(avg0)} players, roles: {roles0}")
                print(f"Team1: {len(avg1)} players, roles: {roles1}")

                print(f"[DEBUG] Calling plot_formation_scatter...")
                print(f"[DEBUG] avg0 shape: {avg0.shape}, avg1 shape: {avg1.shape}")
                print(f"[DEBUG] frame_range: {frame_range}")

                plot_formation_scatter(
                    avg0, roles0,
                    avg1, roles1,
                    segment_id="Overall",
                    frame_range=frame_range,
                    top_formations0=top_formations0,
                    top_formations1=top_formations1,
                    ground_truth0=ground_truth0,
                    ground_truth1=ground_truth1,
                    save_path=combined_save_path
                )

                print(f"[INFO] Voronoi diagram saved to {combined_save_path}")
            else:
                print("[WARNING] Overall formation JSON files not found. Skipping Voronoi.")
        else:
            print("[WARNING] Overall average position files not found. Skipping Voronoi.")
            
    except Exception as e:
        print(f"[ERROR] Failed to create Voronoi diagram: {e}")
        import traceback
        traceback.print_exc()
    
    # 7-6) 포메이션 혼동행렬 및 성능 분석
    print("\n[INFO] Starting Formation Confusion Matrix Analysis...")
    try:
        # 설정 파일 경로 (없으면 기본값 사용)
        config_path = "formation_config.json" if os.path.exists("formation_config.json") else None
        confusion_analyzer = FormationConfusionAnalyzer(
            stubs_path=args.stub_dir,
            config_path=config_path
        )
        
        if config_path:
            print("[INFO] Using formation configuration from formation_config.json")
        else:
            print("[INFO] Using default universal formation groups")
        
        # 팀 색상 확인을 위한 로그 출력
        print("\n[INFO] Checking team color assignments...")
        tracks_path = os.path.join(args.stub_dir, "object_tracks.json")
        if os.path.exists(tracks_path):
            with open(tracks_path, 'r') as f:
                tracks = json.load(f)
            
            team_colors = {}
            if tracks and 'player' in tracks[0]:
                for pid, pdata in tracks[0]['player'].items():
                    tid = pdata.get('team_id')
                    tcolor = pdata.get('team_color')
                    if tid not in team_colors and tcolor:
                        team_colors[tid] = tcolor
                        b, g, r = tcolor
                        color_name = "Unknown"
                        if r > 150 and g < 100 and b < 100:
                            color_name = "Red"
                        elif b > 150 and g < 100 and r < 100:
                            color_name = "Blue"
                        elif g > 150 and r > 150:
                            color_name = "Yellow"
                        elif r > 200 and g > 200 and b > 200:
                            color_name = "White"
                        print(f"  Team{tid}: BGR={[int(b), int(g), int(r)]} → {color_name}")
            
            print("\n[NOTE] Team assignment based on colors above.")
            print("  Check output video to verify correct team mapping.")
            print("  If teams are swapped, re-run with formation order reversed.\n")
        
        # 실제 포메이션 설정 - 자동 팀 매핑 시도
        if args.formation0 and args.formation1 and team_colors:
            print("\n[INFO] Team-formation mapping...")
            
            # 단순하게 순서대로 매핑 (특정 팀 하드코딩 제거)
            team_mapping = {}
            ground_truth_formations = {}
            
            # 포메이션 정규화: 변형 포메이션을 base 포메이션으로 변환
            # confusion_analyzer의 normalize_formation 메서드 사용
            formation0_cleaned = args.formation0.replace('-', '')
            formation1_cleaned = args.formation1.replace('-', '')
            
            formation0_normalized = confusion_analyzer.normalize_formation(formation0_cleaned)
            formation1_normalized = confusion_analyzer.normalize_formation(formation1_cleaned)
            
            print(f"\n[INFO] Formation normalization:")
            # 포메이션 표시 형식
            formation0_display = confusion_analyzer.get_formation_display(args.formation0)
            formation1_display = confusion_analyzer.get_formation_display(args.formation1)
            
            print(f"  {args.team0}: {formation0_display}")
            if formation0_cleaned != formation0_normalized:
                print(f"    → Normalized: {formation0_cleaned} → {formation0_normalized} (base formation)")
            
            print(f"  {args.team1}: {formation1_display}")
            if formation1_cleaned != formation1_normalized:
                print(f"    → Normalized: {formation1_cleaned} → {formation1_normalized} (base formation)")
            
            # 팀0에 team0, 팀1에 team1 할당 (기본값)
            # 사용자가 비디오 확인 후 잘못되었으면 순서 바꿔서 재실행
            team_mapping[0] = args.team0
            team_mapping[1] = args.team1
            
            # 실제값은 원본 포메이션과 정규화된 포메이션 모두 저장
            ground_truth_formations = {
                'team0': formation0_cleaned,  # 원본 포메이션
                'team1': formation1_cleaned,  # 원본 포메이션
                'team0_normalized': formation0_normalized,  # 정규화된 포메이션
                'team1_normalized': formation1_normalized   # 정규화된 포메이션
            }
            
            print("\n[INFO] Current team-formation mapping:")
            print(f"Team0 → {args.team0}: {formation0_display}")
            print(f"Team1 → {args.team1}: {formation1_display}")
            print("\n[IMPORTANT] Verify team assignment in output video:")
            print(" - Check if team colors match the actual teams")
            print(" - If teams are incorrectly assigned, re-run with swapped parameters:")
            print(f"python main.py {args.input_videos} \\")
            print(f"--team0 '{args.team1}' --formation0 '{args.formation1}' \\")
            print(f"--team1 '{args.team0}' --formation1 '{args.formation0}'" )
            
            confusion_analyzer.ground_truth_formations = ground_truth_formations
            
        else:
            # 수동 설정 또는 기본값 사용
            print("\n[WARNING] Formations not provided. Using default or manual mapping.")
            print("  To enable automatic mapping, use:")
            print(f"  --formation0 <{args.team0}_formation> --formation1 <{args.team1}_formation>")
            
            # 포메이션 정규화: 변형 포메이션을 base 포메이션으로 변환
            if args.formation0:
                formation0_cleaned = args.formation0.replace('-', '')
                formation0_normalized = confusion_analyzer.normalize_formation(formation0_cleaned)
                formation0_display = confusion_analyzer.get_formation_display(args.formation0)
                print(f"  Team0 ({args.team0}): {formation0_display}")
            else:
                formation0_normalized = '442'
                print(f"  Team0 ({args.team0}): 442 (default)")
                
            if args.formation1:
                formation1_cleaned = args.formation1.replace('-', '')
                formation1_normalized = confusion_analyzer.normalize_formation(formation1_cleaned)
                formation1_display = confusion_analyzer.get_formation_display(args.formation1)
                print(f"  Team1 ({args.team1}): {formation1_display}")
            else:
                formation1_normalized = '433'
                print(f"  Team1 ({args.team1}): 433 (default)")
            
            confusion_analyzer.ground_truth_formations = {
                'team0': formation0_cleaned if args.formation0 else '442',
                'team1': formation1_cleaned if args.formation1 else '433',
                'team0_normalized': formation0_normalized,
                'team1_normalized': formation1_normalized
            }
        
        # 분석 실행
        formation_metrics = confusion_analyzer.analyze_all_teams()
        
        if formation_metrics:
            print(f"\n[INFO] Formation analysis completed successfully")
            print(f"[INFO] Results saved in {args.stub_dir}/")
            print(f"- Role confusion matrices: team*_role_confusion_matrix.png")
            print(f"- Formation confusion matrices: team*_formation_confusion_matrix.png")
            print(f"- Comparison plot: formation_analysis_comparison.png")
            print(f"- Analysis report: formation_analysis_report.txt")
            
            # 간단한 요약 출력 (파생 관계 포함)
            for metrics in formation_metrics:
                team_id = metrics['team_id']
                team_name = args.team0 if team_id == 0 else args.team1
                most_common = metrics['most_common_formation']
                most_common_display = confusion_analyzer.get_formation_display(most_common)
                
                print(f"\n  Team{team_id} ({team_name}):")
                print(f"- Ground truth: {metrics['ground_truth_formation']} ({confusion_analyzer.get_formation_display(metrics['ground_truth_formation'])})")
                print(f"- Most predicted: {most_common_display}")
                print(f"- Average similarity: {metrics['avg_similarity']:.3f}")
                print(f"- Strict accuracy: {metrics.get('strict_accuracy', metrics['accuracy']):.1%}")
                print(f"- Flexible accuracy (normalized): {metrics.get('flexible_accuracy', 0):.1%}")
                
                # 정규화 후 일치 여부
                gt_norm = metrics.get('ground_truth_normalized')
                most_common_norm = confusion_analyzer.normalize_formation(most_common)
                if gt_norm == most_common_norm:
                    print(f"Ground truth and prediction belong to same family ({gt_norm})")
                else:
                    print(f"Different families: {gt_norm} vs {most_common_norm}")
        else:
            print("[WARNING] No formation data available for confusion matrix analysis")
            
    except Exception as e:
        print(f"[ERROR] Formation confusion matrix analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()