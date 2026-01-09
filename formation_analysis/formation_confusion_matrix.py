"""
포메이션 혼동행렬 분석
포메이션 인식 결과에 대한 혼동행렬 생성 및 성능 메트릭 계산
main.py와 연동하여 실제 포메이션 데이터 분석
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# load_formation.py에서 템플릿 import
try:
    from .load_formation import load_formations
except ImportError:
    from load_formation import load_formations

class FormationConfusionAnalyzer:
    """포메이션 인식 결과 혼동행렬 분석"""
    
    def __init__(self, stubs_path: str = "stubs", config_path: str = None):
        self.stubs_path = Path(stubs_path)
        
        # formation_config.json 파일 필수
        if not config_path:
            config_path = "formation_config.json"
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.formation_groups = self._load_formation_groups_from_config(config)
            self.formation_families = config.get('formation_groups', {})
        else:
            print(f"[WARNING] {config_path} not found. Using empty formation groups.")
            self.formation_groups = {}
            self.formation_families = {}
        
        self.formation_templates = self._get_all_formations()
        self.ground_truth_formations = {}
    
    def _load_formation_groups_from_config(self, config: Dict) -> Dict:
        """설정 파일에서 포메이션 그룹 로드
        변형 포메이션 -> base 포메이션 매핑 생성
        """
        groups = {}
        for family_name, family_data in config.get('formation_groups', {}).items():
            base = family_data.get('base')
            variants = family_data.get('variants', [])
            if base and variants:
                # 각 변형 포메이션을 base로 매핑
                for variant in variants:
                    groups[variant] = base
        return groups
    
    def _get_all_formations(self) -> Dict:
        """모든 포메이션 템플릿 생성 - load_formation.py의 템플릿 사용"""
        templates = {}
        
        # 1. load_formation.py에서 기본 템플릿 로드 (9개)
        try:
            forms, form_names, _, _ = load_formations()
            for name in form_names:
                # 포메이션 이름에서 숫자 추출 (예: '433' -> [4, 3, 3])
                nums = [int(c) for c in name if c.isdigit()]
                if len(nums) >= 2:
                    templates[name] = nums
            print(f"[INFO] Loaded {len(templates)} templates from load_formation.py")
        except Exception as e:
            print(f"[WARNING] Could not load templates from load_formation.py: {e}")
            # 폴백: 기본 템플릿 직접 정의
            templates = {
                '343': [3, 4, 3],
                '352': [3, 5, 2],
                '41212': [4, 1, 2, 1, 2],
                '4231': [4, 2, 3, 1],
                '442': [4, 4, 2],
                '4213': [4, 2, 1, 3],
                '4123': [4, 1, 2, 3],
                '541': [5, 4, 1],
                '532': [5, 3, 2]
            }
        
        # 2. formation_config.json의 변형 포메이션들도 추가
        # (템플릿에 없는 변형들만 추가)
        for variant, base in self.formation_groups.items():
            if variant not in templates:
                variant_clean = variant.replace('-', '')
                if variant_clean.isdigit():
                    nums = [int(c) for c in variant_clean if c.isdigit()]
                    if len(nums) >= 2:
                        templates[variant] = nums
        
        # 3. 추가 기본 포메이션 (템플릿에 없는 경우에만)
        additional_templates = {
            '433': [4, 3, 3],
            '4141': [4, 1, 4, 1]
        }
        
        for name, nums in additional_templates.items():
            if name not in templates:
                templates[name] = nums
        
        print(f"[INFO] Total formations available: {len(templates)}")
        return templates
    
    def normalize_formation(self, formation: str) -> str:
        """
        변형 포메이션을 base 포메이션으로 변환
        예: 4231 → 433, 532 → 352, 451 → 433
        """
        # 하이픈 제거
        formation = formation.replace('-', '')
        
        # formation_groups에서 base 포메이션 찾기
        if formation in self.formation_groups:
            return self.formation_groups[formation]
        
        # 그룹에 없으면 원본 반환
        return formation
    
    def get_formation_display(self, formation: str) -> str:
        """
        포메이션 표시 형식 반환
        base 포메이션: "433 (기본 포메이션)"
        변형 포메이션: "4231 (433에서 파생됨)"
        """
        formation_clean = formation.replace('-', '')
        base = self.normalize_formation(formation)
        
        if formation_clean == base:
            return f"{formation} (기본 포메이션)"
        else:
            return f"{formation} ({base}에서 파생됨)"
    
    def find_best_match_with_ground_truth(self, top_matches: List, ground_truth: str) -> Tuple[str, float, bool]:
        """
        top_matches 리스트에서 ground_truth와 가장 잘 맞는 포메이션 찾기
        모든 포메이션을 base로 변환 후 비교
        
        Returns:
            tuple: (best_formation, similarity, is_match)
        """
        if not top_matches:
            return ('unknown', 0.0, False)
        
        # 실제값 정규화 (base 포메이션으로 변환)
        ground_truth_norm = self.normalize_formation(ground_truth)
        
        # top_matches를 순회하며 첫 번째 매치 찾기
        for formation, similarity in top_matches:
            # 예측된 포메이션도 base로 변환
            formation_norm = self.normalize_formation(formation)
            if formation_norm == ground_truth_norm:
                return (formation, similarity, True)
        
        # 매치가 없으면 가장 높은 유사도 반환
        return (top_matches[0][0], top_matches[0][1], False)
    
    def calculate_flexible_accuracy(self, predicted: list, actual: list) -> dict:
        """
        유연한 정확도 계산 (유사 포메이션 허용)
        
        Returns:
            dict: strict_accuracy, flexible_accuracy, partial_matches
        """
        if not predicted:
            return {
                'strict_accuracy': 0,
                'flexible_accuracy': 0,
                'partial_matches': [],
                'strict_correct': 0,
                'flexible_correct': 0,
                'total': 0
            }
        
        # 엄격한 정확도 (정확히 일치)
        strict_correct = sum(1 for p, a in zip(predicted, actual) if p == a)
        strict_accuracy = strict_correct / len(predicted)
        
        # 유연한 정확도 (유사 포메이션 허용)
        flexible_correct = 0
        partial_matches = []
        
        for pred, act in zip(predicted, actual):
            # 정규화된 포메이션 비교
            norm_pred = self.normalize_formation(pred)
            norm_act = self.normalize_formation(act)
            
            if norm_pred == norm_act:
                flexible_correct += 1
                if pred != act:  # 다르지만 같은 그룹
                    partial_matches.append((pred, act, norm_pred))
        
        flexible_accuracy = flexible_correct / len(predicted)
        
        return {
            'strict_accuracy': strict_accuracy,
            'flexible_accuracy': flexible_accuracy,
            'partial_matches': partial_matches,
            'strict_correct': strict_correct,
            'flexible_correct': flexible_correct,
            'total': len(predicted)
        }
    
    def load_segment_formations(self, team_id: int) -> List[Dict]:
        """세그먼트별 포메이션 데이터 로드 (개선된 버전)"""
        formations = []
        
        # CSV 파일에서 세그먼트별 포메이션 로드
        csv_path = self.stubs_path / f"team{team_id}_formation_segments.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                formations.append({
                    'segment': row.get('segment', len(formations)),
                    'formation': row.get('formation', None),
                    'frames': row.get('frames', 0)
                })
        
        # 개별 세그먼트 메타 파일에서 추가 정보 로드
        for segment_id in range(1, 20):
            meta_path = self.stubs_path / f"team{team_id}_segment{segment_id}_meta.json"
            if meta_path.exists() and os.path.getsize(meta_path) > 0:
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    # top_matches 전체 저장 (순차 비교용)
                    if 'top_matches' in meta and meta['top_matches']:
                        # 해당 세그먼트 업데이트 또는 추가
                        found = False
                        for form_data in formations:
                            if form_data.get('segment') == segment_id:
                                form_data['top_matches'] = meta['top_matches']  # 전체 리스트 저장
                                form_data['predicted_formation'] = meta['top_matches'][0][0]
                                form_data['similarity'] = meta['top_matches'][0][1]
                                found = True
                                break
                        
                        if not found:
                            formations.append({
                                'segment': segment_id,
                                'top_matches': meta['top_matches'],  # 전체 리스트 저장
                                'predicted_formation': meta['top_matches'][0][0],
                                'similarity': meta['top_matches'][0][1],
                                'frame_range': meta.get('frame_range', [])
                            })
                except Exception as e:
                    print(f"  Warning: Failed to load {meta_path}: {e}")
        
        # 세그먼트 데이터가 없으면 overall 데이터 로드
        if not formations:
            formations = self.load_overall_formations(team_id)
                    
        return formations
    
    def load_overall_formations(self, team_id: int) -> List[Dict]:
        """전체 포메이션 데이터 로드 (세그먼트 없이)"""
        formations = []
        
        # overall_formation.json에서 전체 포메이션 로드
        overall_path = self.stubs_path / f"team{team_id}_overall_formation.json"
        if overall_path.exists():
            try:
                with open(overall_path, 'r') as f:
                    overall_data = json.load(f)
                
                if 'top_matches' in overall_data and overall_data['top_matches']:
                    formations.append({
                        'segment': 'Overall',
                        'top_matches': overall_data['top_matches'],
                        'predicted_formation': overall_data['top_matches'][0][0],
                        'similarity': overall_data['top_matches'][0][1],
                        'frame_range': 'Full video'
                    })
                    print(f"  Loaded overall formation data for Team{team_id}")
            except Exception as e:
                print(f"  Warning: Failed to load {overall_path}: {e}")
        
        return formations
    
    def create_role_confusion_matrix(self, team_id: int) -> Optional[np.ndarray]:
        """역할 할당 혼동행렬 생성"""
        try:
            # 역할 할당 데이터 로드
            pos_json_path = self.stubs_path / f"team{team_id}_pos.json"
            if not pos_json_path.exists():
                return None
                
            with open(pos_json_path, 'r') as f:
                meta = json.load(f)
            
            player_ids = meta.get("player_ids", [])
            
            # 각 세그먼트의 역할 할당 수집
            role_assignments = {}
            for segment_id in range(1, 20):
                meta_path = self.stubs_path / f"team{team_id}_segment{segment_id}_meta.json"
                if meta_path.exists() and os.path.getsize(meta_path) > 0:
                    with open(meta_path, 'r') as f:
                        segment_meta = json.load(f)
                    
                    roles = segment_meta.get('roles', [])
                    for player_id, role in zip(player_ids, roles):
                        if player_id not in role_assignments:
                            role_assignments[player_id] = []
                        role_assignments[player_id].append(role)
            
            # 세그먼트 데이터가 없으면 overall 데이터 사용
            if not role_assignments:
                overall_path = self.stubs_path / f"team{team_id}_overall_formation.json"
                if overall_path.exists():
                    with open(overall_path, 'r') as f:
                        overall_data = json.load(f)
                    
                    roles = overall_data.get('roles', [])
                    for player_id, role in zip(player_ids, roles):
                        if player_id not in role_assignments:
                            role_assignments[player_id] = []
                        role_assignments[player_id].append(role)
            
            if not role_assignments:
                return None
                
            # 혼동행렬 생성
            all_roles = sorted(list(set(sum(role_assignments.values(), []))))
            confusion_mat = np.zeros((len(player_ids), len(all_roles)), dtype=int)
            
            for i, player_id in enumerate(player_ids):
                if player_id in role_assignments:
                    for role in role_assignments[player_id]:
                        if role in all_roles:
                            j = all_roles.index(role)
                            confusion_mat[i, j] += 1
            
            # 시각화
            plt.figure(figsize=(12, 8))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Greens',
                       xticklabels=all_roles,
                       yticklabels=[f"Player {pid}" for pid in player_ids])
            plt.title(f'Team{team_id} - Role Assignment Matrix')
            plt.xlabel('Assigned Role')
            plt.ylabel('Player ID')
            plt.tight_layout()
            
            output_path = self.stubs_path / f"team{team_id}_role_confusion_matrix.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved role assignment matrix: {output_path}")
            
            return confusion_mat
            
        except Exception as e:
            print(f"  Warning: Could not create role confusion matrix: {e}")
            return None
    
    def create_formation_confusion_matrix(self, team_id: int, 
                                         formations: List[Dict]) -> Optional[np.ndarray]:
        """포메이션 예측 혼동행렬 생성 (Ground Truth와 비교)"""
        
        predicted = []
        actual = []
        
        # 실제 포메이션
        ground_truth = self.ground_truth_formations.get(f'team{team_id}', '442')
        ground_truth_norm = self.normalize_formation(ground_truth)
        
        # top_matches를 고려한 예측 수집
        for form_data in formations:
            # top_matches가 있으면 ground_truth와 가장 잘 맞는 것 선택
            if 'top_matches' in form_data:
                best_formation, _, _ = self.find_best_match_with_ground_truth(
                    form_data['top_matches'], ground_truth
                )
                predicted.append(best_formation)
            else:
                pred_formation = form_data.get('predicted_formation') or form_data.get('formation')
                if pred_formation:
                    predicted.append(pred_formation)
                else:
                    continue
            actual.append(ground_truth)
        
        if not predicted:
            print(f"  No predicted formations for team{team_id}")
            return None
        
        # 사용된 포메이션들
        all_formations = sorted(list(set(predicted + actual)))
        
        # 혼동행렬 생성
        cm = confusion_matrix(actual, predicted, labels=all_formations)
        
        # 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=all_formations,
                   yticklabels=all_formations)
        plt.title(f'Team{team_id} - Formation Prediction Confusion Matrix')
        plt.xlabel('Predicted Formation')
        plt.ylabel('Actual Formation (Ground Truth)')
        plt.tight_layout()
        
        output_path = self.stubs_path / f"team{team_id}_formation_confusion_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved formation confusion matrix: {output_path}")
        
        # 정확도 계산
        accuracy = accuracy_score(actual, predicted)
        print(f"  Formation prediction accuracy: {accuracy:.1%}")
        
        return cm
    
    def calculate_metrics(self, team_id: int, formations: List[Dict]) -> Dict:
        """성능 메트릭 계산 (개선된 버전)"""
        
        predicted = []
        similarities = []
        matched_predictions = []  # 실제값과 매치된 예측
        
        # 실제값 (원본 포메이션)
        ground_truth = self.ground_truth_formations.get(f'team{team_id}', '442')
        ground_truth_norm = self.ground_truth_formations.get(f'team{team_id}_normalized', 
                                                              self.normalize_formation(ground_truth))
        
        for form_data in formations:
            # top_matches를 활용한 순차 비교
            if 'top_matches' in form_data:
                best_formation, similarity, is_match = self.find_best_match_with_ground_truth(
                    form_data['top_matches'], ground_truth
                )
                predicted.append(best_formation)
                similarities.append(similarity)
                if is_match:
                    matched_predictions.append(best_formation)
            else:
                pred_formation = form_data.get('predicted_formation') or form_data.get('formation')
                if pred_formation:
                    predicted.append(pred_formation)
                    similarities.append(form_data.get('similarity', 0))
        
        if not predicted:
            return {}
        
        actual = [ground_truth] * len(predicted)
        actual_norm = [ground_truth_norm] * len(predicted)
        
        # 유연한 정확도 계산 (정규화된 포메이션으로 비교)
        flex_results = self.calculate_flexible_accuracy(predicted, actual)  # predicted를 내부에서 정규화
        
        # 분류 보고서
        try:
            report = classification_report(actual, predicted, 
                                          output_dict=True, 
                                          zero_division=0)
        except:
            report = {'accuracy': 0}
        
        metrics = {
            'team_id': team_id,
            'team_name': f'Team{team_id}',
            'total_segments': len(formations),
            'predicted_segments': len(predicted),
            'avg_similarity': np.mean(similarities) if similarities else 0,
            'max_similarity': np.max(similarities) if similarities else 0,
            'min_similarity': np.min(similarities) if similarities else 0,
            'most_common_formation': max(set(predicted), key=predicted.count) if predicted else 'N/A',
            'ground_truth_formation': ground_truth,  # 원본 포메이션 표시
            'ground_truth_normalized': ground_truth_norm,  # 정규화된 포메이션
            'accuracy': report.get('accuracy', 0),
            'strict_accuracy': flex_results['strict_accuracy'],
            'flexible_accuracy': flex_results['flexible_accuracy'],
            'partial_matches': flex_results['partial_matches'],
            'matched_predictions': matched_predictions  # 추가: 실제값과 매치된 예측들
        }
        
        # 포메이션별 빈도
        formation_counts = {}
        for form in predicted:
            formation_counts[form] = formation_counts.get(form, 0) + 1
        metrics['formation_distribution'] = formation_counts
        
        return metrics
    
    def create_comparison_plot(self, metrics_list: List[Dict]):
        """팀 간 비교 시각화"""
        
        if len(metrics_list) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 평균 유사도 비교
        ax = axes[0, 0]
        teams = [f"Team{m['team_id']}" for m in metrics_list]
        similarities = [m['avg_similarity'] for m in metrics_list]
        bars = ax.bar(teams, similarities, color=['#FF6B6B', '#4ECDC4'])
        ax.set_ylabel('Average Similarity Score')
        ax.set_title('Formation Recognition Confidence')
        ax.set_ylim([0, 1])
        for bar, sim in zip(bars, similarities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{sim:.3f}', ha='center', va='bottom')
        
        # 2. 포메이션 분포
        ax = axes[0, 1]
        for i, m in enumerate(metrics_list):
            if 'formation_distribution' in m:
                formations = list(m['formation_distribution'].keys())
                counts = list(m['formation_distribution'].values())
                x_pos = np.arange(len(formations))
                width = 0.35
                offset = width * (i - 0.5)
                ax.bar(x_pos + offset, counts, width, 
                      label=f"Team{m['team_id']}", alpha=0.8)
        ax.set_xlabel('Formation')
        ax.set_ylabel('Frequency')
        ax.set_title('Formation Distribution')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(formations, rotation=45)
        ax.legend()
        
        # 3. 세그먼트 수 비교
        ax = axes[1, 0]
        segment_counts = [m['predicted_segments'] for m in metrics_list]
        bars = ax.bar(teams, segment_counts, color=['#95E1D3', '#FFA502'])
        ax.set_ylabel('Number of Segments')
        ax.set_title('Analyzed Segments per Team')
        for bar, count in zip(bars, segment_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(count), ha='center', va='bottom')
        
        # 4. 메트릭 요약 테이블
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for m in metrics_list:
            table_data.append([
                f"Team{m['team_id']}",
                m['most_common_formation'],
                f"{m['avg_similarity']:.3f}",
                f"{m['accuracy']:.1%}"
            ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Team', 'Most Common', 'Avg Similarity', 'Accuracy'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.suptitle('Formation Analysis Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        output_path = self.stubs_path / "formation_analysis_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved comparison plot: {output_path}")
    
    def generate_report(self, metrics_list: List[Dict]) -> str:
        """분석 리포트 생성 (포메이션 정규화 정보 포함)"""
        
        report = "=" * 60 + "\n"
        report += "FORMATION RECOGNITION ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for m in metrics_list:
            team_id = m['team_id']
            report += f"Team {team_id}:\n"
            report += "-" * 30 + "\n"
            
            # 실제 포메이션 표시 (파생 관계 포함)
            gt = m['ground_truth_formation']
            gt_norm = m.get('ground_truth_normalized', self.normalize_formation(gt))
            gt_display = self.get_formation_display(gt)
            report += f"  Ground Truth Formation: {gt_display}\n"
            if gt != gt_norm:
                report += f"    → Normalized to: {gt_norm} (base formation)\n"
            
            # 가장 많이 예측된 포메이션 (파생 관계 포함)
            most_common = m['most_common_formation']
            most_common_norm = self.normalize_formation(most_common)
            most_common_display = self.get_formation_display(most_common)
            report += f"  Most Predicted Formation: {most_common_display}\n"
            if most_common != most_common_norm:
                report += f"    → Normalized to: {most_common_norm} (base formation)\n"
            
            # 정확도 표시 및 설명
            report += f"  Strict Accuracy: {m.get('strict_accuracy', m['accuracy']):.1%}\n"
            report += f"  Flexible Accuracy (with normalization): {m.get('flexible_accuracy', 0):.1%}\n"
            
            # 유연한 정확도가 높은 이유 설명
            if m.get('flexible_accuracy', 0) > m.get('strict_accuracy', 0):
                report += f"    → Higher flexible accuracy because formations in same family are considered equal\n"
                if gt_norm == most_common_norm:
                    report += f"    → {gt} and {most_common} both belong to {gt_norm} family\n"
            
            report += f"  Average Similarity Score: {m['avg_similarity']:.3f}\n"
            
            # 세그먼트 수에 따라 표현 변경
            if m['predicted_segments'] == 1:
                report += f"  Analysis Type: Overall formation (full video)\n"
            else:
                report += f"  Total Segments Analyzed: {m['predicted_segments']}\n"
            
            # 실제값과 매치된 예측 표시
            if m.get('matched_predictions'):
                report += f"  Predictions matching Ground Truth (after normalization):\n"
                unique_matches = list(set(m['matched_predictions']))
                for pred in unique_matches:
                    pred_display = self.get_formation_display(pred)
                    report += f"    - {pred_display}\n"
            
            # 부분 일치 설명 개선
            if m.get('partial_matches'):
                report += f"  Partial Matches (same family but different variant):\n"
                seen = set()
                for pred, actual, group in m['partial_matches'][:3]:
                    if (pred, actual) not in seen:
                        seen.add((pred, actual))
                        report += f"    - {pred} ≈ {actual} (both are {group} variants)\n"
            
            # 포메이션 분포
            if 'formation_distribution' in m:
                report += f"  Formation Distribution:\n"
                for form, count in m['formation_distribution'].items():
                    percentage = (count / m['predicted_segments']) * 100
                    form_display = self.get_formation_display(form)
                    form_norm = self.normalize_formation(form)
                    
                    # 실제값과 일치 여부 표시
                    match_marker = ""
                    if form_norm == gt_norm:
                        match_marker = " ✓" if form == gt else " ≈"
                    
                    report += f"    - {form_display}: {count} ({percentage:.1f}%){match_marker}\n"
                
                # 설명 추가
                report += "\n  Legend: ✓ = exact match, ≈ = same family\n"
            report += "\n"
        
        report += "=" * 60 + "\n"
        
        # 리포트 저장
        report_path = self.stubs_path / "formation_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Saved analysis report: {report_path}")
        
        return report
    
    def analyze_all_teams(self):
        """전체 팀 분석 실행"""
        
        print("\n=== Formation Confusion Matrix Analysis ===")
        
        all_metrics = []
        
        for team_id in [0, 1]:
            print(f"\nAnalyzing Team{team_id}...")
            
            # 세그먼트별 포메이션 데이터 로드
            formations = self.load_segment_formations(team_id)
            
            if formations:
                print(f"  Found {len(formations)} formation segments")
                
                # 역할 할당 혼동행렬
                self.create_role_confusion_matrix(team_id)
                
                # 포메이션 예측 혼동행렬
                self.create_formation_confusion_matrix(team_id, formations)
                
                # 메트릭 계산
                metrics = self.calculate_metrics(team_id, formations)
                
                # 추가 출력
                print(f"Strict Accuracy: {metrics.get('strict_accuracy', 0):.1%}")
                print(f"Flexible Accuracy: {metrics.get('flexible_accuracy', 0):.1%}")
                if metrics.get('partial_matches'):
                    print(f"  Found {len(metrics['partial_matches'])} partial matches")
                
                all_metrics.append(metrics)
            else:
                print(f"  No formation data found for Team{team_id}")
        
        if all_metrics:
            # 비교 플롯 생성
            self.create_comparison_plot(all_metrics)
            
            # 리포트 생성
            report = self.generate_report(all_metrics)
            print("\n" + report)
            
            # 메트릭 JSON 저장
            metrics_path = self.stubs_path / "formation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"✓ Saved metrics: {metrics_path}")
        
        return all_metrics


# 단독 실행 시
if __name__ == "__main__":
    analyzer = FormationConfusionAnalyzer(stubs_path="stubs")
    metrics = analyzer.analyze_all_teams()
    
    print("\n=== Analysis Complete ===")
    if metrics:
        print(f"Successfully analyzed {len(metrics)} teams")
    else:
        print("No data available for analysis")
