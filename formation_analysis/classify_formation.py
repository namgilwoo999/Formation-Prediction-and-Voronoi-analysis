import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment

from .load_formation import load_formations

def scale_pos(pos):
    
    x_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    y_scaler = MinMaxScaler(feature_range=(0.0, 0.7))
    
    x_scaled = x_scaler.fit_transform(pos.T[0].reshape(-1, 1))
    y_scaled = y_scaler.fit_transform(pos.T[1].reshape(-1, 1))
    
    pos_scaled = np.hstack((x_scaled, y_scaled))
    
    return pos_scaled

def compute_similarity(form1, form2):
    
    # 최소-최대 정규화
    # x: 0.0~1.0
    # y: 0.0~0.7
    form1 = scale_pos(form1)
    form2 = scale_pos(form2)
    
    # 유사도 행렬 생성
    sim_mat = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            norm = np.linalg.norm(form1[i] - form2[j], 2) # L2 norm
            m = 1 - ((norm**2) / (1/3))
            sim_mat[i, j] = np.max([m, 0])
            
    # Hungarian 알고리즘으로 유사도 계산
    row_ind, col_ind = linear_sum_assignment(-sim_mat)
    similarity = sim_mat[row_ind, col_ind].sum() / 10
    
    return similarity, col_ind

class FormationClassification:
    def __init__(self, pos, player_ids):
        self.pos = pos
        self.player_ids = player_ids
        self.form_summary = None
        self.assign_mat = None
        self.roles = None

    def compute_form_summary(self):
        
        pos_role = self.pos.copy()
        
        for i in range(2): # one iteration is enough
            # 각 역할에 대한 다변량 정규 확률변수 계산
            rvs = []
            for role in range(10):
                mean = [np.mean(pos_role[:, role, 0]), np.mean(pos_role[:, role, 1])]
                cov = np.cov(pos_role[:, role, 0], pos_role[:, role, 1])
                rvs.append(multivariate_normal(mean, cov, allow_singular=True))
                
            # 각 위치의 로그 확률에 기반한 비용 행렬 구성
            # 비용 행렬을 사용하여 선수에게 레이블 할당
            self.assign_mat = np.zeros((10, 10))
            for f in range(len(pos_role)):
                cost_mat = np.zeros((10, 10))
                for role in range(10):
                    cost_mat[role] = rvs[role].logpdf(pos_role[f])

                # Hungarian 알고리즘 실패를 방지하기 위해 -inf 및 nan 값 대체
                cost_mat = np.nan_to_num(cost_mat, nan=-1e10, posinf=-1e10, neginf=-1e10)

                # Hungarian 알고리즘을 실행하여 각 선수에게 역할 레이블 할당
                row_ind, col_ind = linear_sum_assignment(-cost_mat)
                pos_role[f] = pos_role[f, col_ind]
                # 역할 할당 행렬의 카운트 추가
                for row, col in zip(row_ind, col_ind):
                    self.assign_mat[row, col] += 1
                    
            # 10개 역할의 좌표 계산
            self.form_summary = np.mean(pos_role, axis=0)
            
            # 최소-최대 정규화
            # x: 0.0~1.0
            # y: 0.0~0.7
            self.form_summary = scale_pos(self.form_summary)
        
        forms, form_names, forwards, DF = load_formations()
        sims = []
        assigns = []
        for form in forms:
            sim, assign = compute_similarity(self.form_summary, form)
            sims.append(sim)
            assigns.append(assign)

        rank_idx = np.argsort(sims)[::-1]
        self.top_formation_name = form_names[rank_idx[0]]
        self.top_similarity_score = sims[rank_idx[0]]
        self.roles = [forms[rank_idx[0]][role][2] for role in assigns[rank_idx[0]]]
        
        # 상위 k개 포메이션 정보 저장
        self.top_k_formations = []
        for i in range(min(10, len(rank_idx))):  # 상위 10개까지 저장
            self.top_k_formations.append([
                form_names[rank_idx[i]], 
                sims[rank_idx[i]]
            ])
    
    def get_top_k_formations(self, k=5):
        """
        상위 k개의 포메이션과 유사도 반환
        Returns:
            list: [[formation_name, similarity], ...]
        """
        if hasattr(self, 'top_k_formations'):
            return self.top_k_formations[:k]
        else:
            return []

    def visualize_form_summary(self,title=None,save_path=None):
        
        
        # 포메이션 템플릿 데이터 로드
        forms, form_names, forwards, DF = load_formations()
        
        # 포메이션 요약과 각 포메이션 템플릿 간의 유사도 계산
        sims = []
        assigns = []
        for form in forms:
            sim, assign = compute_similarity(self.form_summary, form)
            sims.append(sim)
            assigns.append(assign)
            
        # 유사도로 인덱스 정렬
        rank_idx = np.argsort(sims)[::-1]
        
        # 할당된 역할 리스트 생성
        self.roles = [forms[rank_idx[0]][role][2] for role in assigns[rank_idx[0]]]

        # visualize the formation summary
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        plt.subplots_adjust(wspace=.3, hspace=.3)
        
        if title is not None:
            axes[0].set_title(title)
        else:
            axes[0].set_title("Visual Formation Summary")

        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_xlim(-0.8, 1.2)
        axes[0].set_ylim(-0.2, 0.9)
        axes[0].plot([-0.1, -0.1], [-0.1, 0.8], color="black", linewidth=.1)
        axes[0].plot([1.1, 1.1], [-0.1, 0.8], color="black", linewidth=.1)
        axes[0].plot([-0.1, 1.1], [-0.1, -0.1], color="black", linewidth=.1)
        axes[0].plot([-0.1, 1.1], [0.8, 0.8], color="black", linewidth=.1)
        for i, role in enumerate(self.roles):
            if np.isin(role, forwards):
                axes[0].scatter(self.form_summary[i, 0], self.form_summary[i, 1], color="red")
            elif np.isin(role, DF):
                axes[0].scatter(self.form_summary[i, 0], self.form_summary[i, 1], color="blue")
            else:
                axes[0].scatter(self.form_summary[i, 0], self.form_summary[i, 1], color="green")
                
        # annotate the role labels
        for i, (x, y) in enumerate(self.form_summary):
            axes[0].annotate(self.roles[i], (x+.01, y+.01))
            
        # show the similarity toward top 5 formation
        for i in range(5):
            axes[0].text(-0.7, 0.5-i*0.1, "{}. {} - Similarity: {:.3f}".format(i+1, form_names[rank_idx[i]], sims[rank_idx[i]]))
            
        # show the role assignment matrix
        sns.heatmap(self.assign_mat,
                    fmt='1.0f',
                    cmap="Greens",
                    annot=True,
                    cbar=False,
                    xticklabels=self.player_ids,
                    yticklabels=self.roles
                )

        axes[1].set_title("Role assignments")
        axes[1].set_xlabel("Player")
        axes[1].set_ylabel("Role")
        axes[1].tick_params(axis='x', rotation=45)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"[INFO] Saved visualization to {save_path}")
            plt.close()
        else:
            plt.show()