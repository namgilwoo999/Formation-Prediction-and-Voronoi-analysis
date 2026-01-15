축구 경기 영상에서 **포메이션을 자동으로 예측하고 전술 분석**을 수행하는 AI 시스템

YOLOv11 객체 탐지와 시공간 분석을 결합하여 **4-4-2, 4-2-3-1, 3-5-2** 등의 포메이션을 자동으로 분류하고, Voronoi 다이어그램, 타임라인, 분포 차트로 시각화합니다.

---

## Problem Statement

### Background
축구에서 포메이션은 팀의 전술적 의도를 나타내는 핵심 요소입니다. 전통적으로 포메이션 분석은 전문가가 경기 영상을 반복 시청하며 수동으로 기록하는 방식으로 이루어져 왔으며, **1경기 분석에 평균 8-10시간**이 소요됩니다.

### Problem
- **수동 분석의 비효율성**: 전문 분석가의 인건비와 시간 소요가 막대함
- **주관적 판단**: 분석가마다 포메이션 해석이 달라 일관성 부족
- **소규모 팀의 접근성**: 유소년·아마추어 팀은 전문 분석 도구를 사용할 예산 없음

### Objective
경기 영상을 입력받아 **포메이션을 자동으로 인식**하는 시스템 개발
- **자동 분류**: 4-4-2, 4-2-3-1, 3-5-2 등 주요 포메이션 자동 인식
- **시각화**: Voronoi 다이어그램, 타임라인, 분포 차트로 직관적 이해 지원

### Value Proposition
- **시간 절감**: 8-10시간 → **단시간 내** 자동 분석
- **비용 절감**: 전문 분석가 없이도 고품질 전술 분석 가능
- **객관성**: 알고리즘 기반으로 일관된 분석 결과 제공
- **접근성**: 유소년·아마추어 팀도 사용 가능한 오픈소스 솔루션

### Approach
- **YOLOv11** 객체 탐지로 선수/공 추적
- **YOLOv8-pose** 경기장 keypoints 탐지
- **호모그래피 변환**으로 실제 구장 좌표 매핑
- **유클리드 거리 기반 Hungarian 알고리즘**으로 포메이션 템플릿 매칭
- **FotMob과 Sofascore**으로 포메이션 Top-N Accuracy, MRR 정량 분석

---

## 주요 기능

### 1. 자동 포메이션 예측
- **4-4-2, 4-2-3-1, 3-5-2, 4-3-3** 등 주요 포메이션 자동 분류
- 세그먼트별 평균 위치를 템플릿과 정합하여 정확도 향상

### 2. Static/TTI 공간 점유 평가 
- Voronoi 기반 정적 공간 동적 공간 비교

### 3. 시각화
- **Voronoi 다이어그램**: 선수별 점유 영역 표시
- **타임라인**: 경기 전체 흐름과 포메이션 시점

### 4. 통합 파이프라인
경기 영상 → 객체 탐지 → 추적 → 좌표 변환 → 팀 할당 → 포메이션 분류 → Voronoi 공간 점유 평가를 자동으로 수행

---

## Quick Start

### 1. 환경 설정

```bash
# Conda 환경 생성 (권장)
conda create -n football_analysis python=3.10 -y
conda activate football_analysis

# PyTorch 설치 (CUDA 11.8)
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 필수 패키지 설치
pip install ultralytics==8.3.* opencv-python==4.10.* numpy==1.26.* scipy==1.15.* \
            scikit-learn==1.6.* pandas==2.2.* matplotlib==3.10.*
```

### 2. 모델 다운로드

YOLOv11 모델 파일을 `models/` 폴더에 배치하세요:
- **객체 탐지 모델**: `models/best.pt` (선수, 골키퍼, 심판, 공 탐지용)
- **키포인트 모델**: `models/keypoints_best.pt` (경기장 라인 탐지용)

> **참고**: 모델 파일은 별도로 제공되거나 직접 학습해야 합니다.

### 3. 실행

```bash
# 기본 사용법
python main.py input_videos/your_match.mp4 \
    --output_video output_videos/result.mp4 \
    --team0 "팀A" \
    --team1 "팀B"

# 포메이션 지정 (FotMob이나 Sofascore 실제 포메이션 입력)
python main.py input_videos/your_match.mp4 \
    --output_video output_videos/result.mp4 \
    --team0 "Bayern Munich" \
    --team1 "Ajax" \
    --formation0 "4-2-3-1" \
    --formation1 "4-3-3"
```

### 4. 결과 확인

실행 후 다음 파일들이 생성됩니다:
- `output_videos/result.mp4`: 주석이 추가된 경기 영상
- `stubs/팀A_vs_팀B/tracks.json`: 선수 추적 데이터
- `stubs/팀A_vs_팀B/formations_team0.npy`: 팀0 포메이션 데이터
- `stubs/팀A_vs_팀B/voronoi_*.png`: Voronoi 다이어그램
- `stubs/팀A_vs_팀B/formation_timeline.png`: 포메이션 타임라인


## 출력 결과 설명

### 1. 주석이 추가된 영상 (`output_videos/*.mp4`)
- 선수, 골키퍼, 심판, 공에 바운딩 박스 표시
- 팀별 색상으로 구분
- 추적 ID 표시

### 2. 추적 데이터 (`stubs/*/tracks.json`)
프레임별 객체 위치, ID, 팀 정보 JSON 형식 저장

### 3. 포메이션 데이터 (`stubs/*/formations_team*.npy`)
세그먼트별 평균 위치, 할당된 포메이션, 유사도 점수

### 4. Voronoi 다이어그램 (`stubs/*/voronoi_*.png`)
선수별 점유 영역을 색상으로 표시

## 시스템 요구사항

### 최소 요구사항
- **OS**: Windows 10/11, Ubuntu 20.04+
- **Python**: 3.10 이상
- **RAM**: 8GB 이상
- **저장공간**: 5GB 이상 (모델 + 데이터)

### 권장 사양
- **GPU**: NVIDIA RTX 3060 이상 (CUDA 11.8)
- **RAM**: 16GB 이상
- **저장공간**: 20GB 이상

---

## 고급 사용법

### Voronoi 분석
```bash
python improved_voronoi_analysis.py
```
Voronoi 점유 영역 분석


## 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **객체 탐지** | YOLOv11 (Ultralytics) |
| **추적** | ByteTrack (IoU 기반) |
| **좌표 변환** | Homography (OpenCV) |
| **포메이션 분류** | L2 Norm + Hungarian 알고리즘 |
| **팀 할당** | KMeans 클러스터링 (LAB 색공간) |
| **시각화** | Matplotlib, OpenCV |

---

## Troubleshooting

### 1. 모델 로딩 오류
```
Error: Can't load model from models/best.pt
```
**해결**: `models/best.pt` 파일이 존재하는지 확인하세요.

### 2. CUDA 오류
```
RuntimeError: CUDA out of memory
```
**해결**: 배치 크기를 줄이거나 CPU 모드로 실행하세요.

### 3. 포메이션 인식 실패
**원인**: 경기장 키포인트 탐지 실패
**해결**: `--stub_dir` 옵션으로 중간 결과를 확인하고 키포인트 모델을 재학습하세요.

---

## License

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.

---

## Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) - 객체 탐지
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 다중 객체 추적

---

**Last Updated**: 2026-01-08

