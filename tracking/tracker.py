import os
import sys
import pickle
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import supervision as sv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.bbox_utils import get_bbox_center,get_bbox_width,is_valid_bbox

def bbox_iou(b1, b2):
    #IoU 계산
    xA = max(b1[0], b2[0]); yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2]); yB = min(b1[3], b2[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    inter = interW * interH
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0

class Tracker:
    def __init__(
        self,
        model_path: str,
        project_dir,
        video_name: str,
        max_lost_frames: int = 120,
        reassign_dist: float = 100.0,
        color_dist_thresh: float = 30.0,
        iou_thresh: float = 0.3
    ):
        # 1) YOLO 모델 & ByteTrack 초기화
        self.model   = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # 3) ID 리링크 설정
        self.max_lost          = max_lost_frames
        self.reassign_dist     = reassign_dist
        self.color_dist_thresh = color_dist_thresh
        self.iou_thresh        = iou_thresh
        self.lost_buffer       = {}   # old_id -> {bbox, missed, mean_color}
        self.id_map            = {}   # cur_id -> orig_id

        # 4) 카메라 누적 이동량
        self.cum_mov = None

        # 5) detect_frames 에 원본 frames 저장
        self.frames  = None

    def set_camera_movement(self, cumulative_movement: np.ndarray):
        """
        cumulative_movement: (N_frames,2) frame0에서 framei까지의 dx,dy 배열
        """
        self.cum_mov = cumulative_movement

    def _mean_color(self, frame, bbox):
        x1,y1,x2,y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(3, dtype=float)
        return roi.reshape(-1,3).mean(axis=0)

    def _update_lost_buffer(self, frame_idx, active_ids, current_bboxes):
        # 1) active old_id 제거
        for old_id in list(self.lost_buffer):
            if old_id in active_ids:
                del self.lost_buffer[old_id]

        # 2) 신규 cur_id 리링크 시도 (IoU OR (spatial+color))
        frame = self.frames[frame_idx]
        for cur_id, bbox in current_bboxes.items():
            if cur_id in self.id_map:
                continue
            cur_color = self._mean_color(frame, bbox)
            cx1, cy1  = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            for old_id, info in list(self.lost_buffer.items()):
                lb         = info['bbox']
                cx2, cy2   = ((lb[0]+lb[2])/2, (lb[1]+lb[3])/2)
                spatial    = np.hypot(cx1-cx2, cy1-cy2)
                color_dist = np.linalg.norm(cur_color - info['mean_color'])
                iou        = bbox_iou(bbox, lb)
                # 거리와 색상만 사용
                if (spatial < self.reassign_dist and
                    color_dist < self.color_dist_thresh):
                    print(f"[REMAP] frame {frame_idx}: {cur_id}→{old_id}"
                          f" (IoU {iou:.2f}, dist {spatial:.1f}, col {color_dist:.1f})")
                    self.id_map[cur_id] = old_id
                    del self.lost_buffer[old_id]
                    break

        # 3) missed++ 및 만료 처리
        for old_id in list(self.lost_buffer):
            self.lost_buffer[old_id]['missed'] += 1
            if self.lost_buffer[old_id]['missed'] > self.max_lost:
                del self.lost_buffer[old_id]

    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for f_idx, frame_tracks in enumerate(obj_tracks):
                for tid, info in frame_tracks.items():
                    info['position'] = get_bbox_center(info['bbox'])

    def interpolate_ball_positions(self, ball_positions):
        coords = []
        for entry in ball_positions:
            b = entry.get(1, {}).get('bbox')
            coords.append(b if b and len(b)==4 else [np.nan]*4)
        df = pd.DataFrame(coords, columns=['x1','y1','x2','y2'])
        df = df.interpolate(limit_direction="both").bfill().ffill().fillna(0)
        return [{1:{'bbox':row}} for row in df.to_numpy().tolist()]

    def detect_frames(self, frames):
        self.frames = frames
        dets = []
        for idx, frame in enumerate(frames):
            #  YOLO 예측
            det = self.model.predict(frame, conf=0.1)
            dets.append(det[0] if isinstance(det, list) else det)

            if (idx+1) % 100 == 0:
                print(f"[DEBUG] Processed frame {idx+1}/{len(frames)}")
        return dets

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """트랙킹 전체 파이프라인"""
        if read_from_stub and stub_path and os.path.exists(stub_path):
            return pickle.load(open(stub_path,'rb'))

        detections = self.detect_frames(frames)
        print(f"[DEBUG] Detections: {len(detections)}/{len(frames)}")

        tracks = {
            'players':     [],
            'referees':    [],
            'goalkeepers': [],
            'ball':        []
        }

        for i, det in enumerate(detections):
            ds        = sv.Detections.from_ultralytics(det)
            names     = det.names
            inv_names = {v:k for k,v in names.items()}

            raw_tracks = self.tracker.update_with_detections(ds)

            # 프레임 슬롯 초기화
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['goalkeepers'].append({})
            tracks['ball'].append({})

            # current active bboxes (ROI 내부만)
            current_bboxes = {}
            for tr in raw_tracks:
                bbox = tr[0]; tid = tr[4]
                current_bboxes[tid] = bbox.tolist()
            active_ids = set(current_bboxes)

            # ID 리링크
            self._update_lost_buffer(i, active_ids, current_bboxes)

            # players / referees / goalkeepers 저장
            for tr in raw_tracks:
                cid, tid = tr[3], tr[4]
                bbox     = tr[0].tolist()
                cls_name = names[cid]
                orig     = self.id_map.get(tid, tid)
                if tid not in self.id_map:
                    self.id_map[tid] = orig
                
                info = {'bbox': bbox, 'class_name': cls_name}

                if cls_name == 'player':
                    tracks['players'][i][orig]     = {'bbox': bbox}
                elif cls_name == 'referee':
                    tracks['referees'][i][orig]    = {'bbox': bbox}
                elif cls_name == 'goalkeeper':
                    tracks['goalkeepers'][i][orig] = {'bbox': bbox}

            # lost_buffer 추가 (disappear)
            for orig in list(self.id_map.values()):
                if orig not in active_ids and orig not in self.lost_buffer:
                    prev = i-1
                    lb = None
                    if prev >= 0:
                        if orig in tracks['players'][prev]:
                            lb = tracks['players'][prev][orig]['bbox']
                        elif orig in tracks['goalkeepers'][prev]:
                            lb = tracks['goalkeepers'][prev][orig]['bbox']
                    if lb is not None:
                        self.lost_buffer[orig] = {
                            'bbox': lb,
                            'missed': 0,
                            'mean_color': self._mean_color(frames[prev], lb)
                        }

            # ball detection (전체 프레임)
            for ent in ds:
                cid  = ent[3]
                bbox = ent[0].tolist()
                if cid == inv_names['ball']:
                    tracks['ball'][i][1] = {'bbox': bbox}

        # 누락된 프레임 패딩
        n = len(frames)
        for k in tracks:
            while len(tracks[k]) < n:
                tracks[k].append({})

        if stub_path:
            pickle.dump(tracks, open(stub_path,'wb'))
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        if not is_valid_bbox(bbox):
            return frame
        x_center, _ = get_bbox_center(bbox)
        y2          = int(bbox[3])
        w           = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(w), int(0.35*w)),
            angle=0, startAngle=-45, endAngle=235,
            color=color, thickness=2, lineType=cv2.LINE_4
        )
        if track_id is not None:
            rw, rh = 40, 20
            x1 = x_center - rw//2
            y1 = y2     - rh//2 + 15
            x2 = x_center + rw//2
            y2_ = y1 + rh
            cv2.rectangle(
                frame, (int(x1),int(y1)), (int(x2),int(y2_)),
                color, cv2.FILLED
            )
            tx = x1 + (2 if track_id>99 else 12)
            cv2.putText(
                frame, str(track_id), (int(tx),int(y1+15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        if not is_valid_bbox(bbox):
            return frame
        x, _ = get_bbox_center(bbox)
        y    = int(bbox[1])
        pts  = np.array([[x,y],[x-10,y-20],[x+10,y-20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0,0,0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        out = []
        for i, frame in enumerate(video_frames):
            img     = frame.copy()
            p_dict  = tracks['players'][i]
            r_dict  = tracks['referees'][i]
            gk_dict = tracks['goalkeepers'][i]
            b_dict  = tracks['ball'][i]

            for tid, info in p_dict.items():
                img = self.draw_ellipse(
                    img, info['bbox'],
                    info.get('team_color',(0,0,255)),
                    tid
                )
                if info.get('has_ball', False):
                    img = self.draw_triangle(img, info['bbox'], (0,0,255))

            for info in r_dict.values():
                img = self.draw_ellipse(img, info['bbox'], (0,255,255))
            for info in gk_dict.values():
                img = self.draw_ellipse(img, info['bbox'], (255,255,0))  # GK 색상
            for info in b_dict.values():
                img = self.draw_triangle(img, info['bbox'], (0,255,0))

            out.append(img)
        return out
