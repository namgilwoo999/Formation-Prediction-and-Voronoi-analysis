from .abstract_writer import AbstractWriter

import os
import json
import numpy as np
from typing import Any, List


class TracksJsonWriter(AbstractWriter):
    

    def __init__(self, save_dir: str = '', object_fname: str = 'object_tracks', 
                 keypoints_fname: str = 'keypoint_tracks') -> None:
        
        super().__init__()
        self.save_dir = save_dir
        self.obj_path = os.path.join(self.save_dir, f'{object_fname}.json')
        self.kp_path = os.path.join(self.save_dir, f'{keypoints_fname}.json')

        if os.path.exists(save_dir):
            self._remove_existing_files(files=[self.kp_path, self.obj_path]) 
        else:
            os.makedirs(save_dir)
    
    def get_object_tracks_path(self) -> str:
        """객체 트랙 JSON 파일의 경로를 반환합니다."""
        return self.obj_path
    
    def get_keypoints_tracks_path(self) -> str:
        """키포인트 트랙 JSON 파일의 경로를 반환합니다."""
        return self.kp_path

    def write(self, filename: str, tracks: Any) -> None:
        
        # 모든 트랙을 직렬화 가능한 형식으로 변환
        serializable_tracks = self._make_serializable(tracks)

        if os.path.exists(filename):
            # 파일이 존재하면 기존 데이터를 로드하고 새 트랙 추가
            with open(filename, 'r') as f:
                existing_data = json.load(f)
            existing_data.append(serializable_tracks)
            data_to_save = existing_data
        else:
            # 파일이 없으면 현재 트랙으로 새 리스트 생성
            data_to_save = [serializable_tracks]

        # 직렬화된 데이터를 파일에 쓰기
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)  # 가독성 향상을 위해 들여쓰기 추가

    def _make_serializable(self, obj: Any) -> Any:
        
        if isinstance(obj, dict):
            # 키와 값 모두 직렬화 가능한지 확인
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # 리스트를 재귀적으로 변환
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            # 튜플을 재귀적으로 변환
            return tuple(self._make_serializable(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            # numpy 배열을 리스트로 변환
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool8)):
            # numpy bool을 Python bool로 변환
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            # numpy int를 Python int로 변환
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            # numpy float를 Python float로 변환
            return float(obj)
        elif isinstance(obj, (int, float, bool, str, type(None))):
            # Python 네이티브 타입은 변환 불필요
            return obj
        else:
            # 알려진 타입이 아니면 문자열로 변환 시도
            return str(obj)
        
    def _remove_existing_files(self, files: List[str]) -> None:
        
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
