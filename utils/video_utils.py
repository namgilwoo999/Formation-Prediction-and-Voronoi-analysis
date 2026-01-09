import cv2
import os
import itertools
import time

def read_video_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Can't open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Input video FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def save_video(frames, output_path, fps):
    frames = iter(frames)

    # 시작 시간 기록
    start_time = time.time()

    # 첫 프레임 가져오기 시도
    try:
        first_frame = next(frames)
    except StopIteration:
        print("[WARNING] No frames to save - generator is empty")
        print("[WARNING] This may happen if:")
        print("1. The input video is too short or corrupted")
        print("2. The processing failed to generate any frames")
        print("3. All frames were filtered out during processing")
        return False

    height, width = first_frame.shape[:2]

    # 비디오 라이터 생성
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not writer.isOpened():
        print(f"[ERROR] Failed to create video writer for {output_path}")
        return False

    # 프레임 쓰기
    frame_count = 1
    writer.write(first_frame)
    last_print_time = start_time

    for frame in frames:
        if frame is None:
            print(f"\n[WARNING] Skipping None frame at position {frame_count}")
            continue
        writer.write(frame)
        frame_count += 1

        # 1초마다 진행률 출력
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            elapsed = current_time - start_time
            processing_fps = frame_count / elapsed
            print(f"\rProcessing: {frame_count} frames | {processing_fps:.2f} FPS | Elapsed: {elapsed:.1f}s", end='', flush=True)
            last_print_time = current_time

    writer.release()

    if frame_count == 0:
        print(f"\n[WARNING] No valid frames were written to {output_path}")
        return False

    # 최종 통계 출력
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print(f"\n[INFO] Video saved to {output_path}")
    print(f"[INFO] Total frames: {frame_count}")
    print(f"[INFO] Processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"[INFO] Processing speed: {avg_fps:.2f} FPS")
    print(f"[INFO] Output video FPS: {fps}")

    return True

