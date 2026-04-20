"""
Video analysis using MediaPipe PoseLandmarker (Tasks API).
Compatible with mediapipe >= 0.10.20.

Requires: pose_landmarker_heavy.task model file.
Download from:
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

Extracts biomechanical features from athlete performance videos:
  - Joint angles (knee, hip, ankle, trunk)
  - Symmetry index (left vs right knee)
  - Stride metrics (variance, cadence, step counts)
"""

import os
import cv2
import numpy as np
from scipy.signal import find_peaks

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# ─── Model path ──────────────────────────────────────────────────────────────
# Looks in the same directory as this script, then in the project root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, "pose_landmarker_full.task"),
    os.path.join(_SCRIPT_DIR, "..", "pose_landmarker_full.task"),
    "pose_landmarker_full.task",
]
MODEL_PATH = None
for _p in _MODEL_CANDIDATES:
    if os.path.isfile(_p):
        MODEL_PATH = _p
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "pose_landmarker_heavy.task not found. Download it:\n"
        "  curl -o pose_landmarker_heavy.task "
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )

# ─── Landmark indices (same as legacy 33-point model) ────────────────────────
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


def _calculate_angle(a, b, c) -> float:
    """Angle at point b formed by segments ba and bc, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / np.pi)
    return float(360.0 - angle if angle > 180.0 else angle)


def process_video(video_path: str) -> dict | None:
    """
    Run MediaPipe PoseLandmarker on a video and return extracted biomechanical features.

    Returns None if no pose is detected in any frame.
    Returns a dict with keys matching the video_analysis table columns.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Create PoseLandmarker for VIDEO mode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # Collectors for per-frame measurements
    left_knee_angles = []
    right_knee_angles = []
    hip_angles = []
    ankle_angles = []
    trunk_angles = []

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Convert to MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp in milliseconds
        timestamp_ms = int(frame_idx * 1000.0 / fps)

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            frame_idx += 1
            continue

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            frame_idx += 1
            continue

        # First detected pose
        landmarks = result.pose_landmarks[0]

        def coord(idx):
            lm = landmarks[idx]
            return [lm.x * w, lm.y * h]

        # Key landmarks
        shoulder_l = coord(LEFT_SHOULDER)
        hip_l = coord(LEFT_HIP)
        hip_r = coord(RIGHT_HIP)
        knee_l = coord(LEFT_KNEE)
        knee_r = coord(RIGHT_KNEE)
        ankle_l = coord(LEFT_ANKLE)
        ankle_r = coord(RIGHT_ANKLE)
        toe_l = coord(LEFT_FOOT_INDEX)

        # Joint angles
        left_knee_angles.append(_calculate_angle(hip_l, knee_l, ankle_l))
        right_knee_angles.append(_calculate_angle(hip_r, knee_r, ankle_r))
        hip_angles.append(_calculate_angle(shoulder_l, hip_l, knee_l))
        ankle_angles.append(_calculate_angle(knee_l, ankle_l, toe_l))

        # Trunk lean: angle between vertical line through shoulder and the shoulder-hip segment
        trunk_angles.append(_calculate_angle([shoulder_l[0], 0], shoulder_l, hip_l))

        frame_idx += 1

    cap.release()
    landmarker.close()

    if not left_knee_angles:
        print(f"  WARNING: No poses detected in {video_path}")
        return None

    # ─── Derived metrics ────────────────────────────────────────────────────
    lk = np.array(left_knee_angles)
    rk = np.array(right_knee_angles)

    # Step detection via knee-flexion minima
    left_steps, _ = find_peaks(-lk, distance=int(fps * 0.3))
    right_steps, _ = find_peaks(-rk, distance=int(fps * 0.3))

    total_left = len(left_steps)
    total_right = len(right_steps)

    # Max flexion = minimum angle (deeper bend)
    max_lk = float(np.min(lk))
    max_rk = float(np.min(rk))

    # Symmetry index: 0 = perfectly symmetric, higher = more asymmetric
    denom = (max_lk + max_rk) / 2.0
    symmetry_index = abs(max_lk - max_rk) / denom if denom != 0 else 0.0

    # Stride variance (temporal consistency of left steps)
    if total_left > 2:
        stride_times = np.diff(left_steps) / fps
        stride_variance = float(np.var(stride_times))
    else:
        stride_variance = 0.0

    # Cadence: steps per minute (average of both legs)
    total_frames = len(left_knee_angles)
    duration_min = total_frames / (fps * 60.0)
    total_steps = total_left + total_right
    cadence = round(total_steps / duration_min, 2) if duration_min > 0 else 0.0

    return {
        "max_left_knee_flexion": round(max_lk, 2),
        "max_right_knee_flexion": round(max_rk, 2),
        "max_hip_extension": round(float(np.max(hip_angles)), 2),
        "max_ankle_dorsiflexion": round(float(np.min(ankle_angles)), 2),
        "avg_trunk_lean": round(float(np.mean(trunk_angles)), 2),
        "symmetry_index": round(symmetry_index, 4),
        "stride_variance": round(stride_variance, 6),
        "total_left_steps": total_left,
        "total_right_steps": total_right,
        "cadence_spm": cadence,
    }


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_processor.py <video_path>")
        print("Example: python model_processor.py performance_video/videos/athlete_01.mp4")
        sys.exit(1)

    video = sys.argv[1]
    print(f"Processing: {video}")
    result = process_video(video)

    if result:
        print("\n✅ Results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("\n❌ No pose detected.")