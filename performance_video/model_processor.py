import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.pose import Pose
from scipy.signal import find_peaks
import os
import mysql.connector
import re

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "tanmayakamma",
    "database": "athlete_db", # Ensure this database exists
    "port": 3306
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return round(float(angle), 2)

def process_video(video_path, athlete_id):
    pose_detector = Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    left_knee_angles, right_knee_angles, hip_angles, ankle_angles, trunk_angles = [], [], [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h, w, _ = frame.shape
        results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            def get_coord(idx):
                lm = results.pose_landmarks.landmark[idx]
                return [lm.x * w, lm.y * h]

            # Landmark coordinates
            hip_l, knee_l, ankle_l, toe_l = get_coord(23), get_coord(25), get_coord(27), get_coord(31)
            hip_r, knee_r, ankle_r = get_coord(24), get_coord(26), get_coord(28)
            shoulder_l = get_coord(11)

            # Calculations
            left_knee_angles.append(calculate_angle(hip_l, knee_l, ankle_l))
            right_knee_angles.append(calculate_angle(hip_r, knee_r, ankle_r))
            hip_angles.append(calculate_angle(shoulder_l, hip_l, knee_l))
            ankle_angles.append(calculate_angle(knee_l, ankle_l, toe_l))
            trunk_angles.append(calculate_angle([shoulder_l[0], 0], shoulder_l, hip_l))

    cap.release()
    if not left_knee_angles: return None

    # Performance Metrics
    left_knee_np = np.array(left_knee_angles)
    left_steps, _ = find_peaks(-left_knee_np, distance=15)
    
    max_l_knee = np.min(left_knee_np)
    max_r_knee = np.min(np.array(right_knee_angles))
    si = abs(max_l_knee - max_r_knee) / ((max_l_knee + max_r_knee) / 2) if (max_l_knee + max_r_knee) != 0 else 0
    stride_var = np.var(np.diff(left_steps) / fps) if len(left_steps) > 1 else 0.0

    return {
        "athlete_id": athlete_id,
        "max_left_knee": round(float(max_l_knee), 2),
        "max_right_knee": round(float(max_r_knee), 2),
        "max_hip": round(float(np.max(hip_angles)), 2),
        "max_ankle": round(float(np.min(ankle_angles)), 2),
        "avg_trunk_lean": round(float(np.mean(trunk_angles)), 2),
        "symmetry_index": round(float(si), 3),
        "stride_variance": round(float(stride_var), 4),
        "total_left_steps": len(left_steps)
    }

if __name__ == "__main__":
    db = mysql.connector.connect(**DB_CONFIG)
    cursor = db.cursor()
    video_folder = "performance_video/videos"

    for file in os.listdir(video_folder):
        if file.endswith((".mp4", ".mov")):
            # Regex to extract ID from "athlete_01.mp4"
            match = re.search(r'athlete_(\d+)', file)
            if match:
                athlete_id = int(match.group(1))
                print(f"Processing Athlete ID {athlete_id}...")
                result = process_video(os.path.join(video_folder, file), athlete_id)
                
                if result:
                    query = """
                        INSERT INTO video_analysis 
                        (athlete_id, max_left_knee, max_right_knee, max_hip, max_ankle, avg_trunk_lean, symmetry_index, stride_variance, total_left_steps)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(query, list(result.values()))
                    db.commit()
    cursor.close()
    db.close()