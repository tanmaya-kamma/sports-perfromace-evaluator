import cv2
import numpy as np
import mediapipe as mp

from mediapipe.python.solutions.pose import Pose
from scipy.signal import find_peaks
import os
import mysql.connector   # ✅ ADDED

# ✅ DB CONNECTION (ADD YOUR PASSWORD)
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="tanmayakamma",   # ⚠️ change this to your MySQL password
    database="gait_analysis",
    port=3306
)
cursor = db.cursor()


# 🔹 Angle Calculation Function
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def process_video(video_path):
    pose_detector = Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if not cap.isOpened():
        print(f"❌ Video not opening: {video_path}")
        return None

    left_knee_angles = []
    right_knee_angles = []
    hip_angles = []
    ankle_angles = []
    trunk_angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            def get_coord(idx):
                lm = landmarks[idx]
                return [lm.x * w, lm.y * h]

            # LEFT
            hip_l = get_coord(23)
            knee_l = get_coord(25)
            ankle_l = get_coord(27)
            toe_l = get_coord(31)

            # RIGHT
            hip_r = get_coord(24)
            knee_r = get_coord(26)
            ankle_r = get_coord(28)

            # SHOULDER
            shoulder_l = get_coord(11)

            # Angles
            left_knee_angles.append(calculate_angle(hip_l, knee_l, ankle_l))
            right_knee_angles.append(calculate_angle(hip_r, knee_r, ankle_r))
            hip_angles.append(calculate_angle(shoulder_l, hip_l, knee_l))
            ankle_angles.append(calculate_angle(knee_l, ankle_l, toe_l))

            vertical_point = [shoulder_l[0], 0]
            trunk_angles.append(calculate_angle(vertical_point, shoulder_l, hip_l))

    cap.release()

    if not left_knee_angles:
        return "❌ No pose detected in video."

    left_knee_np = np.array(left_knee_angles)
    right_knee_np = np.array(right_knee_angles)

    # Step detection
    left_steps, _ = find_peaks(-left_knee_np, distance=15)

    # Metrics
    max_left_knee_flexion = np.min(left_knee_np)
    max_right_knee_flexion = np.min(right_knee_np)
    max_hip_extension = np.max(hip_angles) if hip_angles else 0
    max_ankle_dorsiflexion = np.min(ankle_angles) if ankle_angles else 0
    avg_trunk = np.mean(trunk_angles) if trunk_angles else 0

    # Symmetry Index
    if (max_left_knee_flexion + max_right_knee_flexion) != 0:
        SI = abs(max_left_knee_flexion - max_right_knee_flexion) / (
            (max_left_knee_flexion + max_right_knee_flexion) / 2
        )
    else:
        SI = 0

    # Stride Variance
    if len(left_steps) > 1:
        stride_times = np.diff(left_steps) / fps
        stride_variance = np.var(stride_times)
    else:
        stride_variance = 0.0

    # 🔥 STRIDE CLASSIFICATION
    if stride_variance < 0.015:
        stride_label = "Good stride (very consistent)"
    elif stride_variance < 0.035:
        stride_label = "Moderate stride (slightly variable)"
    else:
        stride_label = "Bad stride (irregular)"

    return {
        "Max Left Knee Flexion (deg)": round(max_left_knee_flexion, 2),
        "Max Right Knee Flexion (deg)": round(max_right_knee_flexion, 2),
        "Max Hip Extension (deg)": round(max_hip_extension, 2),
        "Max Ankle Dorsiflexion (deg)": round(max_ankle_dorsiflexion, 2),
        "Average Trunk Lean (deg)": round(avg_trunk, 2),
        "Symmetry Index (Lower is more symmetrical)": round(SI, 3),
        "Stride Variance (Lower = more regular)": round(stride_variance, 4),
        "Stride Analysis": stride_label,
        "Total Left Steps Detected": len(left_steps)
    }


# 🔹 RUN FOR ALL VIDEOS
if __name__ == "__main__":
    video_folder = "performance_video/videos"

    if not os.path.exists(video_folder):
        print(f"❌ Folder '{video_folder}' not found.")
    else:
        for file in os.listdir(video_folder):
            if file.endswith((".mp4", ".mov", ".avi")):
                video_path = os.path.join(video_folder, file)
                print(f"\n🔹 Processing: {file}")
                result = process_video(video_path)

                if isinstance(result, dict):
                    for k, v in result.items():
                        print(f"  {k}: {v}")

                    # ✅ INSERT INTO MYSQL
                    cursor.execute("""
                        INSERT INTO video_analysis
                        (video_name, max_left_knee, max_right_knee, max_hip, max_ankle, avg_trunk_lean, symmetry_index, stride_variance, stride_analysis, total_left_steps)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        file,
                        result["Max Left Knee Flexion (deg)"],
                        result["Max Right Knee Flexion (deg)"],
                        result["Max Hip Extension (deg)"],
                        result["Max Ankle Dorsiflexion (deg)"],
                        result["Average Trunk Lean (deg)"],
                        result["Symmetry Index (Lower is more symmetrical)"],
                        result["Stride Variance (Lower = more regular)"],
                        result["Stride Analysis"],
                        result["Total Left Steps Detected"]
                    ))

                    db.commit()   # ✅ SAVE TO DB

                else:
                    print(result)