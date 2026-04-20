"""
Seed the database from the CSV dataset.

Reads athlete_metadata_50.csv and inserts data into the normalized tables:
  athletes -> training_profiles -> engineered_features

Optionally processes videos in performance_video/videos/ and populates:
  video_uploads -> video_analysis

Usage:
  python seed_data.py                  # metadata only
  python seed_data.py --with-video     # metadata + video processing
"""

import argparse
import os
import sys
import re

import mysql.connector
import pandas as pd

from config import DB_CONFIG, VIDEO_DIR
from feature_engineering import engineer_features, compute_age


def seed_metadata(csv_path: str):
    """Load CSV and insert into athletes, training_profiles, engineered_features."""
    df = pd.read_csv(csv_path)
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    inserted = 0

    for _, row in df.iterrows():
        try:
            # -- 1. athletes table --
            dob = row["date_of_birth"]  # already YYYY-MM-DD in CSV

            cursor.execute(
                """
                INSERT INTO athletes
                    (athlete_code, full_name, date_of_birth, biological_gender,
                     height_cm, weight_kg, leg_length_cm)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE athlete_code = athlete_code
                """,
                (
                    row["athlete_code"],
                    row["full_name"],
                    dob,
                    row["biological_gender"],
                    float(row["height_cm"]),
                    float(row["weight_kg"]),
                    float(row["leg_length_cm"]),
                ),
            )
            athlete_id = cursor.lastrowid

            # If ON DUPLICATE KEY hit, fetch existing id
            if athlete_id == 0:
                cursor.execute(
                    "SELECT athlete_id FROM athletes WHERE athlete_code = %s",
                    (row["athlete_code"],),
                )
                athlete_id = cursor.fetchone()[0]

            # -- 2. training_profiles table --
            cursor.execute(
                """
                INSERT INTO training_profiles
                    (athlete_id, years_of_training, pb_100m_s, pb_400m_s, pb_5k_min,
                     resting_heart_rate, vo2_max, injury_history, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    athlete_id,
                    int(row["years_of_training"]),
                    float(row["pb_100m_s"]),
                    float(row["pb_400m_s"]),
                    float(row["pb_5k_min"]),
                    int(row["resting_heart_rate"]),
                    float(row["vo2_max"]),
                    int(row["injury_history"]),
                    row.get("notes", None),
                ),
            )
            profile_id = cursor.lastrowid

            # -- 3. engineered_features table --
            age = compute_age(dob)
            feats = engineer_features(
                height_cm=float(row["height_cm"]),
                weight_kg=float(row["weight_kg"]),
                leg_length_cm=float(row["leg_length_cm"]),
                age=age,
                years_of_training=int(row["years_of_training"]),
                pb_100m_s=float(row["pb_100m_s"]),
                resting_heart_rate=int(row["resting_heart_rate"]),
                vo2_max=float(row["vo2_max"]),
            )

            cursor.execute(
                """
                INSERT INTO engineered_features
                    (athlete_id, profile_id, age_at_computation,
                     bmi, leg_height_ratio, exp_age_ratio,
                     performance_index_100m, heart_rate_score, vo2_max_normalized)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    athlete_id,
                    profile_id,
                    age,
                    feats["bmi"],
                    feats["leg_height_ratio"],
                    feats["exp_age_ratio"],
                    feats["performance_index_100m"],
                    feats["heart_rate_score"],
                    feats["vo2_max_normalized"],
                ),
            )

            inserted += 1
            conn.commit()

        except Exception as e:
            print(f"  ERROR on row {row.get('athlete_code', '?')}: {e}")
            conn.rollback()

    cursor.close()
    conn.close()
    print(f"✅ Seeded {inserted} athletes into 3 tables.")


def seed_videos():
    """Scan performance_video/videos/, process each video, store in DB."""
    # Add performance_video folder to path so we can import model_processor
    pv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "performance_video")
    sys.path.insert(0, pv_dir)
    from model_processor import process_video

    video_dir = VIDEO_DIR  # "performance_video/videos" from config.py

    if not os.path.isdir(video_dir):
        print(f"⚠️  Video directory not found: {video_dir}")
        return

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    processed = 0
    failed = 0

    for fname in sorted(os.listdir(video_dir)):
        if not fname.endswith((".mp4", ".mov")):
            continue

        match = re.search(r"athlete_(\d+)", fname)
        if not match:
            continue

        # Map file number to athlete_code -> athlete_id
        file_num = int(match.group(1))
        athlete_code = f"ATH{file_num:02d}"

        cursor.execute(
            "SELECT athlete_id FROM athletes WHERE athlete_code = %s",
            (athlete_code,),
        )
        result = cursor.fetchone()
        if not result:
            print(f"  ⏭  No athlete for code {athlete_code}, skipping {fname}")
            continue

        athlete_id = result[0]
        file_path = os.path.join(video_dir, fname)

        # -- video_uploads --
        try:
            cursor.execute(
                """
                INSERT INTO video_uploads (athlete_id, file_name, file_path, status)
                VALUES (%s, %s, %s, 'processing')
                """,
                (athlete_id, fname, file_path),
            )
            video_id = cursor.lastrowid
            conn.commit()

            # -- process video --
            print(f"  Processing {fname} (athlete_id={athlete_id})...")
            analysis = process_video(file_path)

            if analysis:
                cursor.execute(
                    """
                    INSERT INTO video_analysis
                        (video_id, athlete_id,
                         max_left_knee_flexion, max_right_knee_flexion,
                         max_hip_extension, max_ankle_dorsiflexion, avg_trunk_lean,
                         symmetry_index, stride_variance,
                         total_left_steps, total_right_steps, cadence_spm)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        video_id,
                        athlete_id,
                        analysis["max_left_knee_flexion"],
                        analysis["max_right_knee_flexion"],
                        analysis["max_hip_extension"],
                        analysis["max_ankle_dorsiflexion"],
                        analysis["avg_trunk_lean"],
                        analysis["symmetry_index"],
                        analysis["stride_variance"],
                        analysis["total_left_steps"],
                        analysis["total_right_steps"],
                        analysis["cadence_spm"],
                    ),
                )
                cursor.execute(
                    "UPDATE video_uploads SET status = 'completed' WHERE video_id = %s",
                    (video_id,),
                )
                processed += 1
            else:
                cursor.execute(
                    "UPDATE video_uploads SET status = 'failed' WHERE video_id = %s",
                    (video_id,),
                )
                failed += 1

            conn.commit()

        except Exception as e:
            print(f"  ERROR processing {fname}: {e}")
            conn.rollback()
            failed += 1

    cursor.close()
    conn.close()
    print(f"✅ Processed {processed} videos. ({failed} failed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="athlete_metadata_50.csv", help="Path to CSV")
    parser.add_argument("--with-video", action="store_true", help="Also process videos")
    args = parser.parse_args()

    print("=" * 50)
    print("  Seeding metadata...")
    print("=" * 50)
    seed_metadata(args.csv)

    if args.with_video:
        print()
        print("=" * 50)
        print("  Processing videos...")
        print("=" * 50)
        seed_videos()