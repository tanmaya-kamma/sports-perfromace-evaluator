"""
Centralized configuration for the Athlete Performance Evaluator.
All DB credentials, paths, and constants live here — single source of truth.
"""

import os

# ─── Database ───────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "tanmayakamma")
DB_NAME = os.getenv("DB_NAME", "athlete_perf_db")

DB_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

DB_CONFIG = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME,
    "port": DB_PORT,
}

# ─── Paths ──────────────────────────────────────────────────────────────────
VIDEO_DIR = os.getenv("VIDEO_DIR", "performance_video/videos")
MODEL_PATH = "athlete_rf_model.pkl"
FEATURE_LIST_PATH = "feature_list.pkl"

# ─── Feature Engineering Constants ──────────────────────────────────────────
# World record 100m (Usain Bolt) — used to normalize sprint performance
WR_100M = 9.58

# The canonical ordered feature list used for model training and prediction.
# ANY change here must be reflected in fusion_trainer.py and app.py.
MODEL_FEATURES = [
    # --- Metadata-derived (engineered_features table) ---
    "bmi",
    "leg_height_ratio",
    "exp_age_ratio",
    "performance_index_100m",
    "heart_rate_score",
    "vo2_max_normalized",
    # --- Video-derived (video_analysis table) ---
    "max_left_knee_flexion",
    "max_right_knee_flexion",
    "max_hip_extension",
    "max_ankle_dorsiflexion",
    "avg_trunk_lean",
    "symmetry_index",
    "stride_variance",
    "cadence_spm",
]