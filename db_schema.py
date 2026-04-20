"""
Database schema definition and initialization.

Tables (normalized):
  1. athletes              — immutable bio/profile data
  2. training_profiles     — training stats & physical benchmarks (can have multiple per athlete over time)
  3. engineered_features   — derived features computed from athletes + training_profiles
  4. video_uploads         — tracks every uploaded video file
  5. video_analysis        — MediaPipe pose features extracted from a video
  6. performance_scores    — RF model output linking engineered_features + video_analysis
  7. shap_explanations     — per-feature SHAP values for each score

Foreign-key chain:
  athletes ──< training_profiles
  athletes ──< engineered_features >── training_profiles
  athletes ──< video_uploads
  video_uploads ──< video_analysis
  athletes ──< performance_scores >── engineered_features
  performance_scores >── video_analysis
  performance_scores ──< shap_explanations
"""

import mysql.connector
from config import DB_CONFIG, DB_NAME

# ─── Raw SQL for each table ─────────────────────────────────────────────────

CREATE_DATABASE = f"CREATE DATABASE IF NOT EXISTS {DB_NAME};"

TABLES = {}

# 1. Core athlete identity — things that don't change (or change very rarely)
TABLES["athletes"] = """
CREATE TABLE IF NOT EXISTS athletes (
    athlete_id      INT AUTO_INCREMENT PRIMARY KEY,
    athlete_code    VARCHAR(20)  NOT NULL UNIQUE,
    full_name       VARCHAR(120) NOT NULL,
    date_of_birth   DATE         NOT NULL,
    biological_gender ENUM('Male','Female','Other') NOT NULL,
    height_cm       DECIMAL(5,1) NOT NULL,
    weight_kg       DECIMAL(5,1) NOT NULL,
    leg_length_cm   DECIMAL(5,1) NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;
"""

# 2. Training / fitness snapshot — one athlete can have many over time
TABLES["training_profiles"] = """
CREATE TABLE IF NOT EXISTS training_profiles (
    profile_id          INT AUTO_INCREMENT PRIMARY KEY,
    athlete_id          INT NOT NULL,
    years_of_training   INT NOT NULL,
    pb_100m_s           DECIMAL(5,2),
    pb_400m_s           DECIMAL(6,2),
    pb_5k_min           DECIMAL(5,2),
    resting_heart_rate  INT,
    vo2_max             DECIMAL(5,1),
    injury_history      TINYINT(1) DEFAULT 0,
    notes               TEXT,
    recorded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id) ON DELETE CASCADE
) ENGINE=InnoDB;
"""

# 3. Engineered features derived from profile + training data
TABLES["engineered_features"] = """
CREATE TABLE IF NOT EXISTS engineered_features (
    feature_id              INT AUTO_INCREMENT PRIMARY KEY,
    athlete_id              INT NOT NULL,
    profile_id              INT NOT NULL,
    age_at_computation      INT NOT NULL,
    bmi                     DECIMAL(5,2) NOT NULL,
    leg_height_ratio        DECIMAL(5,4) NOT NULL,
    exp_age_ratio           DECIMAL(5,4) NOT NULL,
    performance_index_100m  DECIMAL(5,4),
    heart_rate_score        DECIMAL(8,6),
    vo2_max_normalized      DECIMAL(5,4),
    computed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id) ON DELETE CASCADE,
    FOREIGN KEY (profile_id) REFERENCES training_profiles(profile_id) ON DELETE CASCADE
) ENGINE=InnoDB;
"""

# 4. Video file tracking
TABLES["video_uploads"] = """
CREATE TABLE IF NOT EXISTS video_uploads (
    video_id        INT AUTO_INCREMENT PRIMARY KEY,
    athlete_id      INT NOT NULL,
    file_name       VARCHAR(255) NOT NULL,
    file_path       VARCHAR(500) NOT NULL,
    status          ENUM('uploaded','processing','completed','failed') DEFAULT 'uploaded',
    uploaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id) ON DELETE CASCADE
) ENGINE=InnoDB;
"""

# 5. Pose-extraction results from MediaPipe
TABLES["video_analysis"] = """
CREATE TABLE IF NOT EXISTS video_analysis (
    analysis_id             INT AUTO_INCREMENT PRIMARY KEY,
    video_id                INT NOT NULL,
    athlete_id              INT NOT NULL,
    max_left_knee_flexion   DECIMAL(6,2),
    max_right_knee_flexion  DECIMAL(6,2),
    max_hip_extension       DECIMAL(6,2),
    max_ankle_dorsiflexion  DECIMAL(6,2),
    avg_trunk_lean          DECIMAL(6,2),
    symmetry_index          DECIMAL(6,4),
    stride_variance         DECIMAL(10,6),
    total_left_steps        INT,
    total_right_steps       INT,
    cadence_spm             DECIMAL(6,2),
    analyzed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id)   REFERENCES video_uploads(video_id) ON DELETE CASCADE,
    FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id) ON DELETE CASCADE
) ENGINE=InnoDB;
"""

# 6. Model prediction output
TABLES["performance_scores"] = """
CREATE TABLE IF NOT EXISTS performance_scores (
    score_id            INT AUTO_INCREMENT PRIMARY KEY,
    athlete_id          INT NOT NULL,
    feature_id          INT NOT NULL,
    analysis_id         INT NOT NULL,
    performance_score   DECIMAL(6,2) NOT NULL,
    model_version       VARCHAR(50) DEFAULT 'rf_v1',
    scored_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (athlete_id)  REFERENCES athletes(athlete_id) ON DELETE CASCADE,
    FOREIGN KEY (feature_id)  REFERENCES engineered_features(feature_id) ON DELETE CASCADE,
    FOREIGN KEY (analysis_id) REFERENCES video_analysis(analysis_id) ON DELETE CASCADE
) ENGINE=InnoDB;
"""

# 7. SHAP explanation per feature per score
TABLES["shap_explanations"] = """
CREATE TABLE IF NOT EXISTS shap_explanations (
    explanation_id      INT AUTO_INCREMENT PRIMARY KEY,
    score_id            INT NOT NULL,
    feature_name        VARCHAR(100) NOT NULL,
    feature_value       DECIMAL(10,4),
    shap_value          DECIMAL(10,6) NOT NULL,
    influence_direction ENUM('positive','negative') NOT NULL,

    FOREIGN KEY (score_id) REFERENCES performance_scores(score_id) ON DELETE CASCADE
) ENGINE=InnoDB;
"""


# ─── Execution ───────────────────────────────────────────────────────────────

def init_database():
    """Create the database and all tables in dependency order."""
    # Connect without specifying a database first
    conn_config = {k: v for k, v in DB_CONFIG.items() if k != "database"}
    conn = mysql.connector.connect(**conn_config)
    cursor = conn.cursor()

    cursor.execute(CREATE_DATABASE)
    cursor.execute(f"USE {DB_NAME};")

    for table_name, ddl in TABLES.items():
        print(f"  Creating table: {table_name}")
        cursor.execute(ddl)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\n✅ Database '{DB_NAME}' initialized with {len(TABLES)} tables.")


if __name__ == "__main__":
    init_database()