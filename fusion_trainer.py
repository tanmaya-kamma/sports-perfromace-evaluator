"""
Fusion trainer — merges metadata + video features and trains the Random Forest model.

Key fixes over the original:
  1. Target variable is a multi-factor composite — no single-feature leakage.
  2. Features list is imported from config.py (single source of truth).
  3. Scores are written back to performance_scores table with FK links.
  4. Model + feature list saved as versioned artifacts.

Usage:
  python fusion_trainer.py              # train + save model
  python fusion_trainer.py --score-all  # also write scores to DB for all athletes
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine, text

from config import DB_URL, MODEL_FEATURES, MODEL_PATH, FEATURE_LIST_PATH


engine = create_engine(DB_URL)


def build_dataset() -> pd.DataFrame:
    """
    Fetch and merge engineered_features + video_analysis + training_profiles.
    Returns a DataFrame with one row per athlete, all MODEL_FEATURES columns,
    plus raw columns needed for target computation.
    """
    query = """
        SELECT
            ef.athlete_id,
            ef.feature_id,
            ef.bmi,
            ef.leg_height_ratio,
            ef.exp_age_ratio,
            ef.performance_index_100m,
            ef.heart_rate_score,
            ef.vo2_max_normalized,
            va.analysis_id,
            va.max_left_knee_flexion,
            va.max_right_knee_flexion,
            va.max_hip_extension,
            va.max_ankle_dorsiflexion,
            va.avg_trunk_lean,
            va.symmetry_index,
            va.stride_variance,
            va.cadence_spm,
            tp.pb_100m_s,
            tp.pb_400m_s,
            tp.pb_5k_min,
            tp.vo2_max,
            tp.years_of_training
        FROM engineered_features ef
        JOIN video_analysis va ON ef.athlete_id = va.athlete_id
        JOIN training_profiles tp ON ef.profile_id = tp.profile_id
    """
    df = pd.read_sql(query, con=engine)

    if df.empty:
        raise RuntimeError("No fused data found. Seed metadata AND videos first.")

    return df


def compute_target(df: pd.DataFrame) -> pd.Series:
    """
    Build a composite performance score (0–100) from multiple independent dimensions.
    
    This avoids the original bug where performance_index was used both as
    an input feature AND as 50% of the target (direct leakage).

    Components (weighted):
      - Sprint speed     (30%) : lower pb_100m_s = better
      - Endurance        (15%) : lower pb_5k_min = better
      - Aerobic fitness  (20%) : higher vo2_max = better
      - Biomech symmetry (20%) : lower symmetry_index = more symmetric = better
      - Stride consistency(15%): lower stride_variance = better
    """

    def _normalize(series, lower_is_better=True):
        """Min-max normalize to [0, 1]. Handles constant series."""
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        normed = (series - mn) / (mx - mn)
        return (1 - normed) if lower_is_better else normed

    speed = _normalize(df["pb_100m_s"], lower_is_better=True)
    endurance = _normalize(df["pb_5k_min"], lower_is_better=True)
    fitness = _normalize(df["vo2_max"], lower_is_better=False)
    symmetry = _normalize(df["symmetry_index"], lower_is_better=True)
    stride = _normalize(df["stride_variance"], lower_is_better=True)

    composite = (
        speed * 0.30
        + endurance * 0.15
        + fitness * 0.20
        + symmetry * 0.20
        + stride * 0.15
    ) * 100

    # Small noise so the model doesn't memorize a perfect formula
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1.5, size=len(composite))
    composite = np.clip(composite + noise, 0, 100)

    return pd.Series(np.round(composite, 2), index=df.index)


def train():
    """Train the Random Forest model and evaluate via cross-validation."""
    print("📦 Building fused dataset...")
    df = build_dataset()
    print(f"   → {len(df)} athletes with complete data\n")

    X = df[MODEL_FEATURES].copy()
    y = compute_target(df)

    # Handle any NaN in features (shouldn't happen, but be safe)
    X = X.fillna(0)

    print("🤖 Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring="r2")
    print(f"   CV R² scores: {np.round(scores, 3)}")
    print(f"   Mean R²:      {scores.mean():.3f} ± {scores.std():.3f}\n")

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(MODEL_FEATURES, FEATURE_LIST_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")
    print(f"✅ Feature list saved → {FEATURE_LIST_PATH}")

    return model, df


def score_all_athletes(model, df: pd.DataFrame):
    """Predict scores for every athlete and write to performance_scores table."""
    X = df[MODEL_FEATURES].fillna(0)
    predictions = model.predict(X)

    print("\n📝 Writing scores to performance_scores table...")
    with engine.begin() as conn:
        for i, row in df.iterrows():
            score = float(np.round(predictions[i], 2))
            conn.execute(
                text("""
                    INSERT INTO performance_scores
                        (athlete_id, feature_id, analysis_id, performance_score, model_version)
                    VALUES (:aid, :fid, :anid, :score, :ver)
                    ON DUPLICATE KEY UPDATE
                        performance_score = :score,
                        model_version = :ver,
                        scored_at = CURRENT_TIMESTAMP
                """),
                {
                    "aid": int(row["athlete_id"]),
                    "fid": int(row["feature_id"]),
                    "anid": int(row["analysis_id"]),
                    "score": score,
                    "ver": "rf_v1",
                },
            )

    print(f"✅ Wrote {len(df)} scores to DB.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-all", action="store_true", help="Write scores to DB")
    args = parser.parse_args()

    model, df = train()

    if args.score_all:
        score_all_athletes(model, df)