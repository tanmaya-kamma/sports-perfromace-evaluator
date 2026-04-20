"""
Athlete Performance Intelligence API.

Endpoints:
  GET  /athletes                              → Dashboard list
  GET  /athlete/{id}/profile                  → Full athlete profile + engineered features
  GET  /athlete/{id}/video-metrics            → Biomechanical analysis from video
  GET  /athlete/{id}/score                    → Performance score
  POST /athlete/{id}/explain                  → SHAP explanation
  POST /onboarding/profile                    → Register new athlete
  POST /onboarding/upload-video/{id}          → Upload + analyze video
  POST /athlete/{id}/generate-score           → Run model and save score for one athlete
"""

import os
import shutil

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, text

from config import DB_URL, MODEL_FEATURES, MODEL_PATH, VIDEO_DIR
from feature_engineering import compute_age, engineer_features

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="Athlete Performance Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(VIDEO_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

engine = create_engine(DB_URL)

# ─── ML model (loaded once at startup) ──────────────────────────────────────
model = None
explainer = None


def load_ml():
    global model, explainer
    try:
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
        print("✅ ML model + SHAP explainer loaded.")
    except FileNotFoundError:
        print("⚠️  Model file not found — run fusion_trainer.py first.")
    except Exception as e:
        print(f"⚠️  Model load error: {e}")


load_ml()


# ═════════════════════════════════════════════════════════════════════════════
# 1. DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/athletes")
async def get_dashboard():
    """Returns list of all athletes with their latest score (if any)."""
    query = text("""
        SELECT
            a.athlete_id,
            a.athlete_code,
            a.full_name,
            a.biological_gender,
            TIMESTAMPDIFF(YEAR, a.date_of_birth, CURDATE()) AS age,
            ps.performance_score,
            ps.scored_at,
            CASE WHEN vu.video_id IS NOT NULL THEN 1 ELSE 0 END AS has_video
        FROM athletes a
        LEFT JOIN (
            SELECT athlete_id, performance_score, scored_at
            FROM performance_scores
            WHERE score_id IN (
                SELECT MAX(score_id) FROM performance_scores GROUP BY athlete_id
            )
        ) ps ON a.athlete_id = ps.athlete_id
        LEFT JOIN (
            SELECT athlete_id, MIN(video_id) AS video_id
            FROM video_uploads WHERE status = 'completed'
            GROUP BY athlete_id
        ) vu ON a.athlete_id = vu.athlete_id
        ORDER BY a.athlete_id
    """)
    df = pd.read_sql(query, con=engine)
    df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
    return df.to_dict(orient="records")


# ═════════════════════════════════════════════════════════════════════════════
# 2. ATHLETE PROFILE
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/athlete/{athlete_id}/profile")
async def get_profile(athlete_id: int):
    """Returns bio info, latest training profile, and engineered features."""
    query = text("""
        SELECT
            a.athlete_id, a.athlete_code, a.full_name,
            a.date_of_birth, a.biological_gender,
            a.height_cm, a.weight_kg, a.leg_length_cm,
            tp.profile_id, tp.years_of_training, tp.pb_100m_s, tp.pb_400m_s,
            tp.pb_5k_min, tp.resting_heart_rate, tp.vo2_max,
            tp.injury_history, tp.notes,
            ef.feature_id, ef.age_at_computation, ef.bmi, ef.leg_height_ratio,
            ef.exp_age_ratio, ef.performance_index_100m,
            ef.heart_rate_score, ef.vo2_max_normalized
        FROM athletes a
        LEFT JOIN training_profiles tp ON a.athlete_id = tp.athlete_id
        LEFT JOIN engineered_features ef ON tp.profile_id = ef.profile_id
        WHERE a.athlete_id = :aid
        ORDER BY tp.recorded_at DESC
        LIMIT 1
    """)
    df = pd.read_sql(query, con=engine, params={"aid": athlete_id})

    if df.empty:
        raise HTTPException(status_code=404, detail="Athlete not found.")

    row = df.iloc[0]

    # Check for video
    video_filename = f"athlete_{athlete_id:02d}.mp4"
    video_available = os.path.exists(os.path.join(VIDEO_DIR, video_filename))

    return {
        "bio": {
            "athlete_id": int(row["athlete_id"]),
            "athlete_code": row["athlete_code"],
            "full_name": row["full_name"],
            "date_of_birth": str(row["date_of_birth"]),
            "gender": row["biological_gender"],
            "height_cm": float(row["height_cm"]),
            "weight_kg": float(row["weight_kg"]),
            "leg_length_cm": float(row["leg_length_cm"]),
        },
        "training": {
            "years_of_training": _safe_int(row.get("years_of_training")),
            "pb_100m_s": _safe_float(row.get("pb_100m_s")),
            "pb_400m_s": _safe_float(row.get("pb_400m_s")),
            "pb_5k_min": _safe_float(row.get("pb_5k_min")),
            "resting_heart_rate": _safe_int(row.get("resting_heart_rate")),
            "vo2_max": _safe_float(row.get("vo2_max")),
            "injury_history": bool(row.get("injury_history", 0)),
            "notes": row.get("notes"),
        },
        "engineered_features": {
            "bmi": _safe_float(row.get("bmi")),
            "leg_height_ratio": _safe_float(row.get("leg_height_ratio")),
            "exp_age_ratio": _safe_float(row.get("exp_age_ratio")),
            "performance_index_100m": _safe_float(row.get("performance_index_100m")),
            "heart_rate_score": _safe_float(row.get("heart_rate_score")),
            "vo2_max_normalized": _safe_float(row.get("vo2_max_normalized")),
        },
        "video": {
            "file_name": video_filename,
            "url": f"/videos/{video_filename}",
            "available": video_available,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. VIDEO METRICS
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/athlete/{athlete_id}/video-metrics")
async def get_video_metrics(athlete_id: int):
    """Returns biomechanical features from video analysis."""
    query = text("""
        SELECT va.*, vu.file_name, vu.status
        FROM video_analysis va
        JOIN video_uploads vu ON va.video_id = vu.video_id
        WHERE va.athlete_id = :aid
        ORDER BY va.analyzed_at DESC
        LIMIT 1
    """)
    df = pd.read_sql(query, con=engine, params={"aid": athlete_id})

    if df.empty:
        raise HTTPException(status_code=404, detail="No video analysis found for this athlete.")

    row = df.iloc[0]
    return {
        "video_file": row["file_name"],
        "status": row["status"],
        "metrics": {
            "max_left_knee_flexion": _safe_float(row["max_left_knee_flexion"]),
            "max_right_knee_flexion": _safe_float(row["max_right_knee_flexion"]),
            "max_hip_extension": _safe_float(row["max_hip_extension"]),
            "max_ankle_dorsiflexion": _safe_float(row["max_ankle_dorsiflexion"]),
            "avg_trunk_lean": _safe_float(row["avg_trunk_lean"]),
            "symmetry_index": _safe_float(row["symmetry_index"]),
            "stride_variance": _safe_float(row["stride_variance"]),
            "total_left_steps": _safe_int(row["total_left_steps"]),
            "total_right_steps": _safe_int(row["total_right_steps"]),
            "cadence_spm": _safe_float(row["cadence_spm"]),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. PERFORMANCE SCORE
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/athlete/{athlete_id}/score")
async def get_score(athlete_id: int):
    """Returns the latest performance score for an athlete."""
    query = text("""
        SELECT ps.score_id, ps.performance_score, ps.model_version, ps.scored_at
        FROM performance_scores ps
        WHERE ps.athlete_id = :aid
        ORDER BY ps.scored_at DESC
        LIMIT 1
    """)
    df = pd.read_sql(query, con=engine, params={"aid": athlete_id})

    if df.empty:
        raise HTTPException(status_code=404, detail="No score found. Generate one first via POST /athlete/{id}/generate-score")

    row = df.iloc[0]
    return {
        "athlete_id": athlete_id,
        "performance_score": float(row["performance_score"]),
        "model_version": row["model_version"],
        "scored_at": str(row["scored_at"]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. GENERATE SCORE (on demand for a single athlete)
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/athlete/{athlete_id}/generate-score")
async def generate_score(athlete_id: int):
    """Run the RF model for a single athlete and save the score."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train first.")

    feature_row, feature_id, analysis_id = _get_feature_vector(athlete_id)
    prediction = float(np.clip(model.predict(feature_row)[0], 0, 100))
    score = round(prediction, 2)

    # Save to DB
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO performance_scores
                    (athlete_id, feature_id, analysis_id, performance_score, model_version)
                VALUES (:aid, :fid, :anid, :score, 'rf_v1')
            """),
            {"aid": athlete_id, "fid": feature_id, "anid": analysis_id, "score": score},
        )

    return {"athlete_id": athlete_id, "performance_score": score, "status": "saved"}


# ═════════════════════════════════════════════════════════════════════════════
# 6. SHAP EXPLANATION
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/athlete/{athlete_id}/explain")
async def explain_performance(athlete_id: int):
    """Generate SHAP explanation and save to DB."""
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    feature_row, feature_id, analysis_id = _get_feature_vector(athlete_id)

    # Predict
    prediction = float(np.clip(model.predict(feature_row)[0], 0, 100))
    score = round(prediction, 2)

    # SHAP values
    shap_vals = explainer.shap_values(feature_row)
    impacts = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]

    # Save score
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO performance_scores
                    (athlete_id, feature_id, analysis_id, performance_score, model_version)
                VALUES (:aid, :fid, :anid, :score, 'rf_v1')
            """),
            {"aid": athlete_id, "fid": feature_id, "anid": analysis_id, "score": score},
        )
        score_id = result.lastrowid

        # Save SHAP explanations
        for feat_name, feat_val, shap_val in zip(
            MODEL_FEATURES, feature_row.values[0], impacts
        ):
            conn.execute(
                text("""
                    INSERT INTO shap_explanations
                        (score_id, feature_name, feature_value, shap_value, influence_direction)
                    VALUES (:sid, :fname, :fval, :sval, :dir)
                """),
                {
                    "sid": score_id,
                    "fname": feat_name,
                    "fval": float(feat_val) if not np.isnan(feat_val) else None,
                    "sval": float(round(shap_val, 6)),
                    "dir": "positive" if shap_val > 0 else "negative",
                },
            )

    # Build response
    feature_impacts = sorted(
        zip(MODEL_FEATURES, feature_row.values[0], impacts),
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    explanation = []
    for feat_name, feat_val, shap_val in feature_impacts[:5]:
        direction = "Positive" if shap_val > 0 else "Negative"
        pretty_name = feat_name.replace("_", " ").title()
        explanation.append({
            "factor": pretty_name,
            "feature_value": round(float(feat_val), 4),
            "shap_impact": round(float(shap_val), 4),
            "direction": direction,
            "summary": f"{direction} impact ({abs(round(float(shap_val), 2))} pts) from {pretty_name}.",
        })

    return {
        "athlete_id": athlete_id,
        "performance_score": score,
        "top_factors": explanation,
        "score_id": score_id,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. ONBOARDING — CREATE PROFILE
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/onboarding/profile")
async def create_athlete_profile(
    athlete_code: str = Form(...),
    full_name: str = Form(...),
    date_of_birth: str = Form(..., description="YYYY-MM-DD"),
    gender: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    leg_length_cm: float = Form(...),
    years_of_training: int = Form(...),
    pb_100m_s: float = Form(None),
    pb_400m_s: float = Form(None),
    pb_5k_min: float = Form(None),
    resting_heart_rate: int = Form(None),
    vo2_max: float = Form(None),
    injury_history: int = Form(0),
    notes: str = Form(None),
):
    """Register a new athlete: bio → training profile → engineered features."""

    age = compute_age(date_of_birth)

    with engine.begin() as conn:
        # 1. athletes
        result = conn.execute(
            text("""
                INSERT INTO athletes
                    (athlete_code, full_name, date_of_birth, biological_gender,
                     height_cm, weight_kg, leg_length_cm)
                VALUES (:code, :name, :dob, :gender, :h, :w, :leg)
            """),
            {
                "code": athlete_code,
                "name": full_name,
                "dob": date_of_birth,
                "gender": gender,
                "h": height_cm,
                "w": weight_kg,
                "leg": leg_length_cm,
            },
        )
        athlete_id = result.lastrowid

        # 2. training_profiles
        result = conn.execute(
            text("""
                INSERT INTO training_profiles
                    (athlete_id, years_of_training, pb_100m_s, pb_400m_s, pb_5k_min,
                     resting_heart_rate, vo2_max, injury_history, notes)
                VALUES (:aid, :yt, :pb1, :pb4, :pb5, :rhr, :vo2, :inj, :notes)
            """),
            {
                "aid": athlete_id,
                "yt": years_of_training,
                "pb1": pb_100m_s,
                "pb4": pb_400m_s,
                "pb5": pb_5k_min,
                "rhr": resting_heart_rate,
                "vo2": vo2_max,
                "inj": injury_history,
                "notes": notes,
            },
        )
        profile_id = result.lastrowid

        # 3. engineered_features
        feats = engineer_features(
            height_cm=height_cm,
            weight_kg=weight_kg,
            leg_length_cm=leg_length_cm,
            age=age,
            years_of_training=years_of_training,
            pb_100m_s=pb_100m_s,
            resting_heart_rate=resting_heart_rate,
            vo2_max=vo2_max,
        )
        conn.execute(
            text("""
                INSERT INTO engineered_features
                    (athlete_id, profile_id, age_at_computation,
                     bmi, leg_height_ratio, exp_age_ratio,
                     performance_index_100m, heart_rate_score, vo2_max_normalized)
                VALUES (:aid, :pid, :age, :bmi, :lhr, :ear, :pi, :hrs, :vo2n)
            """),
            {
                "aid": athlete_id,
                "pid": profile_id,
                "age": age,
                "bmi": feats["bmi"],
                "lhr": feats["leg_height_ratio"],
                "ear": feats["exp_age_ratio"],
                "pi": feats["performance_index_100m"],
                "hrs": feats["heart_rate_score"],
                "vo2n": feats["vo2_max_normalized"],
            },
        )

    return {
        "status": "success",
        "athlete_id": athlete_id,
        "message": "Profile created. Upload video next via POST /onboarding/upload-video/{id}",
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. ONBOARDING — UPLOAD & ANALYZE VIDEO
# ═════════════════════════════════════════════════════════════════════════════
@app.post("/onboarding/upload-video/{athlete_id}")
async def upload_and_analyze_video(athlete_id: int, file: UploadFile = File(...)):
    """Save video file, run MediaPipe, store results."""
    from performance_video.model_processor import process_video

    # Verify athlete exists
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM athletes WHERE athlete_id = :aid"),
            {"aid": athlete_id},
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Athlete not found.")

    # Save file
    video_filename = f"athlete_{athlete_id:02d}.mp4"
    file_path = os.path.join(VIDEO_DIR, video_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with engine.begin() as conn:
        # Track upload
        result = conn.execute(
            text("""
                INSERT INTO video_uploads (athlete_id, file_name, file_path, status)
                VALUES (:aid, :fn, :fp, 'processing')
            """),
            {"aid": athlete_id, "fn": video_filename, "fp": file_path},
        )
        video_id = result.lastrowid

        # Run MediaPipe
        analysis = process_video(file_path)

        if analysis:
            conn.execute(
                text("""
                    INSERT INTO video_analysis
                        (video_id, athlete_id, max_left_knee_flexion, max_right_knee_flexion,
                         max_hip_extension, max_ankle_dorsiflexion, avg_trunk_lean,
                         symmetry_index, stride_variance,
                         total_left_steps, total_right_steps, cadence_spm)
                    VALUES (:vid, :aid, :lk, :rk, :hip, :ank, :trunk,
                            :sym, :sv, :ls, :rs, :cad)
                """),
                {
                    "vid": video_id,
                    "aid": athlete_id,
                    "lk": analysis["max_left_knee_flexion"],
                    "rk": analysis["max_right_knee_flexion"],
                    "hip": analysis["max_hip_extension"],
                    "ank": analysis["max_ankle_dorsiflexion"],
                    "trunk": analysis["avg_trunk_lean"],
                    "sym": analysis["symmetry_index"],
                    "sv": analysis["stride_variance"],
                    "ls": analysis["total_left_steps"],
                    "rs": analysis["total_right_steps"],
                    "cad": analysis["cadence_spm"],
                },
            )
            conn.execute(
                text("UPDATE video_uploads SET status = 'completed' WHERE video_id = :vid"),
                {"vid": video_id},
            )
            return {"status": "success", "video_id": video_id, "analysis": analysis}
        else:
            conn.execute(
                text("UPDATE video_uploads SET status = 'failed' WHERE video_id = :vid"),
                {"vid": video_id},
            )
            raise HTTPException(
                status_code=422,
                detail="Video uploaded but MediaPipe failed to detect any pose landmarks.",
            )


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _get_feature_vector(athlete_id: int):
    """
    Fetch the full feature vector for a single athlete.
    Returns (DataFrame with 1 row of MODEL_FEATURES, feature_id, analysis_id).
    """
    query = text("""
        SELECT
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
            va.cadence_spm
        FROM engineered_features ef
        JOIN video_analysis va ON ef.athlete_id = va.athlete_id
        WHERE ef.athlete_id = :aid
        ORDER BY ef.computed_at DESC, va.analyzed_at DESC
        LIMIT 1
    """)
    df = pd.read_sql(query, con=engine, params={"aid": athlete_id})

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Incomplete data — need both engineered features and video analysis.",
        )

    feature_id = int(df.iloc[0]["feature_id"])
    analysis_id = int(df.iloc[0]["analysis_id"])
    feature_row = df[MODEL_FEATURES].fillna(0)

    return feature_row, feature_id, analysis_id


def _safe_float(val, decimals=4):
    """Safely convert to rounded float, return None for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)


def _safe_int(val):
    """Safely convert to int, return None for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return int(val)


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)