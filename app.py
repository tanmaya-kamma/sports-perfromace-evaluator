import pandas as pd
import joblib
import shap
import mysql.connector
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
from fastapi import Form, File, UploadFile
import shutil
app = FastAPI(title="Athlete Performance Intelligence API")

# --- CORS Configuration (Optional: Allows frontend to connect) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- NEW: Static File Mounting ---
# This allows the frontend to stream videos from your local folder
# Access via: http://localhost:8000/videos/athlete_01.mp4
VIDEO_DIR = "performance_video/videos"
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")

# --- Database Configuration ---
DB_URL = "mysql+mysqlconnector://root:tanmayakamma@localhost:3306/athlete_db"
engine = create_engine(DB_URL)

# --- Global ML Components ---
# This list must match the columns used during model training exactly
FEATURES_LIST = [
    'BMI', 'leg_height_ratio', 'exp_age_ratio', 'performance_index', 
    'heart_rate_score', 'max_left_knee', 'max_right_knee', 
    'max_hip', 'max_ankle', 'avg_trunk_lean', 'symmetry_index', 'stride_variance'
]

model = None
explainer = None

def load_ml_components():
    global model, explainer
    try:
        model = joblib.load("athlete_rf_model.pkl")
        explainer = shap.TreeExplainer(model)
        print("✅ ML Model and SHAP Explainer loaded.")
    except Exception as e:
        print(f"⚠️ Model files missing or incompatible: {e}")

load_ml_components()

# --- 1. DASHBOARD VIEW ---
@app.get("/athletes")
async def get_dashboard():
    """Returns the main list for the Coach's dashboard."""
    query = """
        SELECT a.athlete_id, a.full_name, a.athlete_code, s.performance_score 
        FROM athletes a
        LEFT JOIN athlete_scores s ON a.athlete_id = s.athlete_id
    """
    df = pd.read_sql(query, con=engine)
    df['performance_score'] = df['performance_score'].fillna("N/A")
    return df.to_dict(orient="records")

# --- 2. ATHLETE DETAILS VIEW ---
@app.get("/athlete/{athlete_id}/metadata")
async def get_metadata(athlete_id: int):
    """Returns profile, stats, and the playable video URL."""
    query = "SELECT * FROM athletes WHERE athlete_id = %s"
    df = pd.read_sql(query, con=engine, params=(athlete_id,))
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Athlete not found.")
        
    row = df.iloc[0].to_dict()
    
    # Video Logic
    video_filename = f"athlete_{athlete_id:02d}.mp4"
    video_local_path = os.path.join(VIDEO_DIR, video_filename)
    
    return {
        "profile": {
            "name": row.get("full_name"),
            "code": row.get("athlete_code"),
            "age": row.get("age"),
            "gender": row.get("biological_gender"),
            "training_years": row.get("years_of_training")
        },
        "engineered_metrics": {
            "BMI": round(row.get("BMI", 0), 2),
            "Leg_Height_Ratio": round(row.get("leg_height_ratio", 0), 2),
            "Exp_Age_Ratio": round(row.get("exp_age_ratio", 0), 2),
            "Performance_Index": round(row.get("performance_index", 0), 2)
        },
        "video_assets": {
            "file_name": video_filename,
            "video_url": f"/videos/{video_filename}", # Accessible via API host
            "is_available": os.path.exists(video_local_path)
        }
    }

# --- 3. VIDEO ANALYSIS VIEW ---
@app.get("/athlete/{athlete_id}/video-metrics")
async def get_video(athlete_id: int):
    """Returns biomechanical data from the video_analysis table."""
    query = "SELECT * FROM video_analysis WHERE athlete_id = %s"
    df = pd.read_sql(query, con=engine, params=(athlete_id,))
    
    if df.empty:
        return {"status": "pending", "message": "No video analysis found."}
        
    data = df.iloc[0].to_dict()
    # Filter out internal IDs and timestamps
    return {k: round(v, 2) if isinstance(v, float) else v 
            for k, v in data.items() if k not in ['analysis_id', 'created_at', 'athlete_id']}

# --- 4. EXPLAINABLE AI VIEW ---
@app.post("/athlete/{athlete_id}/explain-performance")
async def explain_performance(athlete_id: int):
    if model is None or explainer is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # 1. Fetch data
    query = """
        SELECT a.*, v.max_left_knee, v.max_right_knee, v.max_hip, 
               v.avg_trunk_lean, v.symmetry_index, v.stride_variance
        FROM athletes a
        JOIN video_analysis v ON a.athlete_id = v.athlete_id
        WHERE a.athlete_id = %s
    """
    df = pd.read_sql(query, con=engine, params=(athlete_id,))
    
    if df.empty:
        raise HTTPException(status_code=400, detail="Incomplete data.")

    # 2. Synchronized Feature Engineering
    # IMPORTANT: Ensure this list matches your fusion_trainer.py EXACTLY.
    # Removed 'max_ankle' because your model wasn't trained with it.
    ACTIVE_FEATURES = [
        'BMI', 'leg_height_ratio', 'exp_age_ratio', 'performance_index', 
        'heart_rate_score', 'max_left_knee', 'max_right_knee', 
        'max_hip', 'avg_trunk_lean', 'symmetry_index', 'stride_variance'
    ]

    try:
        if 'BMI' not in df.columns:
            df['BMI'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
        if 'leg_height_ratio' not in df.columns:
            df['leg_height_ratio'] = (df['leg_length_cm'] / 100) / (df['height_cm'] / 100)
        if 'exp_age_ratio' not in df.columns:
            df['exp_age_ratio'] = df['years_of_training'] / df['age']
        if 'performance_index' not in df.columns:
            df['performance_index'] = df['pb_100m_s'] / 9.58 
        if 'heart_rate_score' not in df.columns:
            df['heart_rate_score'] = 1 / df['resting_heart_rate']
            
        X_athlete = df[ACTIVE_FEATURES]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: {str(e)}")

    # 3. Generate Prediction
    prediction = model.predict(X_athlete)[0]
    
    # 4. Optimized SHAP calculation
    # To speed up, we compute SHAP only for this single row.
    # TreeExplainer is already fast, but ensure it's initialized correctly.
    shap_vals = explainer.shap_values(X_athlete)
    
    # Handle SHAP output format
    impacts = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]

    # 5. Format Explanation
    feature_impacts = dict(zip(ACTIVE_FEATURES, impacts))
    sorted_factors = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    reasoning = []
    for feat, val in sorted_factors[:4]:
        inf = "Positive" if val > 0 else "Negative"
        reasoning.append({
            "factor": feat.replace('_', ' ').title(),
            "impact": float(round(val, 2)),
            "influence": inf,
            "display_text": f"{inf} impact ({abs(round(val, 2))} pts) from {feat.replace('_', ' ')}."
        })

    return {
        "status": "success",
        "athlete_id": int(athlete_id),
        "performance_score": float(round(prediction, 2)),
        "explanation": reasoning
    }

# --- 5. ONBOARDING: SUBMIT METADATA ---
@app.post("/onboarding/profile")
async def create_athlete_profile(
    athlete_id: int = Form(...),
    athlete_code: str = Form(...),
    full_name: str = Form(...),
    age: int = Form(...),
    dob: str = Form(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
    leg_length: float = Form(...),
    training_years: int = Form(...),
    pb_100m: float = Form(...),
    rest_hr: int = Form(...)
):
    # Calculate Engineered Features on the fly
    bmi = round(weight / ((height / 100) ** 2), 2)
    leg_ratio = round((leg_length / 100) / (height / 100), 2)
    exp_ratio = round(training_years / age, 2)
    perf_idx = round(pb_100m / 9.58, 2)
    hr_score = round(1 / rest_hr, 4)

    query = """
        INSERT INTO athletes (
            athlete_id, athlete_code, full_name, age, date_of_birth, 
            biological_gender, height_cm, weight_kg, leg_length_cm, 
            years_of_training, pb_100m_s, resting_heart_rate, 
            BMI, leg_height_ratio, exp_age_ratio, performance_index, heart_rate_score,
            has_video
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0)
    """
    
    

    with engine.begin() as conn:
        conn.execute(
        text(query),
        {
            "athlete_id": athlete_id,
            "athlete_code": athlete_code,
            "full_name": full_name,
            "age": age,
            "dob": dob,
            "gender": gender,
            "height": height,
            "weight": weight,
            "leg_length": leg_length,
            "training_years": training_years,
            "pb_100m": pb_100m,
            "rest_hr": rest_hr,
            "bmi": bmi,
            "leg_ratio": leg_ratio,
            "exp_ratio": exp_ratio,
            "perf_idx": perf_idx,
            "hr_score": hr_score
        }
    )

    return {"status": "success", "message": "Profile created. Please upload video next."}

# --- 6. ONBOARDING: UPLOAD & ANALYZE VIDEO ---
@app.post("/onboarding/upload-video/{athlete_id}")
async def upload_and_analyze_video(athlete_id: int, file: UploadFile = File(...)):
    # 1. Save the raw video file
    video_filename = f"athlete_{athlete_id:02d}.mp4"
    file_location = os.path.join(VIDEO_DIR, video_filename)
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Run Mediapipe Analysis immediately
    # We reuse your existing process_video logic from earlier
    from performance_video.model_processor import process_video # Import your existing video logic script
    
    analysis_results = process_video(file_location, athlete_id)

    if analysis_results:
        # 3. Save Analysis to SQL
        query = """
            INSERT INTO video_analysis 
            (athlete_id, max_left_knee, max_right_knee, max_hip, max_ankle, avg_trunk_lean, symmetry_index, stride_variance, total_left_steps)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with engine.begin() as conn:
            conn.execute(text(query), list(analysis_results.values()))
            # Update the 'has_video' flag in athletes table
            conn.execute(
        text("UPDATE athletes SET has_video = 1 WHERE athlete_id = :athlete_id"),
        {"athlete_id": athlete_id}
    )
        conn.commit()
        return {"status": "success", "analysis": analysis_results}
    
    return {"status": "error", "message": "Video uploaded but Mediapipe failed to detect pose."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)