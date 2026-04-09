from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np

app = FastAPI(title="Athlete Performance API")

# Load the dataset once when the server starts
DATA_PATH = "athlete_metadata_50.csv"

def get_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return None

def engineer_features(row):
    """
    Calculates engineered features for a single athlete row.
    Ensures all floats are rounded to 2 decimal points.
    """
    height_m = row['height_cm'] / 100
    leg_len_m = row['leg_length_cm'] / 100

    # Engineering Logic
    bmi = row['weight_kg'] / (height_m ** 2)
    leg_height_ratio = leg_len_m / height_m
    exp_age_ratio = row['years_of_training'] / row['age']
    perf_index = row['pb_100m_s'] / 9.58
    hr_score = 1 / row['resting_heart_rate']

    return {
        "BMI": round(float(bmi), 2),
        "leg_height_ratio": round(float(leg_height_ratio), 2),
        "exp_age_ratio": round(float(exp_age_ratio), 2),
        "performance_index_100m": round(float(perf_index), 2),
        "heart_rate_score": round(float(hr_score), 4) # HR score is usually very small
    }

@app.get("/athlete/{athlete_id}")
async def get_athlete_details(athlete_id: int):
    df = get_data()
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not found.")

    # Find the athlete
    athlete_row = df[df['athlete_id'] == athlete_id]

    if athlete_row.empty:
        raise HTTPException(status_code=404, detail="Athlete not found.")

    # Convert row to dictionary
    raw_data = athlete_row.iloc[0].to_dict()
    
    # Round raw float values to 2 decimal places
    for key, value in raw_data.items():
        if isinstance(value, float):
            raw_data[key] = round(value, 2)

    # Get Engineered Features
    engineered_data = engineer_features(raw_data)

    # Merge and return
    return {**raw_data, "engineered_features": engineered_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)