import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor

# DB Connection
engine = create_engine("mysql+mysqlconnector://root:tanmayakamma@localhost:3306/athlete_db")

def train_performance_model():
    print("🔗 Fetching data for fusion...")
    df_meta = pd.read_sql("SELECT * FROM feature_map", con=engine)
    df_video = pd.read_sql("SELECT * FROM video_analysis", con=engine)
    
    # Merge on athlete_id
    fused_df = pd.merge(df_meta, df_video, on="athlete_id")
    
    # Define features exactly as they appear in the DB
    features = [
        'BMI', 'leg_height_ratio', 'exp_age_ratio', 'performance_index', 
        'heart_rate_score', 'max_left_knee', 'max_right_knee', 
        'max_hip', 'avg_trunk_lean', 'symmetry_index', 'stride_variance'
    ]
    
    X = fused_df[features]
    
    # Synthetic Target: Speed (50%) + Biomechanical Efficiency (50%)
    y = ((fused_df['performance_index'] * 50) + (1.0 - fused_df['symmetry_index']) * 50)
    
    print("🤖 Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model and the feature list for the API to use
    joblib.dump(model, "athlete_rf_model.pkl")
    joblib.dump(features, "feature_list.pkl")
    print("✅ Model saved as athlete_rf_model.pkl")

if __name__ == "__main__":
    train_performance_model()