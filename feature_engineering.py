import pandas as pd
import sqlite3
import mysql.connector
from sqlalchemy import create_engine


def create_feature_map(df):
    """
    Takes a dataframe and returns engineered features
    """

    # ---- Rename columns ----
    df = df.rename(columns={
        'Height (cm)': 'height',
        'Weight (kg)': 'weight',
        'Age': 'age'
    })

    # ---- Convert height to meters ----
    df['height'] = df['height'] / 100

    # ---- Direct mapping from dataset ----
    df['leg_length'] = df['Leg Length (cm)'] / 100
    df['experience_years'] = df['Years of Training']
    df['personal_best'] = df['100m PB (s)']
    df['weekly_sessions'] = df['Training Intensity']
    df['resting_heart_rate'] = df['Resting Heart Rate']
    df['injury_history'] = df['Injury History (0/1)']

    # ---- Feature Engineering ----
    df['BMI'] = df['weight'] / (df['height'] ** 2)

    df['leg_height_ratio'] = df['leg_length'] / df['height']

    df['exp_age_ratio'] = df['experience_years'] / df['age']

    df['performance_index'] = df['personal_best'] / 9.58

    df['training_intensity'] = df['experience_years'] * df['weekly_sessions']

    df['heart_rate_score'] = 1 / df['resting_heart_rate']

    df['injury_flag'] = df['injury_history'].astype(int)

    # ---- Select features ----
    feature_columns = [
        'BMI',
        'leg_height_ratio',
        'exp_age_ratio',
        'performance_index',
        'training_intensity',
        'heart_rate_score',
        'injury_flag'
    ]

    return df, df[feature_columns]


# 🔥 MAIN BLOCK
if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("athlete_dataset_50.csv")

    print("Columns:", df.columns.tolist())

    # Generate features
    df, feature_map = create_feature_map(df)

    # ---- Add Primary Key ----
    df.reset_index(drop=True, inplace=True)
    df['athlete_id'] = df.index + 1
    feature_map['athlete_id'] = df['athlete_id']

    # ---- Reorder columns ----
    cols = ['athlete_id'] + [c for c in feature_map.columns if c != 'athlete_id']
    feature_map = feature_map[cols]

    print("\nPreview:")
    print(feature_map.head())

    # ---- Save CSV ----
    feature_map.to_csv("feature_map.csv", index=False)
    print("\n✅ CSV saved")

    # ---- Save to SQLite ----
    conn_sqlite = sqlite3.connect("athlete.db")
    feature_map.to_sql("feature_map", conn_sqlite, if_exists="replace", index=False)
    conn_sqlite.close()
    print("✅ Saved to SQLite")

    # ---- Save to MySQL ----
    engine = create_engine("mysql+mysqlconnector://testuser:@localhost/athlete_db")

    feature_map.to_sql("feature_map", engine, if_exists="replace", index=False)

    print("✅ Data saved to MySQL (FINAL)")