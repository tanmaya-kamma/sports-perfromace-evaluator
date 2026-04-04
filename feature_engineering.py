import pandas as pd
import sqlite3

def create_feature_map(df):
    """
    Takes a dataframe and returns it with engineered features
    """

    # ---- Rename columns ----
    df = df.rename(columns={
        'Height (cm)': 'height',
        'Weight (kg)': 'weight',
        'Age': 'age'
    })

    # ---- Convert height to meters ----
    df['height'] = df['height'] / 100

    # ---- Add missing columns ----
    if 'leg_length' not in df:
        df['leg_length'] = df['height'] * 0.45

    if 'experience_years' not in df:
        df['experience_years'] = 5

    if 'personal_best' not in df:
        df['personal_best'] = 10

    if 'world_record' not in df:
        df['world_record'] = 9.58

    if 'weekly_sessions' not in df:
        df['weekly_sessions'] = 4

    if 'resting_heart_rate' not in df:
        df['resting_heart_rate'] = 60

    if 'injury_history' not in df:
        df['injury_history'] = 0

    # ---- Handle zeros ----
    df[['height', 'age', 'world_record', 'resting_heart_rate']] = \
        df[['height', 'age', 'world_record', 'resting_heart_rate']].replace(0, pd.NA)

    df['injury_history'] = df['injury_history'].fillna(0)

    # ---- Features ----
    df['BMI'] = df['weight'] / (df['height'] ** 2)
    df['leg_height_ratio'] = df['leg_length'] / df['height']
    df['exp_age_ratio'] = df['experience_years'] / df['age']
    df['performance_index'] = df['personal_best'] / df['world_record']
    df['training_intensity'] = df['experience_years'] * df['weekly_sessions']
    df['heart_rate_score'] = 1 / df['resting_heart_rate']
    df['injury_flag'] = df['injury_history'].astype(int)

    # ---- Fill missing ----
    df.fillna(df.median(numeric_only=True), inplace=True)

    # ---- Feature map ----
    feature_columns = [
        'BMI',
        'leg_height_ratio',
        'exp_age_ratio',
        'performance_index',
        'training_intensity',
        'heart_rate_score',
        'injury_flag'
    ]

    feature_map = df[feature_columns]

    return df, feature_map


# 🔥 MAIN BLOCK
if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("athlete_dataset_50.csv")

    print("Columns:", df.columns)

    # Generate features
    df, feature_map = create_feature_map(df)

    # ---- Add Primary Key ----
    df.reset_index(drop=True, inplace=True)
    df['athlete_id'] = df.index + 1

    feature_map['athlete_id'] = df['athlete_id']

    cols = ['athlete_id'] + [col for col in feature_map.columns if col != 'athlete_id']
    feature_map = feature_map[cols]

    # ---- Preview ----
    print("✅ Feature Map Preview:")
    print(feature_map.head())

    # ---- Save CSV ----
    feature_map.to_csv("feature_map.csv", index=False)
    print("✅ Feature map saved as feature_map.csv")

    # ---- Save to SQLite ----
    conn = sqlite3.connect("athlete.db")
    feature_map.to_sql("feature_map", conn, if_exists="replace", index=False)
    conn.close()

    print("✅ Data saved to SQLite database (athlete.db)")