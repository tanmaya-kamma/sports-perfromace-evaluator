import pandas as pd
import mysql.connector

def create_feature_map(df):

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

    return df


# 🔥 MAIN
if __name__ == "__main__":

    # ---- Load dataset ----
    df = pd.read_csv("athlete_dataset_50.csv")
    print("Columns:", df.columns)

    # ---- Feature engineering ----
    df = create_feature_map(df)

    # ---- Add Primary Key ----
    df.reset_index(drop=True, inplace=True)
    df['athlete_id'] = df.index + 1

    # ---- Feature map ----
    feature_map = df[[
        'athlete_id',
        'BMI',
        'leg_height_ratio',
        'exp_age_ratio',
        'performance_index',
        'training_intensity',
        'heart_rate_score',
        'injury_flag'
    ]]

    # ---- Preview ----
    print("✅ Feature Map Preview:")
    print(feature_map.head())

    # ---- Save CSV ----
    feature_map.to_csv("feature_map.csv", index=False)
    print("✅ Feature map saved as feature_map.csv")

    # ==============================
    # 🔥 MYSQL CONNECTION
    # ==============================

    conn = mysql.connector.connect(
        host="localhost",
        user="testuser",
        password="",
        database="athlete_db"
    )

    cursor = conn.cursor()

    # ---- Create table ----
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feature_map (
        athlete_id INT PRIMARY KEY,
        BMI FLOAT,
        leg_height_ratio FLOAT,
        exp_age_ratio FLOAT,
        performance_index FLOAT,
        training_intensity FLOAT,
        heart_rate_score FLOAT,
        injury_flag INT
    )
    """)

    # ---- Clear old data (optional but safe) ----
    cursor.execute("DELETE FROM feature_map")

    # ---- Insert data ----
    for _, row in feature_map.iterrows():
        cursor.execute("""
        INSERT INTO feature_map (
            athlete_id, BMI, leg_height_ratio, exp_age_ratio,
            performance_index, training_intensity,
            heart_rate_score, injury_flag
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

    print("✅ Data saved to MySQL database!")