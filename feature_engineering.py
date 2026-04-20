"""
Feature engineering module.

Pure computation — takes raw athlete + training data and returns derived features.
Called by:
  - seed_data.py   (bulk initial load)
  - app.py         (onboarding endpoint)
  - fusion_trainer.py (training pipeline)

NO database or API logic lives here.
"""
from typing import Optional
from datetime import date, datetime
from config import WR_100M


def compute_age(date_of_birth) -> int:
    """Calculate current age from date of birth."""
    if isinstance(date_of_birth, str):
        # Handle multiple date formats
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y"):
            try:
                date_of_birth = datetime.strptime(date_of_birth, fmt).date()
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse date: {date_of_birth}")

    today = date.today()
    return today.year - date_of_birth.year - (
        (today.month, today.day) < (date_of_birth.month, date_of_birth.day)
    )


WR_100M = 9.58  # make sure this is defined

def engineer_features(
    height_cm: float,
    weight_kg: float,
    leg_length_cm: float,
    age: int,
    years_of_training: int,
    pb_100m_s: Optional[float] = None,
    resting_heart_rate: Optional[int] = None,
    vo2_max: Optional[float] = None,
) -> dict:
    """
    Compute all derived features from raw athlete + training data.

    Returns a dict with keys matching the engineered_features table columns.
    """

    WR_100M = 9.58  # Usain Bolt world record

    height_m = height_cm / 100.0 if height_cm is not None else None
    leg_m = leg_length_cm / 100.0 if leg_length_cm is not None else None

    # Safe BMI
    bmi = (
        round(weight_kg / (height_m ** 2), 2)
        if height_m and weight_kg is not None
        else None
    )

    # Safe ratios
    leg_height_ratio = (
        round(leg_m / height_m, 4)
        if height_m and leg_m is not None
        else None
    )

    exp_age_ratio = (
        round(years_of_training / age, 4)
        if age and years_of_training is not None
        else None
    )

    # Sprint performance
    performance_index_100m = (
        round(pb_100m_s / WR_100M, 4)
        if pb_100m_s is not None
        else None
    )

    # Heart rate score (avoid division by zero)
    heart_rate_score = (
        round(1.0 / resting_heart_rate, 6)
        if resting_heart_rate not in (None, 0)
        else None
    )

    # VO2 max normalization
    vo2_max_normalized = (
        round(vo2_max / 80.0, 4)
        if vo2_max is not None
        else None
    )

    return {
        "bmi": bmi,
        "leg_height_ratio": leg_height_ratio,
        "exp_age_ratio": exp_age_ratio,
        "performance_index_100m": performance_index_100m,
        "heart_rate_score": heart_rate_score,
        "vo2_max_normalized": vo2_max_normalized,
    }