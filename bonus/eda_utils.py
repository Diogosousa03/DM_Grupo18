"""Data loading and cleaning for the ACDC interactive EDA dashboard.

Reuses the cleaning helpers of the EDA notebook (src/eda), so the dashboard
applies the same rules: invalid values and placeholder codes -> NaN,
plan standardization and lifetime_value conversion.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the notebook helpers (src/eda) importable, as done in the notebook
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from eda import PLAN_ORDER, clean_invalid_values, clean_plan, temp_copy  # noqa: E402

NUMERIC_COLS = [
    'age', 'n_colleagues_on_account', 'plan_days_per_week', 'days_since_last_visit',
    'total_visits', 'avg_weekly_visits', 'entitled_visits_last_period',
    'actual_visits_last_period', 'meeting_room_hours', 'event_space_hours',
    'guest_passes_issued', 'n_renewals',
]
BINARY_COLS = [
    'is_team_account', 'uses_meeting_rooms', 'uses_event_space', 'uses_guest_passes',
    'uses_print', 'uses_mail_handling', 'uses_phone_booths', 'uses_day_passes',
    'uses_storage', 'uses_bike_locker',
]
GENDER_MAP = {'female': 'Female', 'f': 'Female', 'male': 'Male', 'm': 'Male',
              'other': 'Other', 'o': 'Other'}
INVALID_STRINGS = ['unknown', 'none', 'null', '?', '-', 'nan', '', 'n/a']
TRUE_VALUES = {'1', '1.0', 'true', 't', 'yes', 'y'}
FALSE_VALUES = {'0', '0.0', 'false', 'f', 'no', 'n'}


def to_binary(series: pd.Series) -> pd.Series:
    """Convert True/False, yes/no, 1/0 style columns to 1/0 (NaN when unrecognised)."""
    key = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[key.isin(TRUE_VALUES)] = 1.0
    out[key.isin(FALSE_VALUES)] = 0.0
    return out


def clean_text(series: pd.Series) -> pd.Series:
    """Trim, lower-case and replace placeholder strings by NaN."""
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.replace(INVALID_STRINGS, np.nan)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw dataset, with display-ready label columns."""
    df = df.drop_duplicates().reset_index(drop=True).copy()

    # Numeric variables: coerce to numbers, then the notebook rules
    # (negatives, placeholder codes such as 999 and impossible ages -> NaN)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    clean_invalid_values(df, NUMERIC_COLS)

    # Lifetime value: strip currency symbols/text, convert, negatives -> NaN
    if 'lifetime_value' in df.columns:
        df = temp_copy(df, 'lifetime_value_clean', 'lifetime_value')

    # Binary flags
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = to_binary(df[col])

    # Display labels (NaN shown as "Unknown" so filters do not silently drop rows)
    if 'plan_tier' in df.columns:
        df['Plan'] = clean_plan(df).fillna('Unknown')
    if 'gender' in df.columns:
        df['Gender'] = clean_text(df['gender']).map(GENDER_MAP).fillna('Other')
    if 'is_team_account' in df.columns:
        df['Account'] = df['is_team_account'].map({1.0: 'Team', 0.0: 'Individual'}).fillna('Unknown')
    for src, label in [('country', 'Country'), ('industry', 'Industry')]:
        if src in df.columns:
            df[label] = clean_text(df[src]).str.title().fillna('Unknown')
    return df


def top_n_labels(series: pd.Series, n: int = 10) -> pd.Series:
    """Keep the n most frequent categories and group the rest as 'Other (grouped)'."""
    top = series.value_counts().nlargest(n).index
    return series.where(series.isin(top), 'Other (grouped)')