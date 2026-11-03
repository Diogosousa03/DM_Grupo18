import pandas as pd
import numpy as np


def open_ended_mask(df):
    """
    Identifies the members without a defined end date
    (membership_end is '1970-01-01' or missing), i.e. open/active contracts.

    Parameters:
        df -> DataFrame with the 'membership_end' column

    Returns:
        A boolean Series (True = no end date / open contract).
    """
    return (df['membership_end'].astype(str) == '1970-01-01') | df['membership_end'].isna()


def get_extraction_date(df):
    """
    Estimates the date the dataset was extracted: the most recent badge swipe in the system.

    Parameters:
        df -> DataFrame with the 'last_badge_swipe' column

    Returns:
        The extraction date (pandas Timestamp).
    """
    return pd.to_datetime(df['last_badge_swipe'], errors='coerce').max()


def compute_tenure(df):
    """
    Adds the tenure (time as a member) to df, in days and in months.
    Members without an end date (open contracts) use the extraction date as the end date.
    Negative tenures (end date before the start) become NaN.

    New columns:
        m_start        -> membership_start as a date
        m_end_cleaned  -> membership_end as a date (extraction date for open contracts)
        tenure_days    -> m_end_cleaned - m_start, in days
        tenure_months  -> tenure_days / 30.4375 (1 average month), rounded to 2 decimals

    Parameters:
        df -> DataFrame to change (e.g. the temporary copy df_eda)

    Returns:
        The df with the new columns.
    """
    df['m_start'] = pd.to_datetime(df['membership_start'], errors='coerce')
    df['m_end_cleaned'] = pd.to_datetime(df['membership_end'], errors='coerce')

    df.loc[open_ended_mask(df), 'm_end_cleaned'] = get_extraction_date(df)

    df['tenure_days'] = (df['m_end_cleaned'] - df['m_start']).dt.days
    df.loc[df['tenure_days'] < 0, 'tenure_days'] = np.nan

    df['tenure_months'] = (df['tenure_days'] / 30.4375).round(2)

    return df


def safe_ratio(num, den):
    """
    Divides num by den, handling zero denominators and missing values.

    Rules:
        num or den missing (NaN)  -> NaN
        den == 0 and num == 0     -> 0.0 (e.g. no quota and no visits = 0%)
        den == 0 and num > 0      -> NaN (division by zero is undefined)
        den > 0                   -> num / den

    Parameters:
        num -> numerator Series (e.g. df_eda['actual_visits_last_period'])
        den -> denominator Series (e.g. df_eda['entitled_visits_last_period'])

    Returns:
        A Series with the result of the division
        (e.g. df_eda['utilization_rate'] = safe_ratio(actual, entitled)).
    """
    conditions = [
        num.isna() | den.isna(),
        (den == 0) & (num == 0),
        (den == 0) & (num > 0),
        den > 0
    ]
    choices = [np.nan, 0.0, np.nan, num / den]

    return pd.Series(np.select(conditions, choices), index=num.index)
