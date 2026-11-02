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
