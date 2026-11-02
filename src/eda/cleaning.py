import pandas as pd
import numpy as np


def temp_copy(df, var1, var2):
    """
    Creates a temporary copy of the dataset for the EDA (the original df is not changed)
    and adds a numeric version of a text column.
    Negative values in the new column become NaN.

    Parameters:
        df   -> original DataFrame
        var1 -> name of the new numeric column (e.g. 'lifetime_value_clean')
        var2 -> name of the text column to convert (e.g. 'lifetime_value')

    Returns:
        The copy with the new column.
    """
    df_eda = df.copy()

    # Temporary in-memory handling:
    df_eda[var1] = pd.to_numeric(
    df_eda[var2].astype(str).str.replace(r'[^\d.-]', '', regex=True), 
    errors='coerce'
    )

    # A negative value is impossible, so it is treated as missing (NaN)
    df_eda.loc[df_eda[var1] < 0, var1] = np.nan
    return df_eda

#-----------------------------------------------------------------------------------------#

def clean_invalid_values(df, cols):
    """
    Replaces impossible values with NaN (only in the df that is passed, not in the original dataset).
    Ages < 16 or > 100 also become NaN (if the 'age' column exists).
    Columns that do not exist in df are ignored.

    Parameters:
        df   -> DataFrame to clean (e.g. the temporary copy df_eda)
        cols -> list of columns that cannot have negative values (< 0 -> NaN)

    Returns:
        The cleaned df.
    """
    for col in cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    if 'age' in df.columns:
        df.loc[(df['age'] < 16) | (df['age'] > 100), 'age'] = np.nan

    return df
