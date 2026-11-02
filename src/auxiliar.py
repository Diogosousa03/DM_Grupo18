import pandas as pd



# Creates a temporary copy of the dataset for the EDA (the original df is not changed)
# and adds a numeric version of a text column.
#   df   -> original DataFrame
#   var1 -> name of the new numeric column (e.g. 'lifetime_value_clean')
#   var2 -> name of the text column to convert (e.g. 'lifetime_value')
# Returns the copy with the new column.
def temp_copy(df, var1, var2):
    # Copy so that changes do not affect the original df
    df_eda = df.copy()

    # Temporary in-memory handling:
    # remove everything that is not a digit, "." or "-" (e.g. "€", "," and spaces),
    # then convert to a number. Values that cannot be converted become NaN.
    df_eda[var1] = pd.to_numeric(
    df_eda[var2].astype(str).str.replace(r'[^\d.-]', '', regex=True), 
    errors='coerce'
    )
    return df_eda