import pandas as pd



#
def temp_copy(df, var1, var2):
    df_eda = df.copy()

    # Temporary in-memory handling
    df_eda[var1] = pd.to_numeric(
    df_eda[var2].astype(str).str.replace(r'[^\d.-]', '', regex=True), 
    errors='coerce'
    )
    return df_eda