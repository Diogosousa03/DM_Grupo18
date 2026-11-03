# Plan names: every spelling found in the data (lowercase) -> standard name
PLAN_MAP = {
    'flex': 'Flex',
    'hot_desk': 'Hot Desk', 'hot desk': 'Hot Desk', 'hotdesk': 'Hot Desk',
    'dedicated_desk': 'Dedicated Desk', 'dedicated desk': 'Dedicated Desk', 'dedicateddesk': 'Dedicated Desk',
    'full_time': 'Full Time', 'fulltime': 'Full Time', 'full time': 'Full Time',
    'private_office': 'Private Office', 'private office': 'Private Office', 'privateoffice': 'Private Office'
}

# Plan hierarchy, from lowest user plan to highest.
PLAN_ORDER = ['Flex', 'Hot Desk', 'Dedicated Desk', 'Full Time', 'Private Office']

# Ordinal code of each plan (Flex = 1 ... Private Office = 5)
PLAN_CODE = {plan: i for i, plan in enumerate(PLAN_ORDER, start=1)}

#-----------------------------------------------------------------------------------------#

def clean_plan(df, col='plan_tier'):
    """
    Standardizes the plan names (e.g. 'HOT_DESK', 'hot desk ' -> 'Hot Desk').
    Values that are not in PLAN_MAP keep their original value.

    Parameters:
        df  -> DataFrame with the plan column
        col -> name of the plan column (default 'plan_tier')

    Returns:
        A Series with the standardized plan names
        (e.g. df_eda['plan_clean'] = clean_plan(df_eda)).
    """
    return df[col].astype(str).str.strip().str.lower().map(PLAN_MAP).fillna(df[col])
