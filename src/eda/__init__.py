"""
Auxiliary functions for the EDA notebook.
Everything can be imported directly from the package, e.g. "from eda import temp_copy".
"""

from .cleaning import temp_copy, clean_invalid_values
from .plans import PLAN_MAP, PLAN_ORDER, PLAN_CODE, clean_plan
from .features import open_ended_mask, get_extraction_date, compute_tenure, safe_ratio
from .plots import set_labels, finish_figure
