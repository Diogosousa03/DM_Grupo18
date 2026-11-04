"""ACDC - Interactive EDA dashboard.   Run with:  streamlit run app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from eda_utils import PLAN_ORDER, NUMERIC_COLS, clean_dataset, top_n_labels

st.set_page_config(page_title="ACDC - Interactive EDA", layout="wide")

COLOR = '#4682b4'
DATA_CANDIDATES = ['acdc_dataset.csv', 'data/acdc_dataset.csv', '../data/acdc_dataset.csv']


# Data loading
@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(source) -> pd.DataFrame:
    raw = pd.read_csv(source, sep=';')
    return clean_dataset(raw)


def find_data_file():
    here = Path(__file__).parent
    for candidate in DATA_CANDIDATES:
        path = (here / candidate).resolve()
        if path.exists():
            return path
    return None


st.title("ACDC Coworking - Interactive EDA Dashboard")

data_path = find_data_file()
if data_path is not None:
    df = load_data(data_path)
else:
    uploaded = st.file_uploader("Dataset not found automatically. Upload acdc_dataset.csv (';' separated):",
                                type='csv')
    if uploaded is None:
        st.stop()
    df = load_data(uploaded)

num_cols = [c for c in NUMERIC_COLS + ['lifetime_value_clean'] if c in df.columns]
group_options = {label: label for label in ['Plan', 'Gender', 'Account', 'Country', 'Industry']
                 if label in df.columns}

LABELS = {
    'age': 'Age',
    'total_visits': 'Total visits',
    'lifetime_value_clean': 'Lifetime value (EUR)',
    'n_colleagues_on_account': 'Colleagues on account',
    'entitled_visits_last_period': 'Entitled visits (last period)',
    'actual_visits_last_period': 'Actual visits (last period)',
    'avg_weekly_visits': 'Avg weekly visits',
    'meeting_room_hours': 'Meeting room hours',
    'event_space_hours': 'Event space hours',
    'guest_passes_issued': 'Guest passes issued',
    'days_since_last_visit': 'Days since last visit',
    'n_renewals': 'Renewals',
    'plan_days_per_week': 'Plan days per week',
}

def label(col):
    return LABELS.get(col, col.replace('_', ' ').capitalize())

# Highly skewed variables: log scale helps, so it starts on; for the others it starts off
LOG_METRICS = {'lifetime_value_clean', 'total_visits', 'meeting_room_hours',
               'event_space_hours', 'guest_passes_issued', 'avg_weekly_visits'}

# Sidebar filters
st.sidebar.header("Filters")
mask = pd.Series(True, index=df.index)

if 'Plan' in df.columns:
    plans = [p for p in PLAN_ORDER if p in set(df['Plan'])] + (['Unknown'] if 'Unknown' in set(df['Plan']) else [])
    chosen = st.sidebar.multiselect("Plan", plans, default=plans)
    mask &= df['Plan'].isin(chosen)

if 'Account' in df.columns:
    accounts = sorted(df['Account'].unique())
    chosen = st.sidebar.multiselect("Account type", accounts, default=accounts)
    mask &= df['Account'].isin(chosen)

if 'Gender' in df.columns:
    genders = sorted(df['Gender'].unique())
    chosen = st.sidebar.multiselect("Gender", genders, default=genders)
    mask &= df['Gender'].isin(chosen)

if 'age' in df.columns and df['age'].notna().any():
    lo, hi = int(df['age'].min()), int(df['age'].max())
    age_range = st.sidebar.slider("Age range", lo, hi, (lo, hi))
    mask &= df['age'].between(*age_range) | df['age'].isna()  # unknown ages are kept

filtered = df[mask]
st.sidebar.caption(f"{len(filtered):,} of {len(df):,} members selected")
if filtered.empty:
    st.warning("No members match the current filters.")
    st.stop()


# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Members", f"{len(filtered):,}")
if 'lifetime_value_clean' in filtered:
    k2.metric("Median LTV (EUR)", f"{filtered['lifetime_value_clean'].median():,.0f}")
if 'total_visits' in filtered:
    k3.metric("Median total visits", f"{filtered['total_visits'].median():,.0f}")
if 'Account' in filtered:
    k4.metric("Team accounts", f"{(filtered['Account'] == 'Team').mean() * 100:.1f}%")

tab_over, tab_dist, tab_drill, tab_data = st.tabs(
    ["Overview", "Distributions", "Drill-down", "Data"])



# Overview
with tab_over:
    c1, c2 = st.columns(2)
    if 'Plan' in filtered:
        counts = filtered['Plan'].value_counts().reset_index()
        counts.columns = ['Plan', 'Members']
        counts['is_unknown'] = counts['Plan'] == 'Unknown'
        counts = counts.sort_values(['is_unknown', 'Members'], ascending=[True, False])
        order = counts['Plan'].tolist()
        fig = px.bar(counts, x='Members', y='Plan', orientation='h', text='Members',
                     color_discrete_sequence=[COLOR], title="Members by plan")
        fig.update_layout(yaxis={'categoryorder': 'array', 'categoryarray': order[::-1]})
        c1.plotly_chart(fig, use_container_width=True)
    if 'age' in filtered and 'Gender' in filtered:
        fig = px.histogram(filtered.dropna(subset=['age']), x='age', color='Gender', nbins=30,
                           title="Age distribution by gender", barmode='stack')
        c2.plotly_chart(fig, use_container_width=True)
    

# Distributions
with tab_dist:
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    variable = c1.selectbox("Variable", num_cols, format_func=label,
                        index=num_cols.index('total_visits') if 'total_visits' in num_cols else 0)
    split = c2.selectbox("Split by", ['None'] + list(group_options), index=0)
    log_x = c3.checkbox("Log scale", value=False)
    bins = c4.slider("Bins", 10, 100, 40)

    data = filtered.dropna(subset=[variable])

    # Discrete variable = integers with few distinct values (e.g. n_colleagues_on_account)
    discrete = data[variable].nunique() <= 15 and (data[variable] % 1 == 0).all()

    if log_x and not discrete:
        data = data[data[variable] > 0]

    fig = px.histogram(data, x=variable,
                       nbins=None if discrete else bins,
                       log_x=log_x and not discrete,
                       color=None if split == 'None' else split,
                       color_discrete_sequence=[COLOR] if split == 'None' else None,
                       barmode='group' if discrete else 'overlay',
                       opacity=1 if discrete else 0.75)

    if discrete:
        fig.update_traces(xbins=dict(start=-0.5, size=1))   # one bar per integer value
        fig.update_xaxes(dtick=1)                           # show 0, 1, 2, 3...
        fig.update_yaxes(type='log' if log_x else 'linear') # "Log scale" is applied to the y-axis instead
        fig.update_layout(bargap=0.15)

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data[variable].describe().round(2).to_frame().T, use_container_width=True)

# Drill-down
with tab_drill:
    c1, c2, c3 = st.columns([2, 2, 1])
    group_col = c1.selectbox("Group by", list(group_options), key='drill_group')
    metric = c2.selectbox("Metric", num_cols, format_func=label,
                      index=num_cols.index('lifetime_value_clean') if 'lifetime_value_clean' in num_cols else 0,
                      key='drill_metric')
    log_y = c3.checkbox("Log scale", value=metric in LOG_METRICS, key=f'drill_log_{metric}')

    work = filtered.dropna(subset=[metric]).copy()
    work[group_col] = top_n_labels(work[group_col], 10)
    if log_y:
        work = work[work[metric] > 0]

    order = PLAN_ORDER + ['Unknown'] if group_col == 'Plan' else None
    fig = px.box(work, x=group_col, y=metric, log_y=log_y, points=False,
                 color_discrete_sequence=[COLOR],
                 category_orders={group_col: order} if order else None,
                 title=f"{label(metric)} by {group_col.lower()}")
    st.plotly_chart(fig, use_container_width=True)

    summary = (filtered.assign(**{group_col: top_n_labels(filtered[group_col], 10)})
               .groupby(group_col)[metric]
               .agg(Members='size', Mean='mean', Median='median',
                    **{'Zeros (%)': lambda s: (s == 0).mean() * 100})
               .round(2).sort_values('Members', ascending=False))
summary['Note'] = np.where(summary['Members'] < 100, 'Small sample', '')
small = summary.index[summary['Members'] < 100].tolist()
if small:
    st.caption(f"Caution: {', '.join(map(str, small))} has fewer than 100 members, "
               "so its box and averages are less reliable.")   
    st.dataframe(summary, use_container_width=True)

    st.markdown("**Drill into one segment**")
    value = st.selectbox(f"Select a {group_col.lower()}", list(summary.index), key='drill_value')
    segment = filtered.assign(**{group_col: top_n_labels(filtered[group_col], 10)})
    segment = segment[segment[group_col] == value]
    compare = pd.DataFrame({
        'Segment (median)': segment[num_cols].median(),
        'All filtered members (median)': filtered[num_cols].median(),
    }).round(2)
    compare['Difference (%)'] = ((compare.iloc[:, 0] / compare.iloc[:, 1].replace(0, np.nan) - 1) * 100).round(1)
    compare.index = [label(c) for c in compare.index]
    st.caption(f"{len(segment):,} members in this segment")
    st.dataframe(compare, use_container_width=True)


# Data
with tab_data:
    st.dataframe(filtered.head(1000), use_container_width=True)
    st.caption("First 1,000 rows of the filtered data.")