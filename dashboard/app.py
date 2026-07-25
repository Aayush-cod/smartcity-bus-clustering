"""
Smart City Bus Clustering Dashboard
Stakeholder: Urban Planner / Smart City Team
Purpose: Visualise route irregularity clusters from GPS ping data to
support infrastructure and scheduling decisions.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Bee Network Route Analytics",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------
# DATA PATH — update if your folder structure differs
# ---------------------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "dashboard_data_final.csv"
)
# Fallback for a flat folder layout (dashboard_data_final.csv next to this script)
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data_final.csv")

# ---------------------------------------------------------------
# THEME — Bee Network inspired: paper background, amber accent, ink navy text
# ---------------------------------------------------------------
PAPER = "#FAFAF8"
INK = "#1C2333"
AMBER = "#FFC72C"
TEAL = "#0F5C5C"
MUTED = "#6B7280"
CARD = "#FFFFFF"
BORDER = "#E7E4DC"

CLUSTER_COLORS = ["#FFC72C", "#0F5C5C", "#C1440E", "#3B5BA9", "#7A5195", "#2E7D32"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {PAPER};
    color: {INK};
}}

.stApp {{ background-color: {PAPER}; }}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: {INK} !important;
    letter-spacing: -0.01em;
}}

[data-testid="stSidebar"] {{
    background-color: {CARD};
    border-right: 1px solid {BORDER};
}}

.board-row {{
    display: flex;
    gap: 14px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}}
.board-card {{
    background: {INK};
    border-radius: 6px;
    padding: 16px 20px;
    flex: 1;
    min-width: 150px;
    border-bottom: 4px solid {AMBER};
}}
.board-label {{
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #B8BCC8;
    margin-bottom: 6px;
}}
.board-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #FFFFFF;
    line-height: 1;
}}

.cluster-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 12px;
}}
.cluster-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 16px;
    margin-bottom: 8px;
}}
.cluster-stat {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    color: {MUTED};
    margin-bottom: 2px;
}}

.eyebrow {{
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEAL};
    font-weight: 600;
    margin-bottom: 4px;
}}

hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # -1 was used as a "no valid value" sentinel during feature engineering
    # (first ping for a vehicle has no speed; unmatched lines have no deviation).
    # Convert back to NaN here so averages correctly exclude these rows
    # instead of being dragged down by a meaningless -1.
    df["speed_kmh_capped"] = df["speed_kmh_capped"].replace(-1, np.nan)
    df["nearest_schedule_deviation_minutes_v2"] = df["nearest_schedule_deviation_minutes_v2"].replace(-1, np.nan)
    return df

if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at: {DATA_PATH}\n\nUpdate DATA_PATH at the top of app.py to point to your dashboard_data_final.csv.")
    st.stop()

df = load_data(DATA_PATH)

@st.cache_data
def build_cluster_names(data):
    """Derive cluster labels from actual statistics rather than assumed ones."""
    stats = data.groupby("cluster").agg(
        avg_speed=("speed_kmh_capped", "mean"),
        avg_deviation=("nearest_schedule_deviation_minutes_v2", "mean"),
    )
    speed_median = stats["avg_speed"].median()
    deviation_median = stats["avg_deviation"].median()

    names = {}
    for c, row in stats.iterrows():
        tags = []
        if row["avg_deviation"] > deviation_median * 1.5:
            tags.append("Severe Deviation")
        elif row["avg_deviation"] < deviation_median * 0.7:
            tags.append("On-Schedule")
        if row["avg_speed"] < speed_median * 0.7:
            tags.append("Slow-Moving")
        elif row["avg_speed"] > speed_median * 1.15:
            tags.append("Fast-Moving")
        label = " / ".join(tags) if tags else "General Operation"
        names[c] = f"Cluster {int(c)} — {label}"
    return names

CLUSTER_NAMES = build_cluster_names(df)

# ---------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.markdown("### Filters")

operators = sorted(df["nationalOperatorCode"].dropna().unique().tolist())
selected_operators = st.sidebar.multiselect("Operator", operators, default=operators)

lines = sorted(df["lineRef"].dropna().unique().tolist())
selected_lines = st.sidebar.multiselect("Line (leave empty = all)", lines, default=[])

hour_range = st.sidebar.slider("Hour of day", 0, 23, (0, 23))

clusters_available = sorted(df["cluster"].unique().tolist())
selected_clusters = st.sidebar.multiselect(
    "Cluster", clusters_available, default=clusters_available,
    format_func=lambda c: CLUSTER_NAMES.get(c, f"Cluster {c}")
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span style='font-size:12px;color:#6B7280;'>"
    "Data: BODS (Bee Network SIRI-VM, TXC, NeTEx, disruption catalogues)"
    "</span>", unsafe_allow_html=True
)

# apply filters
filtered = df[
    df["nationalOperatorCode"].isin(selected_operators) &
    df["hour_of_day"].between(hour_range[0], hour_range[1]) &
    df["cluster"].isin(selected_clusters)
]
if selected_lines:
    filtered = filtered[filtered["lineRef"].isin(selected_lines)]

# ---------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------
st.markdown("<div class='eyebrow'>Urban Planning · Smart City Team</div>", unsafe_allow_html=True)
st.title("🚌 Bee Network Route Analytics")
st.markdown(
    "Clustering vehicle GPS behaviour to surface route irregularities for infrastructure and scheduling decisions."
)

# ---------------------------------------------------------------
# KPI DEPARTURE BOARD
# ---------------------------------------------------------------
avg_dev = filtered["nearest_schedule_deviation_minutes_v2"].mean()
avg_speed = filtered["speed_kmh_capped"].mean()
avg_disruption = filtered["disruption_count"].mean()
n_pings = len(filtered)
n_lines = filtered["lineRef"].nunique()

st.markdown(f"""
<div class="board-row">
  <div class="board-card">
    <div class="board-label">Total Pings</div>
    <div class="board-value">{n_pings:,}</div>
  </div>
  <div class="board-card">
    <div class="board-label">Active Lines</div>
    <div class="board-value">{n_lines}</div>
  </div>
  <div class="board-card">
    <div class="board-label">Avg Schedule Deviation</div>
    <div class="board-value">{avg_dev:.0f} min</div>
  </div>
  <div class="board-card">
    <div class="board-label">Avg Speed</div>
    <div class="board-value">{avg_speed:.1f} km/h</div>
  </div>
  <div class="board-card">
    <div class="board-label">Avg Disruption Exposure</div>
    <div class="board-value">{avg_disruption:.0f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# MAP
# ---------------------------------------------------------------
st.markdown("### Route Behaviour Map")
st.caption("Each point is a GPS ping, coloured by behavioural cluster. Zoom and pan to inspect specific corridors.")

if len(filtered) > 20000:
    map_sample = filtered.sample(20000, random_state=42)
    st.caption(f"Showing a random sample of 20,000 of {len(filtered):,} filtered points for map performance.")
else:
    map_sample = filtered

if len(map_sample) > 0:
    # cast cluster to a labelled string so Plotly treats it as a CATEGORY, not a numeric scale
    map_sample = map_sample.copy()
    map_sample["cluster_label"] = map_sample["cluster"].map(CLUSTER_NAMES)

    color_map = {CLUSTER_NAMES[c]: CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in clusters_available}

    fig_map = px.scatter_map(
        map_sample,
        lat="latitude", lon="longitude",
        color="cluster_label",
        category_orders={"cluster_label": [CLUSTER_NAMES[c] for c in clusters_available]},
        color_discrete_map=color_map,
        hover_data={"lineRef": True, "nationalOperatorCode": True, "speed_kmh_capped": ":.1f",
                    "nearest_schedule_deviation_minutes_v2": ":.0f", "latitude": False, "longitude": False,
                    "cluster_label": False},
        zoom=9.3,
        height=520,
    )
    fig_map.update_layout(
        map_style="carto-positron",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=PAPER,
        legend_title_text="Behaviour Cluster",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=BORDER, borderwidth=1),
        font=dict(family="Inter", color=INK),
    )
    fig_map.update_traces(marker=dict(size=7, opacity=0.7))
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No data matches the current filters.")

# ---------------------------------------------------------------
# DELAY SEVERITY RANKING — which cluster/colour means more delay
# ---------------------------------------------------------------
st.markdown("#### Which colour means more delay?")
st.caption("Clusters ranked by average schedule deviation — read this alongside the map legend above.")

severity = filtered.groupby("cluster").agg(
    avg_deviation=("nearest_schedule_deviation_minutes_v2", "mean"),
    pings=("cluster", "count"),
).reset_index().sort_values("avg_deviation", ascending=True)
severity["label"] = severity["cluster"].map(CLUSTER_NAMES)
severity["color"] = severity["cluster"].apply(lambda c: CLUSTER_COLORS[c % len(CLUSTER_COLORS)])

fig_severity = px.bar(
    severity, x="avg_deviation", y="label", orientation="h",
    text=severity["avg_deviation"].round(0).astype(int).astype(str) + " min",
)
fig_severity.update_traces(marker_color=severity["color"], textposition="outside")
fig_severity.update_layout(
    plot_bgcolor=PAPER, paper_bgcolor=PAPER,
    font=dict(family="Inter", color=INK),
    xaxis_title="Average schedule deviation (minutes)", yaxis_title="",
    height=280, margin=dict(l=0, r=40, t=10, b=0),
    showlegend=False,
)
st.plotly_chart(fig_severity, use_container_width=True)

# ---------------------------------------------------------------
# CLUSTER PROFILES
# ---------------------------------------------------------------
st.markdown("### Cluster Profiles")
st.caption("Behavioural summary per cluster — helps translate cluster numbers into operational meaning.")

profile = filtered.groupby("cluster").agg(
    pings=("cluster", "count"),
    avg_speed=("speed_kmh_capped", "mean"),
    avg_deviation=("nearest_schedule_deviation_minutes_v2", "mean"),
    avg_disruption=("disruption_count", "mean"),
    top_operator=("nationalOperatorCode", lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"),
).reset_index()

cols = st.columns(3)
for i, row in profile.iterrows():
    with cols[i % 3]:
        c = int(row["cluster"])
        color = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        st.markdown(f"""
        <div class="cluster-card" style="border-left: 5px solid {color};">
            <div class="cluster-title">{CLUSTER_NAMES.get(c, f'Cluster {c}')}</div>
            <div class="cluster-stat">Pings: {row['pings']:,}</div>
            <div class="cluster-stat">Avg speed: {row['avg_speed']:.1f} km/h</div>
            <div class="cluster-stat">Avg deviation: {row['avg_deviation']:.0f} min</div>
            <div class="cluster-stat">Avg disruption exposure: {row['avg_disruption']:.0f}</div>
            <div class="cluster-stat">Top operator: {row['top_operator']}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# TOP DELAYED LINES TABLE
# ---------------------------------------------------------------
st.markdown("### Most Delayed Lines")
st.caption("Lines ranked by average schedule deviation — a starting point for prioritising which routes to investigate.")

line_ranking = filtered.groupby("lineRef").agg(
    pings=("lineRef", "count"),
    avg_deviation_min=("nearest_schedule_deviation_minutes_v2", "mean"),
    avg_speed_kmh=("speed_kmh_capped", "mean"),
    top_operator=("nationalOperatorCode", lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"),
    dominant_cluster=("cluster", lambda x: x.mode().iloc[0] if not x.mode().empty else "N/A"),
).reset_index()

line_ranking = line_ranking[line_ranking["pings"] >= 20]  # drop lines with too few pings to be reliable
line_ranking["avg_deviation_min"] = line_ranking["avg_deviation_min"].round(0)
line_ranking["avg_speed_kmh"] = line_ranking["avg_speed_kmh"].round(1)
line_ranking["dominant_cluster"] = line_ranking["dominant_cluster"].map(CLUSTER_NAMES)
line_ranking = line_ranking.dropna(subset=["avg_deviation_min"]).sort_values("avg_deviation_min", ascending=False).head(15)
line_ranking["avg_deviation_min"] = line_ranking["avg_deviation_min"].astype(int)
line_ranking["avg_speed_kmh"] = line_ranking["avg_speed_kmh"].apply(lambda v: "N/A" if pd.isna(v) else v)
line_ranking.columns = ["Line", "Pings", "Avg Deviation (min)", "Avg Speed (km/h)", "Top Operator", "Dominant Cluster"]

st.dataframe(
    line_ranking,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Avg Deviation (min)": st.column_config.ProgressColumn(
            "Avg Deviation (min)",
            min_value=0,
            max_value=int(line_ranking["Avg Deviation (min)"].max()) if len(line_ranking) else 100,
            format="%d min",
        ),
    },
)

# ---------------------------------------------------------------
# MODEL SELECTION TRANSPARENCY
# ---------------------------------------------------------------
with st.expander("Why K-Means? Model comparison"):
    comparison = pd.DataFrame([
        {"Model": "K-Means (selected)", "Silhouette Score": 0.456, "Notes": "Balanced, interpretable clusters (6-51K points each)"},
        {"Model": "Gaussian Mixture Model", "Silhouette Score": 0.307, "Notes": "Gaussian assumption violated by one-hot categorical features"},
        {"Model": "DBSCAN", "Silhouette Score": 0.168, "Notes": "Best config still fragmented; high-eps configs degenerate to a single dominant cluster"},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# HOURLY PATTERN
# ---------------------------------------------------------------
st.markdown("### Ping Volume by Hour")
st.caption("Note: reflects the snapshot collection schedule, not true diurnal ridership — see report limitations.")

hourly = filtered.groupby("hour_of_day").size().reset_index(name="count")
fig_hour = px.bar(hourly, x="hour_of_day", y="count", color_discrete_sequence=[TEAL])
fig_hour.update_layout(
    plot_bgcolor=PAPER, paper_bgcolor=PAPER,
    font=dict(family="Inter", color=INK),
    xaxis_title="Hour (24h)", yaxis_title="Pings",
    height=300, margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_hour, use_container_width=True)

st.markdown("---")
st.caption("Smart City Bus Clustering — Greater Manchester Bee Network · Coursework Project")