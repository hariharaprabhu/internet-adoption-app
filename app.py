import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import os
from src.visualization import plot_socioeconomic_clusters

# Constants
MERGED_PATH = "outputs/all_msa_merged.csv"
SUMMARY_PATH = "outputs/all_msa_clusters.csv"
GEOJSON_FOLDER = "outputs/geojson"  # <-- fixed directory

st.set_page_config(page_title="Broadband Equity Dashboard", layout="wide")

st.info("""
🔧 **This dashboard is under active development.**
Data and insights are updated regularly, and the tool may change as we incorporate feedback.

📌 **Disclaimer:** ZIP codes that span multiple counties are associated with the county having the highest total population (based on HUD `TOT_RATIO`).
""")

st.title("📡 Broadband Equity Dashboard")

@st.cache_data
def load_data():
    merged_df = pd.read_csv(MERGED_PATH, dtype={"zip": str})
    cluster_summary = pd.read_csv(SUMMARY_PATH)
    return merged_df, cluster_summary

# Load data
merged_df, cluster_summary = load_data()
available_msas = sorted(merged_df["MSA"].unique())

# Select metro
selected_msa = st.sidebar.selectbox("Select Metro Area", available_msas)
st.markdown(f"### Metro: {selected_msa}")

# Filter MSA data
msa_df = merged_df[merged_df["MSA"] == selected_msa].copy()
msa_summary = cluster_summary[cluster_summary["MSA"] == selected_msa]

# === Hierarchical Filters ===
st.sidebar.markdown("### Additional Filters")

# 1. County filter
county_options = sorted(msa_df["County"].dropna().unique())
selected_counties = st.sidebar.multiselect("Filter by County", county_options)

# 2. City options filtered by county
if selected_counties:
    city_options = sorted(msa_df[msa_df["County"].isin(selected_counties)]["city"].dropna().unique())
else:
    city_options = sorted(msa_df["city"].dropna().unique())
selected_cities = st.sidebar.multiselect("Filter by City", city_options)

# 3. ZIP options filtered by city or county
if selected_cities:
    zip_options = sorted(msa_df[msa_df["city"].isin(selected_cities)]["zip"].dropna().unique())
elif selected_counties:
    zip_options = sorted(msa_df[msa_df["County"].isin(selected_counties)]["zip"].dropna().unique())
else:
    zip_options = sorted(msa_df["zip"].dropna().unique())
selected_zips = st.sidebar.multiselect("Filter by ZIP", zip_options)

# Apply filters in order
if selected_counties:
    msa_df = msa_df[msa_df["County"].isin(selected_counties)]
if selected_cities:
    msa_df = msa_df[msa_df["city"].isin(selected_cities)]
if selected_zips:
    msa_df = msa_df[msa_df["zip"].isin(selected_zips)]

# Load GeoJSON file
geojson_file = os.path.join(GEOJSON_FOLDER, f"{selected_msa.lower().replace(' ', '_')}_zips.geojson")
if not os.path.exists(geojson_file):
    st.error(f"GeoJSON file not found: {geojson_file}")
    st.stop()

with open(geojson_file, "r") as f:
    geojson = json.load(f)

# Plot map
fig = plot_socioeconomic_clusters(msa_df, city_name=selected_msa, geojson_path=geojson_file)

# Metrics logic
if "avg_adoption_rate" in msa_summary.columns:
    adoption = msa_summary["avg_adoption_rate"].iloc[0] * 100
    availability = msa_summary["avg_availability"].iloc[0] * 100
    income = msa_summary["avg_median_income"].iloc[0]
    population = msa_summary["total_population"].iloc[0]
else:
    adoption = msa_df["internet_adoption_rate"].mean() * 100
    availability = msa_df["fcc_gigabit_coverage"].mean() * 100
    income = msa_df["median_income"].median()
    population = msa_df["total_population"].sum()

# Show metrics
st.subheader("Summary Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Avg Internet Adoption", f"{adoption:.1f}%")
col2.metric("Avg Availability", f"{availability:.1f}%")
col3.metric("Median Income", f"${income:,.0f}")
col4.metric("Total Population", f"{int(population):,}")
col5.metric("ZIPs Considered", f"{msa_df['zip'].nunique():,}")
col6.metric("Counties Involved", f"{msa_df['County'].nunique():,}")

# Show map
st.plotly_chart(fig, use_container_width=True)

# Cluster Drilldown Table
msa_summary_display = msa_summary.copy()
msa_summary_display["internet_adoption_rate"] = msa_summary_display["internet_adoption_rate"].round(2)
msa_summary_display["median_income"] = msa_summary_display["median_income"].apply(lambda x: f"${int(x):,}")
percentage_cols = ["poverty_rate", "pct_white", "pct_black", "pct_hispanic", "pct_seniors", "pct_under18"]
for col in percentage_cols:
    if col in msa_summary_display.columns:
        msa_summary_display[col] = msa_summary_display[col].apply(lambda x: f"{x*100:.1f}%" if x < 1 else f"{x:.1f}%")
msa_summary_display = msa_summary_display.rename(columns={
    "zip": "ZIP Code",
    "internet_adoption_rate": "Adoption Rate",
    "median_income": "Median Income",
    "pct_white": "% White", "pct_black": "% Black", "pct_hispanic": "% Hispanic",
    "pct_seniors": "% Seniors", "pct_under18": "% Under 18",
    "poverty_rate": "Poverty Rate"
})
st.markdown("### Explore Cluster Details")
st.dataframe(msa_summary_display, use_container_width=True)

# ZIP-Level Internet Adoption
st.markdown("### ZIP Codes by Internet Adoption")
adoption_sort = st.selectbox("Select Group", ["Low Internet Adoption(Top 10)", "High Internet Adoption (Top 10)"])
plot_df = msa_df.copy()
plot_df["zip"] = plot_df["zip"].astype(str).str.zfill(5)
plot_df["internet_adoption_rate"] = plot_df["internet_adoption_rate"].astype(float)
if adoption_sort == "Low Internet Adoption(Top 10)":
    plot_df = plot_df.sort_values("internet_adoption_rate", ascending=True).head(10)
else:
    plot_df = plot_df.sort_values("internet_adoption_rate", ascending=False).head(10)
fig_zip_adoption = px.bar(
    plot_df,
    x="zip",
    y="internet_adoption_rate",
    title=f"{adoption_sort} in {selected_msa}",
    labels={"zip": "ZIP Code", "internet_adoption_rate": "Internet Adoption Rate"},
    hover_data=["city", "median_income"]
)
fig_zip_adoption.update_layout(
    yaxis=dict(range=[0, 1], title="Adoption Rate (0–1)"),
    xaxis=dict(title="ZIP Code", type='category'),
    margin=dict(t=50, b=40, l=40, r=10)
)
st.plotly_chart(fig_zip_adoption, use_container_width=True)

# Correlation Analysis
st.markdown("## 🔍 Internet Adoption Drivers by ZIP")
corr_vars = {
    "Median Income": "median_income",
    "% White": "pct_white",
    "% Black": "pct_black",
    "% Hispanic": "pct_hispanic",
    "% Seniors": "pct_seniors",
    "% Under 18": "pct_under18",
    "Poverty Rate": "poverty_rate",
    "Avg Household Size": "avg_household_size"
}
correlation_results = []
for label, var in corr_vars.items():
    if var in msa_df.columns:
        df_valid = msa_df[[var, "internet_adoption_rate"]].dropna()
        corr = np.corrcoef(df_valid[var], df_valid["internet_adoption_rate"])[0, 1]
        correlation_results.append((label, corr))
correlation_df = pd.DataFrame(correlation_results, columns=["Feature", "Correlation"])
correlation_df = correlation_df.sort_values("Correlation", ascending=False)
st.markdown("### 🔗 Correlation with Internet Adoption")
fig_corr_bar = px.bar(
    correlation_df,
    x="Correlation",
    y="Feature",
    orientation="h",
    color="Correlation",
    color_continuous_scale="RdBu",
    range_color=[-1, 1],
    labels={"Correlation": "Correlation Coefficient"},
    height=400
)
fig_corr_bar.update_layout(showlegend=False)
st.plotly_chart(fig_corr_bar, use_container_width=True)
