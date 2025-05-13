import streamlit as st
import pandas as pd
from src.visualization import plot_socioeconomic_clusters
import plotly.express as px
import numpy as np

# Constants
CITY_NAME = "Austin"
geojson_path = "outputs/austin_zips.geojson"
MERGED_PATH = "outputs/merged_data.csv"
SUMMARY_PATH = "outputs/cluster_summary.csv"

st.set_page_config(page_title="Broadband Equity Dashboard", layout="wide")
st.info("""
🔧 **This dashboard is under active development.**
Data and insights are updated regularly, and the tool may change as we incorporate feedback.

If you spot an issue or have suggestions, feel free to reach out or check back for the latest version.
""")


st.title("📡 Broadband Equity Dashboard")
st.markdown(f"### City: {CITY_NAME}")





@st.cache_data
def load_data():
    merged_df = pd.read_csv(MERGED_PATH)
    cluster_summary = pd.read_csv(SUMMARY_PATH)
    fig = plot_socioeconomic_clusters(merged_df, city_name=CITY_NAME, geojson_path=geojson_path)
    return merged_df, cluster_summary, fig

merged_df, cluster_summary, fig = load_data()

# --- Use metrics from file if available ---
if "avg_adoption_rate" in cluster_summary.columns:
    adoption = cluster_summary["avg_adoption_rate"].iloc[0]
    availability = cluster_summary["avg_availability"].iloc[0]
    income = cluster_summary["avg_median_income"].iloc[0]
    population = cluster_summary["total_population"].iloc[0]
else:
    # fallback average
    adoption = merged_df["internet_adoption_rate"].mean()
    availability = merged_df["fcc_gigabit_coverage"].mean()
    income = merged_df["median_income"].median()
    population = merged_df["total_population"].sum()
    zips = merged_df["zip"].nunique()

# --- Display metrics ---
st.subheader("Summary Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Avg Internet Adoption", f"{adoption:.1f}%")
col2.metric("Avg Availability", f"{availability:.1f}%")
col3.metric("Median Income", f"${income:,.0f}")
col4.metric("Total Population", f"{int(population):,}")
col5.metric("Number of Zips Considered", f"{int(zips):,}")

with st.expander("ℹ️ About ZIP Code Clustering"):
    st.markdown("""
### How ZIP Code Clusters Are Formed

ZIP codes are grouped into clusters based on **similar socioeconomic patterns** using a machine learning technique called **K-Means Clustering**.

Each color on the map represents a **distinct group of ZIP codes** that share characteristics such as income, education, ethinic and age distribution.

---

### Features Used for Grouping ZIPs

The clusters are based on publicly available Census data, including:

- **Median Income**
- **Poverty Rate**
- **% Seniors (Age 65+)**
- **% Under 18**
- **Average Household Size**
- **% with Bachelor's Degree or Higher**
- **Racial/Ethnic Distribution** (% White, % Black, % Hispanic)

These features were chosen to capture **socioeconomic conditions** that may affect access to technology and infrastructure.

---

### How the Number of Clusters Is Decided

We don’t choose the number of clusters manually. Instead, we test multiple options (typically 2 to 5) and pick the one that performs best using a metric called the **Silhouette Score**, which measures how clearly defined each group is.

This ensures that:

- ZIP codes within each group are **similar** to each other.
- Groups are **meaningfully distinct** from each other.

Different cities may have a different number of clusters based on their population diversity.

---

### What the Colors and Shading Mean

- **Color** indicates which socioeconomic cluster a ZIP code belongs to.
- **Darker shading** within each color group means that ZIP code has **lower internet adoption** compared to others in its group.

This helps identify communities that may need more targeted support — even among those with similar economic conditions.
""")

# --- Plot ---
#st.markdown("### ZIP Code Clusters by Socioeconomic Profile")
st.plotly_chart(fig, use_container_width=True)

# --- Cluster Drilldown ---
st.markdown("### Explore Cluster Details")


# Load and format cluster_df
cluster_df = pd.read_csv(SUMMARY_PATH)

# Round adoption rate
if "internet_adoption_rate" in cluster_df.columns:
    cluster_df["internet_adoption_rate"] = cluster_df["internet_adoption_rate"].round(2)

# Format median income
if "median_income" in cluster_df.columns:
    cluster_df["median_income"] = cluster_df["median_income"].apply(lambda x: f"${int(x):,}")

# Format percentages
percentage_cols = ["poverty_rate", "pct_white", "pct_black", "pct_hispanic", "pct_seniors", "pct_under18"]
for col in percentage_cols:
    if col in cluster_df.columns:
        cluster_df[col] = cluster_df[col].apply(lambda x: f"{x*100:.1f}%" if x < 1 else f"{x:.1f}%")

# Rename for readability
cluster_df = cluster_df.rename(columns={
    "zip": "ZIP Code",
    "internet_adoption_rate": "Adoption Rate",
    "median_income": "Median Income",
    "pct_white": "% White", "pct_black": "% Black", "pct_hispanic": "% Hispanic",
    "pct_seniors": "% Seniors", "pct_under18": "% Under 18",
    "poverty_rate": "Poverty Rate"
})

# Display in app
if len(cluster_df) > 25 or len(cluster_df.columns) > 10:
    with st.expander("🔍 View ZIP-Level Cluster Data"):
        st.dataframe(cluster_df, use_container_width=True)
else:
    st.markdown("### ZIP-Level Cluster Data")
    st.dataframe(cluster_df, use_container_width=True)

st.markdown("### ZIP Codes by Internet Adoption")

# Dropdown for sort type
adoption_sort = st.selectbox("Select Group", ["Low Internet Adoption(Top 10)", "High Internet Adoption (Top 10)"])

# Clean and prepare DataFrame
#st.dataframe(merged_df.head())

plot_df = merged_df.copy()
plot_df["zip"] = plot_df["zip"].astype(str).str.zfill(5)  # ensure ZIPs have leading zeros
plot_df["internet_adoption_rate"] = plot_df["internet_adoption_rate"].astype(float)

# Sort and select
if adoption_sort == "Zips with Low Adoption(Top 10)":
    plot_df = plot_df.sort_values("internet_adoption_rate", ascending=True).head(10)
else:
    plot_df = plot_df.sort_values("internet_adoption_rate", ascending=False).head(10)

# Now make the ZIP codes the x-axis categories
fig_zip_adoption = px.bar(
    plot_df,
    x="zip",
    y="internet_adoption_rate",
    title=f"{adoption_sort} in {CITY_NAME}",
    labels={"zip": "ZIP Code", "internet_adoption_rate": "Internet Adoption Rate"},
    hover_data=["city", "median_income"]
)

# Layout tweaks
fig_zip_adoption.update_layout(
    yaxis=dict(range=[0, 1], title="Adoption Rate (0–1)"),
    xaxis=dict(title="ZIP Code", type='category'),  # treat ZIPs as category
    margin=dict(t=50, b=40, l=40, r=10)
)

st.plotly_chart(fig_zip_adoption, use_container_width=True)


st.markdown("## 🔍 Internet Adoption Drivers by ZIP")

# Define variables to test correlation with
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

# Compute correlation for each
correlation_results = []
for label, var in corr_vars.items():
    if var in merged_df.columns:
        df_valid = merged_df[[var, "internet_adoption_rate"]].dropna()
        corr = np.corrcoef(df_valid[var], df_valid["internet_adoption_rate"])[0, 1]
        correlation_results.append((label, corr))

# Sort by absolute correlation
correlation_df = pd.DataFrame(correlation_results, columns=["Feature", "Correlation"])
correlation_df = correlation_df.sort_values("Correlation", ascending=False)

# --- Correlation Overview (Bar Chart) ---
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

def generate_correlation_summary(correlation_df, top_n_each=5, min_r_threshold=0.05):
    """
    Generate summary using top N positive and negative features with |r| ≥ threshold.
    """
    # Filter to features with meaningful correlation
    filtered_df = correlation_df[correlation_df["Correlation"].abs() >= min_r_threshold]

    # Sort and split
    sorted_df = filtered_df.sort_values("Correlation")
    negative = sorted_df[sorted_df["Correlation"] < 0].head(top_n_each)
    positive = sorted_df[sorted_df["Correlation"] > 0].tail(top_n_each)

    # Format lists
    neg_list = ", ".join([f"**{f}**" for f in negative["Feature"]])
    pos_list = ", ".join([f"**{f}**" for f in positive["Feature"]])

    # Build summary
    summary = ""
    if not negative.empty:
        summary += f"ZIP codes with higher {neg_list.lower()} tend to have **lower internet adoption**.\n\n"
    if not positive.empty:
        summary += f"In contrast, ZIP codes with higher {pos_list.lower()} tend to have **better internet adoption**."

    if not summary:
        summary = "_No strong relationships identified between selected features and internet adoption._"

    return summary


st.markdown("### 📘 Summary for Policymakers")
st.markdown(generate_correlation_summary(correlation_df))


# --- Spacer or Divider ---
st.markdown("---")

# --- Feature Drilldown ---
st.markdown("### 📈 Feature vs Adoption Rate")

selected_label = st.selectbox("Choose a Feature to Explore", correlation_df["Feature"].tolist())
selected_var = corr_vars[selected_label]

# Get correlation value
r = float(correlation_df.loc[correlation_df["Feature"] == selected_label, "Correlation"].values[0])

# Generate plain-English interpretation
def explain_correlation(x_feature: str, r: float):
    """Generate plain-English correlation explanation between X and adoption."""
    
    if abs(r) < 0.05:
        return f"**{x_feature}** shows little to no relationship with Internet Adoption Rate."

    direction = "higher" #if r > 0 else "lower"
    outcome = "higher" if r > 0 else "lower"
    
    return (
        f"ZIP codes with **{direction} {x_feature.lower()}** tend to have "
        f"**{outcome} Internet Adoption Rates** based on the data."
    )

# Show explanation
st.markdown("#### 🔍 Interpretation")
st.markdown(explain_correlation(selected_label,r))

# Drill-down scatter chart
df_scatter = merged_df[[selected_var, "internet_adoption_rate", "city", "zip"]].dropna()
fig_detail = px.scatter(
    df_scatter,
    y=selected_var,
    x="internet_adoption_rate",
    labels={
        "internet_adoption_rate": "Internet Adoption Rate",
        selected_var: selected_label,
    },
    title=f"{selected_label} vs Internet Adoption Rate",
    hover_data=["city", "zip"]
)
fig_detail.update_layout(height=450, margin=dict(t=50, l=40, r=10, b=40))
st.plotly_chart(fig_detail, use_container_width=True)

st.markdown(
    "<div style='text-align: center; font-size: 0.85em; color: gray;'>"
    "This tool is for exploratory and policy research purposes only. Correlation does not imply causation."
    "</div>",
    unsafe_allow_html=True
)
