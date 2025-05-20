import plotly.express as px
import plotly.graph_objects as go
import json
import os
import pandas as pd
import requests
import plotly.graph_objects as go
from shapely.geometry import shape
import shapely
import numpy as np
from collections import defaultdict
import matplotlib.colors as mcolors
import re

def download_geojson_if_missing(geojson_path, url):
    if not os.path.exists(geojson_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(geojson_path, 'w') as f:
                json.dump(response.json(), f)
        else:
            raise Exception(f"Failed to download GeoJSON (status code {response.status_code})")

def prepare_geojson(geojson_path, valid_zips):
    with open(geojson_path, "r") as f:
        geojson = json.load(f)
    features = [f for f in geojson["features"] if f["properties"]["ZCTA5CE10"] in valid_zips]
    for f in features:
        f["id"] = f["properties"]["ZCTA5CE10"]
    return {"type": "FeatureCollection", "features": features}


def format_hover(row):
        #import re

        # Adjust median_income formatting if present in LIME string
        #lime = row["Top Features"].replace("median_income", "Income")
        #lime = re.sub(r"Income: \$?(\d{4,6})", lambda m: f"Income: ${int(m.group(1))//1000}K", lime)

        # Clean original LIME bullets (up to 2 max)
        #lime_parts = lime.replace("<br>", ",").split(",")
        #lime_bullets = [f"• {b.strip()}" for b in lime_parts if b.strip()][:2]

        # Add relevant census fields as bullets
        return (
        f"<b>ZIP:</b> {row['zip']} ({row['city'].title()})<br>"
        f"<b>County:</b> {row['County']}<br>"
        f"<b>Cluster:</b> {row['cluster']}<br>"
        f"<b>Adoption:</b> {row['Internet Adoption Rate']*100:.1f}%<br>"
        f"<b>Avg Household Size:</b> {row['avg_household_size']:.1f}<br>"
        f"<b>Median Income:</b> ${int(row['median_income']):,}<br>"
        f"<b>Poverty Rate:</b> {row['poverty_rate']*100:.0f}%<br><br>"

        f"<b>Race Distribution:</b> (% White: {row['pct_white']*100:.0f}%, "
        f"% Black: {row['pct_black']*100:.0f}%, % Hispanic: {row['pct_hispanic']*100:.0f}%)<br>"

        f"<b>Household Metrics:</b> (% Seniors: {row['pct_seniors']*100:.0f}%, "
        f"% Under 18: {row['pct_under18']*100:.0f}%)<br>"

        f"<b>% Bachelor's Degree or Higher:</b> {row['pct_bachelor_plus']*100:.0f}%"
    )
def rgba_with_opacity(color_str, alpha):
    """Convert hex or rgb string to rgba string with opacity"""
    if color_str.startswith('#'):
        rgb = mcolors.to_rgb(color_str)
        r, g, b = int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
    elif color_str.startswith('rgb'):
        match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
        if not match:
            raise ValueError(f"Invalid RGB format: {color_str}")
        r, g, b = map(int, match.groups())
    else:
        raise ValueError(f"Unsupported color format: {color_str}")
    return f'rgba({r},{g},{b},{alpha:.2f})'


def plot_socioeconomic_clusters(df, city_name, geojson_path, cluster_col="cluster", city_pop_dict=None, city_pop_threshold=50000):
   

    df = df.copy()
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    df = df.rename(columns={
        "internet_adoption_rate": "Internet Adoption Rate",
        "lime_top_features": "Top Features"
    })
    df[cluster_col] = df[cluster_col].astype(str)

    # Safe pastel/complementary neutral color palette
    safe_colors = [
        '#8da0cb', '#fc8d62', '#66c2a5', '#e78ac3', '#a6d854',
        '#b3b3b3', '#abdda4', '#e5c494', '#fdae61', '#b2abd2'
    ]
    cluster_labels = sorted(df[cluster_col].unique())
    cluster_color_map = {
        label: safe_colors[i % len(safe_colors)] for i, label in enumerate(cluster_labels)
    }

    # Reverse intensity logic: low adoption = higher opacity
    df["norm_adoption"] = df.groupby(cluster_col)["Internet Adoption Rate"].transform(
        lambda x: 0.3 + 0.7 * (x.max() - x) / (x.max() - x.min() + 1e-6)
    )
    df["fill_color"] = df.apply(
        lambda row: rgba_with_opacity(cluster_color_map[row[cluster_col]], row["norm_adoption"]),
        axis=1
    )
    df["hover_text"] = df.apply(format_hover, axis=1)

    fig = go.Figure()

    # Draw each polygon with custom color
    for feature in geojson["features"]:
        zip_code = feature["properties"]["ZCTA5CE10"]
        if zip_code not in df["zip"].values:
            continue
        row = df[df["zip"] == zip_code].iloc[0]
        geometry = shape(feature["geometry"])
        if geometry.geom_type == "MultiPolygon":
            polygons = geometry.geoms
        else:
            polygons = [geometry]

        for poly in polygons:
            lon = list(poly.exterior.coords.xy[0])
            lat = list(poly.exterior.coords.xy[1])
            fig.add_trace(go.Scattergeo(
                lon=lon,
                lat=lat,
                fill="toself",
                mode="lines",
                line=dict(color="black", width=0.5),
                fillcolor=row["fill_color"],
                hovertext=row["hover_text"],
                hoverinfo="text",
                showlegend=False
            ))

    # 🔁 Auto-compute city population dictionary if not provided
    if city_pop_dict is None:
        city_pop_dict = (
            df.dropna(subset=["city", "total_population"])
              .groupby(df["city"].str.upper())["total_population"]
              .sum()
              .to_dict()
        )

    # Add city name labels based on population threshold
    city_centers = defaultdict(list)
    for feature in geojson["features"]:
        zip_code = feature["properties"]["ZCTA5CE10"]
        geom = shape(feature["geometry"])
        if not geom.is_valid or zip_code not in df["zip"].values:
            continue
        city_match = df[df["zip"] == zip_code]["city"].values
        if len(city_match) > 0:
            city_centers[city_match[0]].append(geom.centroid)

    city_labels = []
    for city, centroids in city_centers.items():
        city_upper = city.upper()
        if city_pop_dict.get(city_upper, 0) < city_pop_threshold:
            continue  # skip small cities
        avg_x = np.mean([pt.x for pt in centroids])
        avg_y = np.mean([pt.y for pt in centroids])
        jitter_x = avg_x + np.random.uniform(0.01, 0.02)
        jitter_y = avg_y + np.random.uniform(0.01, 0.02)
        city_labels.append((city.title(), jitter_x, jitter_y))

    fig.add_trace(go.Scattergeo(
        lon=[lon for _, lon, lat in city_labels],
        lat=[lat for _, lon, lat in city_labels],
        text=[name for name, _, _ in city_labels],
        mode="text",
        textfont=dict(size=10, color="black"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Layout
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        title=f"{city_name} | ZIP Code Clusters Based on Socioeconomic Patterns<br><sub>Darker areas indicate lower internet adoption within group</sub>",
        height=900,
        width=1000,
        margin=dict(t=30, r=10, l=10, b=10),
        title_font=dict(size=20),
    )

    for cluster_label, color in cluster_color_map.items():
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None],
            mode="markers",
            marker=dict(size=12, color=color),
            name=f"Cluster {cluster_label}"
        ))

    fig.add_annotation(
        text="Colors represent distinct clusters. Opacity reflects internet adoption (darker = lower).",
        showarrow=False,
        xref="paper", yref="paper",
        x=0, y=0,
        font=dict(size=10, color="gray")
    )

    return fig



def plot_nested_clusters(df, city_name, geojson_path, cluster_col="cluster", subcluster_col="broadband_group"):
    df = df.copy()
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    valid_zips = set(df["zip"])
    geojson = prepare_geojson(geojson_path, valid_zips)

    hover_fields = [
        "zip", "internet_adoption_rate", "median_income", "poverty_rate", 
        "pct_bachelor_plus", cluster_col, subcluster_col
    ]

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="zip",
        color=cluster_col,
        locationmode="geojson-id",
        hover_data=hover_fields,
        title=f"{city_name} | ZIP Code Clusters Based on Socioeconomic Patterns /n (Darker areas indicate lower internet adoption within group)",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        height=650,
        margin={"r": 0, "t": 50, "l": 0, "b": 20},
        annotations=[
            go.layout.Annotation(
                text=(
                    "Clusters: socioeconomic groups based on income, education, race, etc.<br>"
                    "Subgroups: broadband access patterns (availability/adoption) within those clusters."
                ),
                x=0.5, y=-0.15, showarrow=False, xref="paper", yref="paper",
                font=dict(size=12), align="center"
            )
        ]
    )

    fig.show()
