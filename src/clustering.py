from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

def find_best_kmeans(X, k_range=range(2, 6), random_state=42, n_init=100):
    sil_scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=500)
        labels = kmeans.fit_predict(X)
        sil_scores[k] = silhouette_score(X, labels)
    best_k = max(sil_scores, key=sil_scores.get)
    return best_k, sil_scores


def run_kmeans(X_scaled, k, random_state=42, n_init=100):
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=300)
    labels = model.fit_predict(X_scaled)
    return model, labels

def summarize_cluster_profiles(merged_df):
    """
    Generate a raw (unscaled) cluster-level summary.
    """
    summary = merged_df.groupby('cluster').agg({
        'internet_adoption_rate': 'mean',
        'total_population': 'sum',
        'avg_household_size':'mean',
        'fcc_10mbps_coverage': 'mean',
        'fcc_25mbps_coverage': 'mean',
        'fcc_100mbps_coverage': 'mean',
        'fcc_250mbps_coverage': 'mean',
        'fcc_gigabit_coverage': 'mean',
        'median_income': 'mean',
        'pct_seniors': 'mean',
        'pct_under18': 'mean',
        'poverty_rate': 'mean',
        'pct_white': 'mean',
        'pct_black': 'mean',
        'pct_hispanic': 'mean',
        'pct_bachelor_plus': 'mean',
        'zip': lambda x: list(x.unique())
    }).reset_index()
    return summary

def cluster_analysis(df, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    best_k, sil_scores = find_best_kmeans(X_scaled)
    model, labels = run_kmeans(X_scaled, best_k)

    df['cluster'] = labels
    centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
    centroids['cluster'] = range(best_k)

    return df, centroids, sil_scores, model, scaler

def kmeans_predict_proba(model, X):
    distances = model.transform(X)
    inverse = 1 / (distances + 1e-6)  # prevent divide-by-zero
    probs = inverse / inverse.sum(axis=1, keepdims=True)
    return probs

def generate_lime_summaries(df, features, model, scaler, centroids_df, n_features=3):
    """
    Generate top feature explanations for each ZIP based on LIME,
    showing only the most impactful features with their actual values
    (no High/Low tags), sorted by influence strength.

    Parameters:
        df (pd.DataFrame): Full DataFrame with ZIP records
        features (list): List of clustering features
        model (KMeans): Trained KMeans model
        scaler (StandardScaler): Scaler used before clustering
        centroids_df (pd.DataFrame): Cluster centroid DataFrame
        n_features (int): Number of top influential features to include

    Returns:
        pd.DataFrame: Original df with a new column 'lime_top_features'
    """
    from lime.lime_tabular import LimeTabularExplainer

    friendly_labels = {
        "pct_bachelor_plus": "% Bachelor Plus",
        "pct_seniors": "% Seniors",
        "pct_under18": "% Under 18",
        "pct_white": "% White",
        "pct_black": "% Black",
        "pct_hispanic": "% Hispanic",
        "poverty_rate": "Poverty Rate",
        "median_income": "Median Income",
        "avg_household_size": "Avg Household Size"
    }

    X_scaled = scaler.transform(df[features])
    X_original = df[features].reset_index(drop=True)

    explainer = LimeTabularExplainer(
        training_data=X_scaled,
        feature_names=features,
        class_names=[f"Cluster {i}" for i in range(model.n_clusters)],
        mode='classification',
        discretize_continuous=True,
        sample_around_instance=True,
        random_state=42  # ensures stable output
    )


    explanations = []

    for i in range(len(df)):
        try:
            exp = explainer.explain_instance(
                data_row=X_scaled[i],
                predict_fn=lambda x: kmeans_predict_proba(model, x),
                num_features=n_features,
                num_samples=5000
            )

            local_exp_dict = exp.local_exp
            if not local_exp_dict:
                explanations.append("No explanation available")
                continue

            label_key = list(local_exp_dict.keys())[0]
            summary_parts = []

            # Sort by absolute impact strength (|weight|)
            sorted_feats = sorted(local_exp_dict[label_key], key=lambda x: abs(x[1]), reverse=True)

            for i_feat, _ in sorted_feats[:n_features]:
                feat_name = features[i_feat]
                val = X_original.loc[i, feat_name]

                # Format value for display
                if "income" in feat_name:
                    val_str = f"${int(val):,}"
                elif feat_name.startswith("pct_") or "poverty" in feat_name:
                    val_str = f"{val * 100:.0f}%"
                elif "household" in feat_name:
                    val_str = f"{val:.1f}"
                else:
                    val_str = str(round(val, 2))

                label_name = friendly_labels.get(feat_name, feat_name.replace("_", " ").title())
                summary_parts.append(f"{label_name}: {val_str}")

            explanations.append(", ".join(summary_parts))

        except Exception:
            explanations.append("LIME failed")

    df["lime_top_features"] = explanations
    return df



def cluster_within_clusters(df, base_cluster_col, new_features, n_subclusters=2):
    results = []
    scaler = StandardScaler()

    for label in df[base_cluster_col].unique():
        cluster_df = df[df[base_cluster_col] == label].copy()
        if len(cluster_df) < n_subclusters:
            cluster_df['subcluster'] = 0  # Not enough data to cluster
        else:
            X = cluster_df[new_features]
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)
            cluster_df['subcluster'] = kmeans.fit_predict(X_scaled)

        cluster_df['base_cluster'] = label
        results.append(cluster_df)

    return pd.concat(results, ignore_index=True)

