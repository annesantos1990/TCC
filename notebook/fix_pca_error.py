
# Fix for TypeError: list indices must be integers or slices, not str
# The error occurs because NODE_FEATURES is a list, but the code tries to access it like a dictionary.

# 1. Define the group dictionary
NODE_FEATURE_PREFIXES = [
    "degree_",
    "clustering_",
    "betweenness_",
    "local_efficiency_",
    "hub_frequency_",
    "weighted_degree_"
]

# Create a dictionary where keys are the metric names (e.g., 'degree')
NODE_FEATURES_BY_TYPE = {
    prefix.rstrip("_"): [c for c in NODE_FEATURES if c.startswith(prefix)]
    for prefix in NODE_FEATURE_PREFIXES
}

# 2. Define plot_pca_features if it's not already defined
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

def plot_pca_features(df, features, title="PCA"):
    # Filter data
    X = df[features].values
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    
    # Add condition for coloring if available
    if "condition" in df.columns:
        df_pca["condition"] = df["condition"].values
        color_col = "condition"
    else:
        color_col = None
        
    # Plot
    fig = px.scatter(
        df_pca, x="PC1", y="PC2", 
        color=color_col,
        title=title,
        opacity=0.8
    )
    fig.show()
    return pca

# 3. Correct usage
# Use df_dataset (or df) and the new dictionary
pca_degree = plot_pca_features(
    df, # Replace with df_dataset if that's your variable name
    NODE_FEATURES_BY_TYPE["degree"],
    title="PCA das Métricas de Grau (Nós)"
)
