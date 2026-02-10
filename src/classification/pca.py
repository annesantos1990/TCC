import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
import seaborn as sns
pio.templates.default = "plotly_dark"

DATASET_PATH = "data/processed/dataset_features.parquet"

df_dataset = pd.read_parquet(DATASET_PATH)
df_dataset.shape
df_filtered = df_dataset.columns[df_dataset.columns.str.contains("age", case=False)]
df_filtered

NODE_FEATURE_PREFIXES = [
    "degree_",
    "clustering_",
    "betweenness_",
    "local_efficiency_",
    "hub_frequency_",
    "weighted_degree_"
]

NODE_FEATURES = [
    c for c in df_dataset.columns
    if any(c.startswith(p) for p in NODE_FEATURE_PREFIXES)
]

NODE_GROUPS = {
    "degree": [c for c in df_dataset.columns if c.startswith("degree_")],
    "clustering": [c for c in df_dataset.columns if c.startswith("clustering_")],
    "betweenness": [c for c in df_dataset.columns if c.startswith("betweenness_")],
    "local_efficiency": [c for c in df_dataset.columns if c.startswith("local_efficiency_")],
    "weighted_degree": [c for c in df_dataset.columns if c.startswith("weighted_degree_")]
}


len(NODE_FEATURES)

X = df_dataset[NODE_FEATURES].values
y = df_dataset["condition"].values

# Padronização dass features

X = np.nan_to_num(X, nan=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled.mean(axis=0)[:5], X_scaled.std(axis=0)[:5]

# PCA com dois componentes


# Create a dictionary where keys are the metric names (e.g., 'degree')
NODE_FEATURE_PREFIXES = [
    "degree_",
    "clustering_",
    "betweenness_",
    "local_efficiency_",
    "hub_frequency_",
    "weighted_degree_"
]

NODE_FEATURES = [
    c for c in df_dataset.columns
    if any(c.startswith(p) for p in NODE_FEATURE_PREFIXES)
]

df_dataset = df_dataset.rename(columns={
    "Gender_ 1=female_2=male": "gender",
    "Age": "age",
    "Handedness": "handedness",
    "Education": "education",
    "DRUG_0=negative_1=Positive": "drug",
    "Smoking_num_(Non-smoker=1, Occasional Smoker=2, Smoker=3)": "smoking",
    "SKID_Diagnoses": "skid_diagnoses",
    "SKID_Diagnoses 1": "skid_diagnoses_1",
    "SKID_Diagnoses 2": "skid_diagnoses_2",
    "Comments_SKID_assessment": "comments_skid_assessment",
    "Hamilton_Scale": "hamilton_scale",
    "BSL23_sumscore": "bsl23_sumscore",
    "BSL23_behavior": "bsl23_behavior",
    "AUDIT": "audit",
    "Standard_Alcoholunits_Last_28days": "standard_alcoholunits_last_28days",
    "Alcohol_Dependence_In_1st-3rd_Degree_relative": "alcohol_dependence_in_1st_3rd_degree_relative",
    "Relationship_Status": "relationship_status"
})


X = df_dataset[NODE_FEATURES]
meta = df_dataset[["condition", "age", "gender"]]

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
pca = PCA(n_components=2)

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca = pd.concat([df_pca, meta.reset_index(drop=True)], axis=1)


color_vars = ["condition", "age", "gender"]

for var in color_vars:
    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        color=var,
        title=f"PCA colored by {var}",
        opacity=0.8,
        template="plotly_dark"
    )
    fig.show()

