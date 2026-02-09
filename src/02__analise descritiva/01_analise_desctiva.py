import pandas as pd
import numpy as np

import plotly.express as px
import plotly.io as pio

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pointbiserialr

pio.templates.default = "plotly_dark"

df_healthy = pd.read_parquet("data/processed/dataset_features.parquet")

print(df_healthy["age"].value_counts())
print(df_healthy["gender"].value_counts())
print(df_healthy["handedness"].value_counts())
print(df_healthy["education"].value_counts())
print(df_healthy["drug"].value_counts())

# garantir tipos
df_healthy["age"] = df_healthy["age"].astype(str)
df_healthy["gender"] = df_healthy["gender"].astype(str)
df_healthy["handedness"] = df_healthy["handedness"].astype(str)
df_healthy["education"] = df_healthy["education"].astype(str)
df_healthy["drug"] = df_healthy["drug"].astype(str)

df_healthy["age_group"] = df_healthy["age"].replace({
    "20-25": "young",
    "25-30": "young",
    "30-35": "young",
    "35-40": "middle",
    "55-60": "older",
    "60-65": "older",
    "65-70": "older",
    "70-75": "older"
})

# Calculando as médias das métricas
degree_cols = [c for c in df_healthy.columns if c.startswith("degree_")]
clustering_cols = [c for c in df_healthy.columns if c.startswith("clustering_")]
efficiency_cols = [c for c in df_healthy.columns if c.startswith("efficiency_")]
betweenness_cols = [c for c in df_healthy.columns if c.startswith("betweenness_")]
df_healthy["degree_mean"] = df_healthy[degree_cols].mean(axis=1)
df_healthy["clustering_mean"] = df_healthy[clustering_cols].mean(axis=1)
df_healthy["efficiency_global"] = df_healthy[efficiency_cols].mean(axis=1)
df_healthy["betweenness_mean"] = df_healthy[betweenness_cols].mean(axis=1)

GRAPH_METRICS = [
    "degree_mean",
    "clustering_mean",  
    "efficiency_global",
    "betweenness_mean"
]

sample_summary = (
    df_healthy[["subject_id", "age_group", "gender"]]
    .drop_duplicates()
    .groupby(["age_group", "gender"])
    .size()
    .reset_index(name="n")
)

sample_summary

df_healthy[["age_group", "gender"]].value_counts()

# Boxplot Age Group
for metric in GRAPH_METRICS:
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        data=df_healthy,
        x="age_group",
        y=metric,
        hue="age_group",
        legend=False
    )
    plt.title(f"{metric} vs Age Group")
    plt.xlabel("Age group")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# plotly
for metric in GRAPH_METRICS:
    fig = px.box(
        df_healthy,
        x="age_group",
        y=metric,
        color="age_group",
        title=f"{metric} vs Age Group",
        labels={
            "age_group": "Age group",
            metric: metric
        }
    )
    fig.update_layout(
        xaxis_title="Age group",
        yaxis_title=metric,
        title=f"{metric} vs Age Group"
    )
    fig.show()

# Boxplot Condition
for metric in GRAPH_METRICS:
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=df_healthy,
        x="condition",
        y=metric,
        palette="Set1"
    )
    plt.title(f"{metric}: EO vs EC")
    plt.xlabel("Condition")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.show()

# Boxplot Age Group x Condition
for metric in GRAPH_METRICS:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=df_healthy,
        x="age_group",
        y=metric,
        hue="condition",
        palette="Set1"
    )
    plt.title(f"{metric}: EO vs EC stratified by age")
    plt.xlabel("Age group")
    plt.ylabel(metric)
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.show()

# Correlation Age Group
age_ord_map = {"young": 0, "middle": 1, "older": 2}
df_healthy["age_ord"] = df_healthy["age_group"].map(age_ord_map)

corr_results = []

for metric in GRAPH_METRICS:
    valid = df_healthy[[metric, "age_ord"]].dropna()
    r, p = spearmanr(valid["age_ord"], valid[metric])
    corr_results.append({
        "metric": metric,
        "spearman_r": r,
        "p_value": p
    })

pd.DataFrame(corr_results)

# Correlation Gender
df_healthy["gender_bin"] = df_healthy["gender"].map({
    "female": 0,
    "male": 1
})

gender_results = []

for metric in GRAPH_METRICS:
    valid = df_healthy[[metric, "gender_bin"]].dropna()
    r, p = pointbiserialr(valid["gender_bin"], valid[metric])
    gender_results.append({
        "metric": metric,
        "point_biserial_r": r,
        "p_value": p
    })

pd.DataFrame(gender_results)

