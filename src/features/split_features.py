import pandas as pd

GLOBAL_FEATURES = [
    "global_efficiency",
    "edges_mean",
    "edges_std",
    "edges_cv"
]

def split_features(df: pd.DataFrame):
    id_cols = ["subject_id", "condition"]

    node_features = [
        c for c in df.columns
        if "_" in c and c.split("_")[-1].isdigit()
    ]

    df_nodes = df[id_cols + node_features]
    df_global = df[id_cols + GLOBAL_FEATURES]

    return df_nodes, df_global, node_features, GLOBAL_FEATURES
