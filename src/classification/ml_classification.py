import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score


DATASET_PATH = "data/processed/dataset_features.parquet"

df = pd.read_parquet(DATASET_PATH)

df = df[df["condition"].isin(["EO", "EC"])]

df["y"] = df["condition"].map({"EO": 0, "EC": 1})

NODE_FEATURES = [
    c for c in df.columns
    if c.startswith(("degree_", "clustering_", "betweenness_", "hub_"))
]

META_FEATURES = ["age", "gender"]

def evaluate_model(X, y, model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "acc": "accuracy",
        "auc": "roc_auc"
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    acc_mean = scores["test_acc"].mean()
    auc_mean = scores["test_auc"].mean()

    print(
        "=== Model Evaluation (Stratified 5-fold CV) ==="
        f"Mean Accuracy: {acc_mean:.3f} (±{scores['test_acc'].std():.3f})"
        f"Mean AUC: {auc_mean:.3f} (±{scores['test_auc'].std():.3f})"
        "Note: ROC-AUC values close to 0.5 indicate random performance."
    )

    return {
        "accuracy_mean": acc_mean,
        "auc_mean": auc_mean
    }

# Regressão Logística
X = df[NODE_FEATURES].select_dtypes(include="number").dropna(axis=1)
y = df["y"]

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

evaluate_model(X, y, pipe_lr)

# Regressão Logística com PCA
pipe_pca_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression(max_iter=1000))
])

evaluate_model(X, y, pipe_pca_lr)

# Regressão Logística com PCA e metadados
X_meta = pd.concat(
    [X, df[META_FEATURES]],
    axis=1
).dropna()

y_meta = df.loc[X_meta.index, "y"]

pipe_pca_meta = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression(max_iter=1000))
])

evaluate_model(X_meta, y_meta, pipe_pca_meta)