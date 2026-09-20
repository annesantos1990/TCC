"""
Classificação de grupo etário (young vs older) a partir de métricas de rede.

Tarefas:
  - EO: features só olhos abertos
  - EC: features só olhos fechados
  - Diff: EC − EO por sujeito

Modelos: Dummy, LogisticRegression (L2), SVM RBF, Random Forest, KNN+PCA
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ==========================
# PATHS / PARAMS
# ==========================
DATASET_PATH = Path("data/processed/dataset_features.parquet")
RANDOM_STATE = 42
N_SPLITS = 5
PCA_COMPONENTS = 10
KNN_NEIGHBORS = 5

YOUNG_AGES = {"20-25", "25-30", "30-35", "35-40"}
OLDER_AGES = {"55-60", "60-65", "65-70", "70-75", "75-80"}

NODE_PREFIXES = (
    "degree_",
    "clustering_",
    "betweenness_",
    "local_efficiency_",
    "hub_frequency_",
    "weighted_degree_",
)
GLOBAL_FEATURES = [
    "global_efficiency",
    "edges_mean",
    "edges_std",
    "edges_cv",
]


# ==========================
# LOAD
# ==========================
df = pd.read_parquet(DATASET_PATH)
df = df[df["condition"].isin(["EO", "EC"])].copy()
df["age"] = df["age"].astype(str)


def map_age_group(age: str) -> str | None:
    if age in YOUNG_AGES:
        return "young"
    if age in OLDER_AGES:
        return "older"
    return None


df["age_group"] = df["age"].map(map_age_group)
df = df[df["age_group"].notna()].copy()
df["y"] = df["age_group"].map({"young": 0, "older": 1}).astype(int)

print(df.drop_duplicates("subject_id")["age_group"].value_counts())

FEATURE_COLS = [
    c
    for c in df.columns
    if c.startswith(NODE_PREFIXES) or c in GLOBAL_FEATURES
]
FEATURE_COLS = (
    df[FEATURE_COLS]
    .select_dtypes(include="number")
    .dropna(axis=1, how="all")
    .columns.tolist()
)
print(f"N features de rede: {len(FEATURE_COLS)}")


def build_feature_matrix(frame: pd.DataFrame, condition: str) -> tuple[pd.DataFrame, pd.Series]:
    sub = frame[frame["condition"] == condition].copy()
    sub = sub.drop_duplicates("subject_id")
    X = sub[FEATURE_COLS].copy()
    y = sub["y"].copy()
    X.index = sub["subject_id"].values
    y.index = sub["subject_id"].values
    return X, y


def build_diff_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    eo = frame[frame["condition"] == "EO"].drop_duplicates("subject_id").set_index("subject_id")
    ec = frame[frame["condition"] == "EC"].drop_duplicates("subject_id").set_index("subject_id")
    common = eo.index.intersection(ec.index)
    X = ec.loc[common, FEATURE_COLS] - eo.loc[common, FEATURE_COLS]
    y = eo.loc[common, "y"]
    return X, y


def get_models() -> dict:
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogReg_L2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "SVM_RBF": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "KNN_PCA": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)),
                ("clf", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
            ]
        ),
    }


def evaluate_all(X: pd.DataFrame, y: pd.Series, models: dict, task: str) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
    }
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        rows.append(
            {
                "task": task,
                "model": name,
                "n": len(y),
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "accuracy_std": float(np.std(scores["test_accuracy"])),
                "balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
                "balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
                "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "roc_auc_std": float(np.std(scores["test_roc_auc"])),
            }
        )
        print(
            f"[{task}] {name}: "
            f"acc={rows[-1]['accuracy_mean']:.3f}±{rows[-1]['accuracy_std']:.3f} | "
            f"bal={rows[-1]['balanced_accuracy_mean']:.3f}±{rows[-1]['balanced_accuracy_std']:.3f} | "
            f"auc={rows[-1]['roc_auc_mean']:.3f}±{rows[-1]['roc_auc_std']:.3f}"
        )
    return pd.DataFrame(rows)


# ==========================
# TASKS
# ==========================
models = get_models()
tasks = {}

X_eo, y_eo = build_feature_matrix(df, "EO")
X_ec, y_ec = build_feature_matrix(df, "EC")
X_diff, y_diff = build_diff_matrix(df)
tasks["EO"] = (X_eo, y_eo)
tasks["EC"] = (X_ec, y_ec)
tasks["Diff_EC_minus_EO"] = (X_diff, y_diff)

for task, (X, y) in tasks.items():
    print(f"\n=== {task}: X={X.shape}, y={y.value_counts().to_dict()} ===")

results = []
for task, (X, y) in tasks.items():
    results.append(evaluate_all(X, y, models, task))

df_results = pd.concat(results, ignore_index=True)
print("\n=== Resumo ===")
print(df_results.to_string(index=False))
