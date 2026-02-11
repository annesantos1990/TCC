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

df_dataset = pd.read_parquet(DATASET_PATH)

df_dataset = df_dataset[df_dataset["condition"].isin(["EO", "EC"])]

df_dataset["y"] = df_dataset["condition"].map({"EO": 0, "EC": 1})

NODE_FEATURES = [
    c for c in df_dataset.columns
    if c.startswith(("degree_", "clustering_", "betweenness_", "hub_"))
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
X = df_dataset[NODE_FEATURES].select_dtypes(include="number").dropna(axis=1)
y = df_dataset["y"]

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
    [X, df_dataset[META_FEATURES]],
    axis=1
).dropna()

y_meta = df_dataset.loc[X_meta.index, "y"]

pipe_pca_meta = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("clf", LogisticRegression(max_iter=1000))
])

evaluate_model(X_meta, y_meta, pipe_pca_meta)