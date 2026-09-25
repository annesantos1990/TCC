"""
Robustez e interpretabilidade — EO + SVM-RBF (análise principal do TCC).

Blocos:
  1. Predições OOF (decision_function): ROC, confusão, sens/spec
  2. RepeatedStratifiedKFold — estabilidade via médias das 20 repetições
  3. Confundidores: gender + education (grafia padronizada)
  4. Importância por famílias (permute só no fold de teste)
  5. Descrição exploratória + eletrodo/região a partir do nome da feature
     (dataset realinhado: betweenness_Oz, degree_Cz, ...)

Espelho: notebook/03_classification/09_ml_age_robustness_interpretability.ipynb
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ==========================
# PATHS / PARAMS
# ==========================
DATASET_PATH = Path("data/processed/dataset_features.parquet")
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 20
N_PERM_IMPORTANCE = 100
TOP_K_FEATURES = 8
META_COLS = ["gender", "education"]

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

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "degree": ("degree_",),
    "clustering": ("clustering_",),
    "betweenness": ("betweenness_",),
    "local_efficiency": ("local_efficiency_",),
    "hub_frequency": ("hub_frequency_",),
    "weighted_degree": ("weighted_degree_",),
    "global": tuple(GLOBAL_FEATURES),
}

# Grafias conhecidas no LEMON / dataset
EDUCATION_FIX = {
    "gymansium": "gymnasium",
}


def map_age_group(age: str) -> str | None:
    if age in YOUNG_AGES:
        return "young"
    if age in OLDER_AGES:
        return "older"
    return None


def make_svm() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def normalize_education(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.replace(EDUCATION_FIX)


def electrode_region(name: str) -> str:
    """Região grosseira 10–20 a partir do prefixo do eletrodo."""
    n = name.upper()
    for prefix, region in (
        ("FP", "frontal_polar"),
        ("AF", "anterior_frontal"),
        ("FT", "fronto_temporal"),
        ("FC", "fronto_central"),
        ("TP", "temporo_parietal"),
        ("CP", "centro_parietal"),
        ("PO", "parieto_occipital"),
        ("F", "frontal"),
        ("C", "central"),
        ("T", "temporal"),
        ("P", "parietal"),
        ("O", "occipital"),
    ):
        if n.startswith(prefix):
            return region
    return "other"


def parse_electrode_from_feature(feat: str) -> str | None:
    """Extrai eletrodo de nomes alinhados (ex.: betweenness_Oz → Oz)."""
    for prefix in NODE_PREFIXES:
        if feat.startswith(prefix):
            elect = feat[len(prefix) :]
            if elect and not elect.isdigit():
                return elect
    return None


def feature_electrode_table(features: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        electrode = parse_electrode_from_feature(feat)
        rows.append(
            {
                "feature": feat,
                "electrode": electrode,
                "region": electrode_region(electrode) if electrode else None,
            }
        )
    return pd.DataFrame(rows)


def load_eo_frame(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    eo = df[df["condition"] == "EO"].copy()
    duplicated = eo.duplicated("subject_id", keep=False)
    assert not duplicated.any(), (
        "Foram encontrados subject_id duplicados na condição EO: "
        f"{eo.loc[duplicated, 'subject_id'].tolist()}"
    )
    eo["age"] = eo["age"].astype(str)
    eo["age_group"] = eo["age"].map(map_age_group)
    eo = eo[eo["age_group"].notna()].copy()
    eo["y"] = eo["age_group"].map({"young": 0, "older": 1}).astype(int)
    if "education" in eo.columns:
        eo["education"] = normalize_education(eo["education"])
    return eo


def network_feature_cols(frame: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in frame.columns
        if c.startswith(NODE_PREFIXES) or c in GLOBAL_FEATURES
    ]
    return (
        frame[cols]
        .select_dtypes(include="number")
        .dropna(axis=1, how="all")
        .columns.tolist()
    )


def encode_meta(frame: pd.DataFrame, cols: list[str] = META_COLS) -> pd.DataFrame:
    parts = []
    for c in cols:
        if c not in frame.columns:
            continue
        s = frame[c]
        if c == "education":
            s = normalize_education(s)
        if pd.api.types.is_numeric_dtype(s):
            parts.append(s.astype(float).to_frame())
        else:
            dummies = pd.get_dummies(s.astype(str), prefix=c, drop_first=False)
            parts.append(dummies.astype(float))
    if not parts:
        raise ValueError("Nenhuma coluna de metadados disponível.")
    out = pd.concat(parts, axis=1)
    out.index = frame.index
    return out


def family_columns(feature_cols: list[str], family: str) -> list[str]:
    keys = FEATURE_FAMILIES[family]
    if family == "global":
        return [c for c in feature_cols if c in keys]
    return [c for c in feature_cols if c.startswith(keys)]


# ==========================
# 1. OOF metrics (decision_function)
# ==========================
def oof_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    model: Pipeline | None = None,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> dict:
    model = make_svm() if model is None else clone(model)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(model, X, y, cv=cv)
    y_score = cross_val_predict(model, X, y, cv=cv, method="decision_function")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    fpr, tpr, thresholds = roc_curve(y, y_score)

    return {
        "y_true": y,
        "y_pred": y_pred,
        "y_score": y_score,
        "confusion_matrix": cm,
        "sensitivity_older": float(sens),
        "specificity_young": float(spec),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_score)),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


# ==========================
# 2. Repeated CV
# ==========================
def repeated_roc_auc(
    X: pd.DataFrame,
    y: pd.Series,
    model: Pipeline | None = None,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray | float]:
    """Retorna scores por fold e médias por repetição (unidade de estabilidade)."""
    model = make_svm() if model is None else clone(model)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    fold_scores = np.asarray(
        cross_validate(model, X, y, cv=cv, scoring="roc_auc")["test_score"],
        dtype=float,
    )
    # ordem do sklearn: repeat0_fold0..4, repeat1_fold0..4, ...
    rep_means = fold_scores.reshape(n_repeats, n_splits).mean(axis=1)
    return {
        "fold_scores": fold_scores,
        "repeat_means": rep_means,
        "mean": float(rep_means.mean()),
        "std_between_repeats": float(rep_means.std()),
        "median": float(np.median(rep_means)),
        "min": float(rep_means.min()),
        "max": float(rep_means.max()),
    }


# ==========================
# 3. Confounder models
# ==========================
def compare_feature_blocks(
    X_net: pd.DataFrame,
    X_meta: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {"roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy"}
    blocks = {
        "network_only": X_net,
        "meta_only": X_meta,
        "network_plus_meta": pd.concat([X_net, X_meta], axis=1),
    }
    rows = []
    for name, Xb in blocks.items():
        scores = cross_validate(make_svm(), Xb, y, cv=cv, scoring=scoring)
        rows.append(
            {
                "block": name,
                "n_features": Xb.shape[1],
                "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
                "roc_auc_std": float(np.std(scores["test_roc_auc"])),
                "balanced_accuracy_mean": float(
                    np.mean(scores["test_balanced_accuracy"])
                ),
                "balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
            }
        )
    return pd.DataFrame(rows)


# ==========================
# 4. Permutation importance (só no fold de teste)
# ==========================
def _auc_decision(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    return float(roc_auc_score(y, model.decision_function(X)))


def grouped_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    n_repeats: int = N_PERM_IMPORTANCE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Permutation importance convencional por família:
    treina no fold de treino; embaralha a família só no teste; mede Δ AUC.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    # pré-treina um modelo por fold
    fitted = []
    baselines = []
    test_slices = []
    for train_idx, test_idx in cv.split(X, y):
        est = make_svm().fit(X.iloc[train_idx], y.iloc[train_idx])
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        fitted.append(est)
        baselines.append(_auc_decision(est, X_te, y_te))
        test_slices.append((X_te, y_te))

    baseline_mean = float(np.mean(baselines))
    rows = []
    for family in FEATURE_FAMILIES:
        cols = family_columns(feature_cols, family)
        if not cols:
            continue
        drops: list[float] = []
        for est, base, (X_te, y_te) in zip(fitted, baselines, test_slices):
            for _ in range(n_repeats):
                Xp = X_te.copy()
                order = rng.permutation(len(Xp))
                Xp.loc[:, cols] = Xp.iloc[order][cols].to_numpy()
                drops.append(base - _auc_decision(est, Xp, y_te))
        drops_arr = np.asarray(drops, dtype=float)
        rows.append(
            {
                "family": family,
                "n_features": len(cols),
                "importance_mean": float(np.mean(drops_arr)),
                "importance_std": float(np.std(drops_arr)),
                "baseline_roc_auc": baseline_mean,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def column_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    cols: list[str],
    n_repeats: int = N_PERM_IMPORTANCE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Importância por coluna (permute só no teste) — exploratória."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    fitted = []
    baselines = []
    test_slices = []
    for train_idx, test_idx in cv.split(X, y):
        est = make_svm().fit(X.iloc[train_idx], y.iloc[train_idx])
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        fitted.append(est)
        baselines.append(_auc_decision(est, X_te, y_te))
        test_slices.append((X_te, y_te))

    rows = []
    for col in cols:
        drops: list[float] = []
        for est, base, (X_te, y_te) in zip(fitted, baselines, test_slices):
            for _ in range(n_repeats):
                Xp = X_te.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                drops.append(base - _auc_decision(est, Xp, y_te))
        drops_arr = np.asarray(drops, dtype=float)
        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(drops_arr)),
                "importance_std": float(np.std(drops_arr)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


# ==========================
# 5. Descriptive stats
# ==========================
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        means[i] = np.mean(rng.choice(values, size=len(values), replace=True))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(values)), float(lo), float(hi)


def describe_features_by_group(
    frame: pd.DataFrame,
    features: list[str],
    group_col: str = "age_group",
) -> pd.DataFrame:
    rows = []
    for feat in features:
        young = frame.loc[frame[group_col] == "young", feat].to_numpy(dtype=float)
        older = frame.loc[frame[group_col] == "older", feat].to_numpy(dtype=float)
        y_mean, y_lo, y_hi = bootstrap_mean_ci(young)
        o_mean, o_lo, o_hi = bootstrap_mean_ci(older)
        rows.append(
            {
                "feature": feat,
                "young_mean": y_mean,
                "young_median": float(np.nanmedian(young)),
                "young_ci95": (y_lo, y_hi),
                "older_mean": o_mean,
                "older_median": float(np.nanmedian(older)),
                "older_ci95": (o_lo, o_hi),
                "cohens_d_young_minus_older": cohens_d(young, older),
            }
        )
    return pd.DataFrame(rows)


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    eo = load_eo_frame()
    feat_cols = network_feature_cols(eo)
    X_net = eo[feat_cols].copy()
    X_net.index = eo["subject_id"].values
    y = eo["y"].copy()
    y.index = eo["subject_id"].values
    X_meta = encode_meta(eo.set_index("subject_id"))

    print(f"EO n={len(y)}, network={X_net.shape[1]}, meta={X_meta.shape[1]}")
    print("gender × age_group:")
    print(pd.crosstab(eo["age_group"], eo["gender"], normalize="index").round(3))
    print("education × age_group:")
    print(pd.crosstab(eo["age_group"], eo["education"], normalize="index").round(3))

    oof = oof_evaluation(X_net, y)
    print(
        f"\nOOF(decision_function): AUC={oof['roc_auc']:.3f} | "
        f"bal={oof['balanced_accuracy']:.3f} | "
        f"sens={oof['sensitivity_older']:.3f} | spec={oof['specificity_young']:.3f}"
    )
    print(oof["confusion_matrix"])

    rep = repeated_roc_auc(X_net, y)
    print(
        f"\nRepeated CV (20 médias): "
        f"{rep['mean']:.3f} ± {rep['std_between_repeats']:.3f} "
        f"[{rep['min']:.3f}, {rep['max']:.3f}]"
    )

    print("\n=== Blocos ===")
    print(compare_feature_blocks(X_net, X_meta, y).to_string(index=False))

    fam = grouped_permutation_importance(X_net, y, feat_cols, n_repeats=10)
    print("\n=== Famílias (n_repeats=10 smoke) ===")
    print(fam.to_string(index=False))

    top_family = fam.iloc[0]["family"]
    top_cols = family_columns(feat_cols, top_family)
    col_imp = column_permutation_importance(X_net, y, top_cols, n_repeats=5)
    top = col_imp.head(TOP_K_FEATURES)
    print(f"\n=== Top cols ({top_family}) ===")
    print(top.to_string(index=False))
    print(feature_electrode_table(top["feature"].tolist()).to_string(index=False))
