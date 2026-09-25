"""
Realinha features de nó por nome de eletrodo e regenera dataset_features.parquet.

Problema: degree_i / betweenness_i usavam índice da ordem local de canais,
que varia entre sujeitos (56–61 canais). Isso torna a coluna i semanticamente
incomparável entre participantes.

Solução:
  1. Remapeia degree_i → degree_{Elect} via *_nodes.csv
  2. Mantém colunas de eletrodos com cobertura >= COVERAGE_MIN (padrão 95%)
     em EO e EC (NaN se ausente; Imputer do modelo trata)
  3. Regenera data/processed/dataset_features.parquet (backup do anterior)

Uso (na raiz do TCC):
  python -m src.pipeline.realign_features_by_electrode
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path.cwd()
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
NETWORKS_DIR = INTERMEDIATE_DIR / "networks"
ALIGNED_DIR = PROJECT_ROOT / "data" / "intermediate_aligned"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "participants.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataset_features.parquet"
BACKUP_FILE = OUTPUT_DIR / "dataset_features_indexed_backup.parquet"

NODE_METRIC_PREFIXES = (
    "degree_",
    "clustering_",
    "betweenness_",
    "local_efficiency_",
    "hub_frequency_",
    "weighted_degree_",
)

# Eletrodo presente em pelo menos esta fração dos arquivos EO e dos EC
COVERAGE_MIN = 0.95


def _node_metric_cols(columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        for prefix in NODE_METRIC_PREFIXES:
            if c.startswith(prefix):
                suffix = c[len(prefix) :]
                if suffix.isdigit():
                    out.append(c)
                break
    return out


def remap_row(features: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha de features indexadas → features nomeadas por eletrodo."""
    assert len(features) == 1
    elects = nodes["Elect"].astype(str).str.strip().tolist()
    if len(elects) != len(set(elects)):
        raise ValueError("Eletrodos duplicados no CSV de nós.")

    row = features.iloc[0].to_dict()
    new_row: dict = {}
    used_index_cols: set[str] = set()

    for prefix in NODE_METRIC_PREFIXES:
        for i, elect in enumerate(elects):
            old = f"{prefix}{i}"
            if old not in row:
                continue
            new_row[f"{prefix}{elect}"] = row[old]
            used_index_cols.add(old)

    # demais colunas (globais, meta locais do parquet intermediário)
    for k, v in row.items():
        if k in used_index_cols:
            continue
        if k in ("subject_id", "condition"):
            continue
        # descarta qualquer degree_999 órfão
        skip = False
        for prefix in NODE_METRIC_PREFIXES:
            if k.startswith(prefix) and k[len(prefix) :].isdigit():
                skip = True
                break
        if not skip:
            new_row[k] = v

    return pd.DataFrame([new_row])


def realign_all_intermediates() -> tuple[list[pd.DataFrame], list[str]]:
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(INTERMEDIATE_DIR.glob("sub-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum parquet em {INTERMEDIATE_DIR}")

    # 1ª passagem: remapeia e coleta cobertura por condição
    remapped: list[tuple[str, str, pd.DataFrame, set[str]]] = []
    elects_eo: list[set[str]] = []
    elects_ec: list[set[str]] = []

    for path in parquet_files:
        subject_id, condition = path.stem.split("_")
        nodes_path = NETWORKS_DIR / f"{path.stem}_nodes.csv"
        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes ausente para {path.stem}: {nodes_path}")

        feat = pd.read_parquet(path)
        nodes = pd.read_csv(nodes_path)
        aligned = remap_row(feat, nodes)
        aligned["subject_id"] = subject_id
        aligned["condition"] = condition.replace("E0", "EO")

        elect_set = {
            c.split("_", 1)[1]
            for c in aligned.columns
            if any(c.startswith(p) for p in NODE_METRIC_PREFIXES)
            and not c.split("_", 1)[1].isdigit()
        }
        # better: from nodes
        elect_set = set(nodes["Elect"].astype(str).str.strip())

        remapped.append((subject_id, aligned["condition"].iloc[0], aligned, elect_set))
        if aligned["condition"].iloc[0] == "EO":
            elects_eo.append(elect_set)
        else:
            elects_ec.append(elect_set)

    def coverage_keep(sets: list[set[str]], min_cov: float) -> set[str]:
        counts: Counter[str] = Counter()
        for s in sets:
            counts.update(s)
        n = len(sets)
        return {e for e, k in counts.items() if k / n >= min_cov}

    keep_eo = coverage_keep(elects_eo, COVERAGE_MIN)
    keep_ec = coverage_keep(elects_ec, COVERAGE_MIN)
    keep = sorted(keep_eo & keep_ec)

    strict_eo = set.intersection(*elects_eo) if elects_eo else set()
    strict_ec = set.intersection(*elects_ec) if elects_ec else set()
    strict = sorted(strict_eo & strict_ec)

    print(
        f"Arquivos remapeados: {len(remapped)} | "
        f"eletrodos @{COVERAGE_MIN:.0%} EO&EC: {len(keep)} | "
        f"intersecao 100% EO&EC: {len(strict)}"
    )
    print(f"Eletrodos mantidos ({len(keep)}): {keep}")

    # 2ª passagem: reindexa colunas de nó ao conjunto keep
    frames: list[pd.DataFrame] = []
    node_cols = [f"{p}{e}" for e in keep for p in NODE_METRIC_PREFIXES]

    for subject_id, condition, aligned, _ in remapped:
        out = aligned.copy()
        # garante todas as colunas de nó alinhadas
        for col in node_cols:
            if col not in out.columns:
                out[col] = pd.NA
        # remove eletrodos fora do keep
        drop_cols = []
        for c in out.columns:
            for prefix in NODE_METRIC_PREFIXES:
                if c.startswith(prefix):
                    elect = c[len(prefix) :]
                    if elect not in keep:
                        drop_cols.append(c)
                    break
        out = out.drop(columns=drop_cols, errors="ignore")
        # ordena: node_cols + outras
        other = [c for c in out.columns if c not in node_cols]
        out = out[node_cols + other]
        out_path = ALIGNED_DIR / f"{subject_id}_{condition}.parquet"
        out.to_parquet(out_path, index=False)
        frames.append(out)

    return frames, keep


def build_processed_dataset(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df_features = pd.concat(frames, ignore_index=True)
    meta = pd.read_csv(METADATA_FILE)
    meta["subject_id"] = meta["ID"]

    df_final = df_features.merge(meta, on="subject_id", how="left", validate="m:1")
    df_final = df_final.rename(
        columns={
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
            "Alcohol_Dependence_In_1st-3rd_Degree_relative": (
                "alcohol_dependence_in_1st_3rd_degree_relative"
            ),
            "Relationship_Status": "relationship_status",
        }
    )

    df_final["relationship_status"] = df_final["relationship_status"].str.lower()
    df_final["handedness"] = df_final["handedness"].str.lower()
    df_final["education"] = df_final["education"].str.lower()
    df_final["condition"] = df_final["condition"].replace({"E0": "EO"})

    healthy_mask = (
        (df_final["drug"] == 0)
        & (df_final["smoking"] == 1)
        & (df_final["audit"] < 8)
        & (df_final["hamilton_scale"] <= 7)
        & (df_final["bsl23_behavior"] == 0)
        & (df_final["skid_diagnoses"].str.lower() == "none")
        & (df_final["skid_diagnoses_2"].isna())
        & (df_final["comments_skid_assessment"].isna())
    )
    df_healthy = df_final.loc[healthy_mask].copy()
    print(
        f"Saudáveis: {df_healthy['subject_id'].nunique()} sujeitos / "
        f"{len(df_healthy)} linhas (antes: {df_final['subject_id'].nunique()})"
    )

    meta_cols = [
        "age",
        "gender",
        "handedness",
        "education",
        "relationship_status",
    ]
    df_meta = pd.get_dummies(df_healthy[meta_cols].copy(), columns=meta_cols, drop_first=True)
    df_analysis = pd.concat([df_healthy, df_meta], axis=1)
    return df_analysis


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists() and not BACKUP_FILE.exists():
        shutil.copy2(OUTPUT_FILE, BACKUP_FILE)
        print(f"Backup: {BACKUP_FILE}")
    elif OUTPUT_FILE.exists():
        print(f"Backup já existia: {BACKUP_FILE}")

    frames, keep = realign_all_intermediates()
    df_analysis = build_processed_dataset(frames)
    df_analysis.to_parquet(OUTPUT_FILE, index=False)
    print(f"Salvo: {OUTPUT_FILE} shape={df_analysis.shape}")

    # sanity: colunas nomeadas
    named = [
        c
        for c in df_analysis.columns
        if c.startswith("betweenness_") and not c.split("_", 1)[1].isdigit()
    ]
    indexed = [
        c
        for c in df_analysis.columns
        if c.startswith("betweenness_") and c.split("_", 1)[1].isdigit()
    ]
    print(f"betweenness nomeadas: {len(named)} | indexadas remanescentes: {len(indexed)}")
    if indexed:
        raise SystemExit("Ainda há features indexadas — abortar.")
    print("OK")


if __name__ == "__main__":
    main()
