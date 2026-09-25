from pathlib import Path
import pandas as pd


# ==========================
# PATHS
# ==========================
PROJECT_ROOT = Path.cwd()
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "participants.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataset_features.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# LOAD DATA
# ==========================
# Preferir features alinhadas por eletrodo (degree_Oz, ...), geradas por:
#   python -m src.pipeline.realign_features_by_electrode
# que grava data/intermediate_aligned e regenera dataset_features.parquet.
print("Lendo features intermediárias...")
ALIGNED_DIR = PROJECT_ROOT / "data" / "intermediate_aligned"
files = list(ALIGNED_DIR.glob("sub-*.parquet"))
if not files:
    files = list(INTERMEDIATE_DIR.glob("sub-*.parquet"))
    print(
        "AVISO: intermediate_aligned vazio — usando data/intermediate "
        "(features podem estar indexadas por posição de canal)."
    )
else:
    print(f"Usando intermediate_aligned ({len(files)} arquivos)")
print(f"Encontrados {len(files)} arquivos")

dfs = []

for file in files:
    subject_id, condition = file.stem.split("_")

    df = pd.read_parquet(file)
    df["subject_id"] = subject_id
    df["condition"] = condition

    dfs.append(df)

df_features = pd.concat(dfs, ignore_index=True)

print("Lendo metadados...")
meta = pd.read_csv(METADATA_FILE)
meta["subject_id"] = meta["ID"]

df_final = df_features.merge(
    meta,
    on="subject_id",
    how="left",
    validate="m:1"
)

df_final = df_final.rename(columns={
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
print(f"Shape inicial: {df_final.shape}")

# VErificando ocorrencias unicas de idade, genero, handedness, education, drug, smoking, skid_diagnoses, skid_diagnoses_1, skid_diagnoses_2, comments_skid_assessment, hamilton_scale, bsl23_sumscore, bsl23_behavior, audit, standard_alcoholunits_last_28days, alcohol_dependence_in_1st_3rd_degree_relative, relationship_status

# print(df_final["age"].value_counts())
# print(df_final["gender"].value_counts())
# print(df_final["handedness"].value_counts())
# print(df_final["education"].value_counts())
# print(df_final["drug"].value_counts())
# print(df_final["smoking"].value_counts())
# print(df_final["skid_diagnoses"].value_counts())
# print(df_final["skid_diagnoses_1"].value_counts())
# print(df_final["skid_diagnoses_2"].value_counts())
# print(df_final["comments_skid_assessment"].value_counts())
# print(df_final["hamilton_scale"].value_counts())
# print(df_final["bsl23_sumscore"].value_counts())
# print(df_final["bsl23_behavior"].value_counts())
# print(df_final["audit"].value_counts())
# print(df_final["standard_alcoholunits_last_28days"].value_counts())
# print(df_final["alcohol_dependence_in_1st_3rd_degree_relative"].value_counts())
# print(df_final["relationship_status"].value_counts())


# padronizar strings
df_final["relationship_status"] = df_final["relationship_status"].str.lower()
df_final["handedness"] = df_final["handedness"].str.lower()
df_final["education"] = df_final["education"].str.lower()

healthy_mask = (
    (df_final["drug"] == 0) &
    (df_final["smoking"] == 1) &
    (df_final["audit"] < 8) &
    (df_final["hamilton_scale"] <= 7) &
    (df_final["bsl23_behavior"] == 0) &
    (df_final["skid_diagnoses"].str.lower() == "none") &
    (df_final["skid_diagnoses_2"].isna()) &
    (df_final["comments_skid_assessment"].isna())
)

df_healthy = df_final.loc[healthy_mask].copy()
print(f"Shape após filtro saudável: {df_healthy.shape}")

print(f"N indivíduos antes do filtro: {df_final['subject_id'].nunique()}")
print(f"N indivíduos após filtro saudável: {df_healthy['subject_id'].nunique()}")

META_COLS = [
    "age",
    "gender",
    "handedness",
    "education",
    "relationship_status"
]

df_meta = df_healthy[META_COLS].copy()
print(f"Shape após seleção de metadados: {df_meta.shape}")

df_meta = pd.get_dummies(
    df_meta,
    columns=META_COLS,
    drop_first=True
)

df_meta

df_analysis = pd.concat(
    [df_healthy, df_meta],
    axis=1
)
# padronizar condição para EO
df_analysis["condition"] = df_analysis["condition"].replace({"E0": "EO"})

print(f"Shape após concatenação: {df_analysis.shape}")

df_analysis.to_parquet(OUTPUT_FILE, index=False)

print(f"Dataset final salvo em: {OUTPUT_FILE}")
print(f"Shape final: {df_analysis.shape}")

