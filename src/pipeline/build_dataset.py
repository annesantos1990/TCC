from pathlib import Path
import pandas as pd


# ==========================
# PATHS
# ==========================
PROJECT_ROOT = Path.cwd()
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
METADATA_FILE = PROJECT_ROOT / "data" / "Metadados" / "participants.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataset_features.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# LOAD DATA
# ==========================
print("Lendo features intermediárias...")
files = list(INTERMEDIATE_DIR.glob("sub-*.parquet"))
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

df_final.to_parquet(OUTPUT_FILE, index=False)

print(f"Dataset final salvo em: {OUTPUT_FILE}")
print(f"Shape final: {df_final.shape}")

