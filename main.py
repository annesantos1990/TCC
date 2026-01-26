"""
EEG Functional Connectivity Analysis using Motif-Synchronization
and Time-Varying Graphs (TVG)

Pipeline:
EEG -> Motifs -> TVG -> Temporal Aggregation -> Network Metrics -> ML Dataset
"""

# ==========================
# IMPORTS
# ==========================

import numpy as np
import pandas as pd
import mne

from src.motifs import trans_motifs
from src.tvg import build_tvg
from src.aggregate import (
    aggregate_static_network,
    edges_per_time,
    hub_occurrence,
    weighted_degree_static,
    temporal_statistics
)
from src.network import graph_metrics


# ==========================
# LOAD EEG DATA
# ==========================

# Ajuste o caminho para o arquivo EEG
eeg_file = "data/example_subject_raw.fif"
subject_id = "sub_001"

raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
raw.pick_types(eeg=True)

eeg_data = raw.get_data()          # (n_channels, n_samples)
sfreq = raw.info["sfreq"]          # frequência amostral

print(f"EEG shape: {eeg_data.shape}")
print(f"Sampling frequency: {sfreq} Hz")


# ==========================
# PARAMETERS
# ==========================

# Parâmetros fisiológicos
window_ms = 200      # tamanho da janela (ms)
max_lag_ms = 50      # atraso máximo (ms)

# Conversão para amostras
window_size = int(window_ms / 1000 * sfreq)
max_lag = int(max_lag_ms / 1000 * sfreq)

print(f"Window size: {window_size} samples")
print(f"Max lag: {max_lag} samples")


# ==========================
# MOTIF TRANSFORMATION
# ==========================

print("Transforming EEG into motifs...")
motifs = trans_motifs(eeg_data)

print(f"Motifs shape: {motifs.shape}")


# ==========================
# BUILD TVG
# ==========================

print("Building Time-Varying Graph (TVG)...")

tvg = build_tvg(
    motifs=motifs,
    window_size=window_size,
    max_lag=max_lag,
    threshold=None   # sem threshold (ML-friendly)
)

print(f"TVG shape: {tvg.shape}")
# (n_channels, n_channels, n_time_windows)


# ==========================
# TEMPORAL AGGREGATIONS
# ==========================

print("Aggregating temporal information...")

# Rede estática agregada
rea = aggregate_static_network(tvg)

# Número de arestas por instante
edges_t = edges_per_time(tvg)

# Frequência de hubs
hub_freq = hub_occurrence(tvg)

# Grau ponderado na rede agregada
wdeg_static = weighted_degree_static(tvg)

# Estatísticas temporais das arestas
edges_stats = temporal_statistics(edges_t)


# ==========================
# NETWORK METRICS
# ==========================

print("Computing network metrics...")

net_metrics = graph_metrics(rea)

degree = net_metrics["degree"]
clustering = net_metrics["clustering"]
betweenness = net_metrics["betweenness"]
local_eff = net_metrics["local_efficiency"]
global_eff = net_metrics["global_efficiency"]


# ==========================
# FEATURE VECTOR (ML)
# ==========================

print("Building feature vector...")

features = {}

# --- Node-level features ---
for i, v in enumerate(degree):
    features[f"degree_{i}"] = v

for i, v in enumerate(clustering):
    features[f"clustering_{i}"] = v

for i, v in enumerate(betweenness):
    features[f"betweenness_{i}"] = v

for i, v in enumerate(local_eff):
    features[f"local_efficiency_{i}"] = v

for i, v in enumerate(hub_freq):
    features[f"hub_frequency_{i}"] = v

for i, v in enumerate(wdeg_static):
    features[f"weighted_degree_{i}"] = v


# --- Global features ---
features["global_efficiency"] = global_eff
features["edges_mean"] = edges_stats["mean"]
features["edges_std"] = edges_stats["std"]
features["edges_cv"] = edges_stats["cv"]


# ==========================
# DATAFRAME FOR ML
# ==========================

df_features = pd.DataFrame(features, index=[subject_id])

print(df_features.head())

# Salvar para uso posterior (PCA / ML)
df_features.to_csv(f"data/features_{subject_id}.csv")

print("Pipeline finished successfully.")
