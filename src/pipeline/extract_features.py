from pathlib import Path
import pandas as pd
import mne
from tqdm import tqdm

from src.graph.motifs import trans_motifs
from src.graph.threshold import estimate_threshold
from src.graph.tvg import build_tvg_parallel
from src.graph.aggregate import (
    aggregate_static_network,
    edges_per_time,
    hub_occurrence,
    weighted_degree_static,
    temporal_statistics
)
from src.graph.export_networks import save_network_exports
from src.graph.network import graph_metrics


# ==========================
# PATHS
# ==========================
PROJECT_ROOT = Path.cwd()
SOURCE_DIR = PROJECT_ROOT / "data" / "preprocessed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "intermediate"
NETWORKS_DIR = OUTPUT_DIR / "networks"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NETWORKS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# PARAMETERS
# ==========================
WINDOW_MS = 100
MAX_LAG_MS = 25
N_SEC = 5
N_SURROGATES = 200
THRESHOLD_PERCENTILE = 95.0


def extract_features(file: Path):
    """Extrai features de um único arquivo EEG e exporta redes estilo Rosário."""

    subject_id, condition = file.stem.split("_")
    stem = f"{subject_id}_{condition}"

    out_file = OUTPUT_DIR / f"{stem}.parquet"
    nodes_csv = NETWORKS_DIR / f"{stem}_nodes.csv"
    temporal_csv = NETWORKS_DIR / f"{stem}_temporal.csv"
    edges_csv = NETWORKS_DIR / f"{stem}_edges.csv"

    if (
        out_file.exists()
        and nodes_csv.exists()
        and temporal_csv.exists()
        and edges_csv.exists()
    ):
        print(f" Pulando {stem} (já existe)")
        return

    raw = mne.io.read_raw_eeglab(file, preload=True, verbose=True)
    raw.pick(eeg=True)
    ch_names = list(raw.ch_names)

    eeg = raw.get_data()
    sfreq = raw.info["sfreq"]

    if N_SEC is not None:
        eeg = eeg[:, :int(N_SEC * sfreq)]

    window_size = int(WINDOW_MS / 1000 * sfreq)
    max_lag = int(MAX_LAG_MS / 1000 * sfreq)

    motifs = trans_motifs(eeg)

    threshold_info = estimate_threshold(
        motifs,
        window_size,
        max_lag,
        n_surrogates=N_SURROGATES,
        percentile=THRESHOLD_PERCENTILE,
    )
    threshold = threshold_info["threshold"]
    print(
        f" Threshold estimado (p{THRESHOLD_PERCENTILE:.0f}): "
        f"{threshold:.4f} "
        f"(nulo: μ={threshold_info['null_mean']:.4f}, "
        f"σ={threshold_info['null_std']:.4f})"
    )

    tvg = build_tvg_parallel(
        motifs=motifs,
        window_size=window_size,
        max_lag=max_lag,
        threshold=threshold,
    )

    # ==========================
    # FEATURES
    # ==========================
    print("Calculando métricas de rede...")
    rea = aggregate_static_network(tvg)
    edges_t = edges_per_time(tvg)
    hub_freq = hub_occurrence(tvg)
    wdeg_static = weighted_degree_static(tvg)
    edges_stats = temporal_statistics(edges_t)

    net = graph_metrics(rea)

    features = {}

    for i, v in tqdm(enumerate(net["degree"])):
        features[f"degree_{i}"] = v

    for i, v in tqdm(enumerate(net["clustering"])):
        features[f"clustering_{i}"] = v

    for i, v in tqdm(enumerate(net["betweenness"])):
        features[f"betweenness_{i}"] = v

    for i, v in tqdm(enumerate(net["local_efficiency"])):
        features[f"local_efficiency_{i}"] = v

    for i, v in tqdm(enumerate(hub_freq)):
        features[f"hub_frequency_{i}"] = v

    for i, v in tqdm(enumerate(wdeg_static)):
        features[f"weighted_degree_{i}"] = v

    features["global_efficiency"] = net["global_efficiency"]
    features["sync_threshold"] = threshold
    features["sync_threshold_null_mean"] = threshold_info["null_mean"]
    features["sync_threshold_null_std"] = threshold_info["null_std"]
    features["edges_mean"] = edges_stats["mean"]
    features["edges_std"] = edges_stats["std"]
    features["edges_cv"] = edges_stats["cv"]

    df = pd.DataFrame(features, index=[0])
    df["subject_id"] = subject_id
    df["condition"] = condition

    df.to_parquet(out_file, index=False)
    print(f" Salvo: {out_file.name}")

    paths = save_network_exports(
        tvg=tvg,
        rea=rea,
        ch_names=ch_names,
        networks_dir=NETWORKS_DIR,
        stem=stem,
    )
    print(
        f" Redes: {paths['nodes'].name}, "
        f"{paths['temporal'].name}, {paths['edges'].name}"
    )

files = list(SOURCE_DIR.glob("sub-*.set"))
len(files), files[:3]

for file in files:
    extract_features(file)
