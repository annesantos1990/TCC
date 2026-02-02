import numpy as np
from joblib import Parallel, delayed
from .sync import motif_sync_pair
from tqdm import tqdm

def compute_window(t, motifs, n_channels, window_size, max_lag, threshold):
    """Função auxiliar para processar uma única janela de tempo"""
    window_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            sync = motif_sync_pair(
                motifs[i, t:t + window_size],
                motifs[j, t:t + window_size],
                max_lag
            )
            if threshold is not None:
                sync = float(sync >= threshold)
            
            window_matrix[i, j] = window_matrix[j, i] = sync
    return window_matrix

def build_tvg_parallel(motifs: np.ndarray,
                      window_size: int,
                      max_lag: int,
                      threshold: float | None = None,
                      n_jobs: int = 4) -> np.ndarray:
    
    n_channels, n_time = motifs.shape
    window_eff = window_size + max_lag
    n_windows = n_time - window_eff

    # Paralelização aqui: divide o loop de n_windows pelos núcleos
    tasks = []

    for t in tqdm(
        range(n_windows),
        desc="Processando TVG",
        leave=False):
        task = delayed(compute_window)(
            t,
            motifs,
            n_channels,
            window_size,
            max_lag,
            threshold
        )
        tasks.append(task)

    results = Parallel(n_jobs=n_jobs)(tasks)

    # Converte a lista de matrizes de volta para o formato (chan, chan, tempo)
    tvg = np.stack(results, axis=-1)
    return tvg