import numpy as np
from .sync import motif_sync_pair

def build_tvg(motifs: np.ndarray,
              window_size: int,
              max_lag: int,
              threshold: float | None = None) -> np.ndarray:
    """
    Constrói o Time-Varying Graph (TVG) a partir de motifs,
    utilizando uma janela deslizante no tempo.

    Cada fatia temporal do TVG representa a conectividade
    funcional estimada dentro de uma janela temporal de
    comprimento (window_size + max_lag).

    Parameters
    ----------
    motifs : np.ndarray
        Array (n_channels, n_timepoints)
    window_size : int
        Tamanho da janela deslizante (em amostras)
    max_lag : int
        Atraso máximo permitido (em amostras)
    threshold : float, optional
        Limiar mínimo de sincronização. Se None, não aplica limiar.

    Returns
    -------
    tvg : np.ndarray
        Tensor (n_channels, n_channels, n_windows)
    """

    n_channels, n_time = motifs.shape
    window_eff = window_size + max_lag
    n_windows = n_time - window_eff

    tvg = np.zeros((n_channels, n_channels, n_windows), dtype=float)

    for t in range(n_windows):
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                sync = motif_sync_pair(
                    motifs[i, t:t + window_size],
                    motifs[j, t:t + window_size],
                    max_lag
                )

                if threshold is not None:
                    sync = float(sync >= threshold)

                tvg[i, j, t] = tvg[j, i, t] = sync

    return tvg
