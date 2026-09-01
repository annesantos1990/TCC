"""Estimativa de threshold por surrogate (Rosário et al., 2015)."""

import numpy as np

from .sync import motif_sync_pair


def estimate_threshold(
    motifs: np.ndarray,
    window_size: int,
    max_lag: int,
    n_surrogates: int = 500,
    percentile: float = 95.0,
    random_state: int | None = 42,
) -> dict:
    """
    Estima o threshold de motif-synchronization via aleatorização.

    Para cada surrogate, sorteia um par de canais e uma janela temporal,
    embaralha os motifs de um canal (destrói sincronização temporal) e
    calcula a sincronização nula. O threshold é o percentil escolhido
    dessa distribuição (padrão: 95%, α = 0,05).

    Parameters
    ----------
    motifs : np.ndarray
        Array (n_channels, n_time) com motifs inteiros.
    window_size : int
        Tamanho da janela em amostras.
    max_lag : int
        Lag máximo em amostras.
    n_surrogates : int
        Número de surrogates gerados.
    percentile : float
        Percentil da distribuição nula (ex.: 95 → p ≤ 0,05).
    random_state : int, optional
        Semente para reprodutibilidade.

    Returns
    -------
    dict
        threshold, percentile, null_mean, null_std, n_surrogates, null_syncs
    """
    rng = np.random.default_rng(random_state)
    n_channels, n_time = motifs.shape
    window_eff = window_size + max_lag
    n_windows = n_time - window_eff

    if n_windows <= 0:
        raise ValueError("Sinal curto demais para janela + lag.")

    null_syncs = np.empty(n_surrogates, dtype=np.float64)

    for k in range(n_surrogates):
        i = int(rng.integers(0, n_channels))
        j = int(rng.integers(0, n_channels))
        while j == i and n_channels > 1:
            j = int(rng.integers(0, n_channels))

        t = int(rng.integers(0, n_windows))
        seg_i = motifs[i, t : t + window_size]
        seg_j = motifs[j, t : t + window_size].copy()
        rng.shuffle(seg_j)

        null_syncs[k] = motif_sync_pair(seg_i, seg_j, max_lag)

    return {
        "threshold": float(np.percentile(null_syncs, percentile)),
        "percentile": percentile,
        "null_mean": float(np.mean(null_syncs)),
        "null_std": float(np.std(null_syncs)),
        "n_surrogates": n_surrogates,
        "null_syncs": null_syncs,
    }
