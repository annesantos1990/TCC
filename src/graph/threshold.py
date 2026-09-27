"""Estimativa de threshold por aleatorização da série temporal (critério da tese)."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .motifs import trans_motifs
from .sync import motif_sync_pair


def _window_sync_values(
    t: int,
    motifs: np.ndarray,
    n_channels: int,
    window_size: int,
    max_lag: int,
) -> np.ndarray:
    """Sincronização contínua (sem limiar) para todos os pares em uma janela."""
    n_pairs = n_channels * (n_channels - 1) // 2
    vals = np.empty(n_pairs, dtype=np.float64)
    k = 0
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            vals[k] = motif_sync_pair(
                motifs[i, t : t + window_size],
                motifs[j, t : t + window_size],
                max_lag,
            )
            k += 1
    return vals


def estimate_threshold(
    eeg: np.ndarray,
    window_size: int,
    max_lag: int,
    percentile: float = 99.0,
    n_shuffles: int = 1,
    random_state: int | None = 42,
    n_jobs: int = 4,
) -> dict:
    """
    Estima o limiar de motif-synchronization pelo critério da tese.

    1. Embaralha independentemente os pontos da série temporal de cada eletrodo
       (potencial), destruindo a estrutura temporal.
    2. Converte as séries embaralhadas em motifs.
    3. Constrói, em cada janela, a sincronização entre todos os pares de canais
       (como na rede, porém sem binarizar).
    4. O limiar é o percentil escolhido dessa distribuição de arestas
       (padrão: 99% → aceita ~1% de sincronizações ao acaso).

    Parameters
    ----------
    eeg : np.ndarray
        Sinal contínuo (n_channels, n_samples).
    window_size, max_lag : int
        Parâmetros da janela em amostras (iguais aos do TVG).
    percentile : float
        Percentil da distribuição nula (tese: 99).
    n_shuffles : int
        Quantas aleatorizações completas das séries (tese: 1).
        Valores > 1 acumulam mais arestas nulas para estabilizar o percentil.
    random_state : int, optional
        Semente para reprodutibilidade.
    n_jobs : int
        Paralelismo entre janelas.

    Returns
    -------
    dict
        threshold, percentile, null_mean, null_std, n_shuffles, n_null_edges
    """
    if eeg.ndim != 2:
        raise ValueError("eeg deve ter shape (n_channels, n_samples).")

    rng = np.random.default_rng(random_state)
    n_channels, _ = eeg.shape
    chunks: list[np.ndarray] = []

    for _ in range(n_shuffles):
        eeg_shuf = eeg.copy()
        for ch in range(n_channels):
            rng.shuffle(eeg_shuf[ch])

        motifs = trans_motifs(eeg_shuf)
        n_time = motifs.shape[1]
        window_eff = window_size + max_lag
        n_windows = n_time - window_eff
        if n_windows <= 0:
            raise ValueError("Sinal curto demais para janela + lag.")

        tasks = [
            delayed(_window_sync_values)(
                t, motifs, n_channels, window_size, max_lag
            )
            for t in range(n_windows)
        ]
        results = Parallel(n_jobs=n_jobs)(
            tqdm(tasks, total=n_windows, desc="Threshold nulo (janelas)", leave=False)
        )
        chunks.extend(results)

    null_syncs = np.concatenate(chunks)
    threshold = float(np.percentile(null_syncs, percentile))

    return {
        "threshold": threshold,
        "percentile": percentile,
        "null_mean": float(np.mean(null_syncs)),
        "null_std": float(np.std(null_syncs)),
        "n_shuffles": n_shuffles,
        "n_null_edges": int(null_syncs.size),
        "n_surrogates": n_shuffles,
        "null_syncs": null_syncs,
    }
