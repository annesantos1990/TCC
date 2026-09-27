import numpy as np


def motif_sync_pair(mot_i, mot_j, max_lag):
    """
    Calcula a sincronização por motifs entre dois canais,
    considerando atraso temporal (lag).

    Implementa o núcleo do método de motif-synchronization
    descrito em Rosário et al. (2015), retornando o grau
    máximo de coincidência entre motifs.

    Parameters
    ----------
    mot_i, mot_j : np.ndarray
        Vetores 1D de motifs (inteiros)
    max_lag : int
        Atraso máximo em número de amostras

    Returns
    -------
    sync_value : float
        Grau de sincronização no intervalo [0, 1]
    """
    mot_i = np.asarray(mot_i)
    mot_j = np.asarray(mot_j)
    n = int(mot_i.shape[0])
    if n == 0:
        return 0.0

    max_lag = min(max(int(max_lag), 0), n - 1)
    n_lags = max_lag + 1

    # Linha L: mot_j deslocado em L amostras (compara mot_i[k] com mot_j[k+L])
    # Posições com k+L >= n são inválidas e excluídas da média.
    idx = np.arange(n, dtype=np.intp)[None, :] + np.arange(n_lags, dtype=np.intp)[:, None]
    valid = idx < n
    j_ext = np.empty(n + max_lag, dtype=mot_j.dtype)
    j_ext[:n] = mot_j
    # preenchimento irrelevante (máscara valid zera essas posições)
    if max_lag:
        j_ext[n:] = mot_j[-1]

    matches = (mot_i[None, :] == j_ext[idx]) & valid
    lengths = n - np.arange(n_lags, dtype=np.float64)
    syncs = matches.sum(axis=1, dtype=np.float64) / lengths
    return float(syncs.max())
