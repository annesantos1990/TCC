import numpy as np

def motif_sync_pair(mot_i, mot_j, max_lag):
    """
    Calcula a sincronização por motifs entre dois canais,
    considerando atraso temporal (lag).

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

    n = len(mot_i)
    best_sync = 0.0

    for lag in range(max_lag + 1):
        valid_length = n - lag
        if valid_length <= 0:
            break

        matches = np.sum(mot_i[:valid_length] == mot_j[lag:lag + valid_length])
        sync = matches / valid_length

        if sync > best_sync:
            best_sync = sync

    return best_sync

def motif_connectivity_matrix(motifs, max_lag):
    """
    Calcula a matriz de conectividade funcional baseada
    em motif-synchronization com lag.

    Parameters
    ----------
    motifs : np.ndarray
        Array (n_channels, n_timepoints)
    max_lag : int
        Atraso máximo em amostras

    Returns
    -------
    C : np.ndarray
        Matriz de conectividade (n_channels x n_channels)
    """

    n_channels = motifs.shape[0]
    C = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            sync = motif_sync_pair(motifs[i], motifs[j], max_lag)
            C[i, j] = C[j, i] = sync

    return C


