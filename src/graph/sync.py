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



