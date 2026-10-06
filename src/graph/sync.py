import numpy as np


def motif_sync_pair(mot_i, mot_j, max_lag):
    """
    Calcula a sincronização por motifs entre dois canais,
    considerando atraso temporal (lag) nos dois sentidos.

    Implementa o núcleo do método de motif-synchronization
    descrito em Rosário et al. (2015): para cada atraso L em
    [0, max_lag], conta as coincidências com i antecedendo j
    (mot_i[k] == mot_j[k+L]) e com j antecedendo i
    (mot_j[k] == mot_i[k+L]), e retorna o máximo. O resultado
    é simétrico: motif_sync_pair(a, b) == motif_sync_pair(b, a).

    Parameters
    ----------
    mot_i, mot_j : np.ndarray
        Vetores 1D de motifs (inteiros)
    max_lag : int
        Atraso máximo em número de amostras, em cada sentido

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

    # Linha L compara a[k] com b[k+L]; posições com k+L >= n são inválidas
    # e excluídas da média (só pares dentro da janela).
    idx = np.arange(n, dtype=np.intp)[None, :] + np.arange(n_lags, dtype=np.intp)[:, None]
    valid = idx < n
    idx = np.minimum(idx, n - 1)
    lengths = n - np.arange(n_lags, dtype=np.float64)

    i_leads = ((mot_i[None, :] == mot_j[idx]) & valid).sum(axis=1, dtype=np.float64)
    j_leads = ((mot_j[None, :] == mot_i[idx]) & valid).sum(axis=1, dtype=np.float64)
    syncs = np.maximum(i_leads, j_leads) / lengths
    return float(syncs.max())
