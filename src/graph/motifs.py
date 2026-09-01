"""Código com a classificação de motifs"""

import numpy as np

def trans_motifs(signal: np.ndarray) -> np.ndarray:

    """
    Transforma o sinal contínuo em motifs (ordem 3),
    conforme a implementação original de motif-synchronization
    (Rosário et al., 2015).


    Parameters
    ----------
    signal : np.ndarray
        Array (n_channels, n_samples)

    Returns
    -------
    motifs : np.ndarray
        Array (n_channels, n_samples - 2)
        com valores inteiros {1,2,3,4,5,6}
    """

    n_channels, n_samples = signal.shape
    motifs = np.zeros((n_channels, n_samples - 2), dtype=np.int32)

    for ch in range(n_channels):
        for t in range(2, n_samples):
            x0 = signal[ch, t - 2]
            x1 = signal[ch, t - 1]
            x2 = signal[ch, t]

            # M1
            if x0 > x1 and x1 > x2 and x0 > x2:
                motifs[ch, t - 2] = 1

            # M2
            elif x0 > x1 and x1 < x2 and x0 > x2:
                motifs[ch, t - 2] = 2

            # M3
            elif x0 < x1 and x1 > x2 and x0 > x2:
                motifs[ch, t - 2] = 3

            # M4
            elif x0 > x1 and x1 < x2 and x0 < x2:
                motifs[ch, t - 2] = 4

            # M5
            elif x0 < x1 and x1 < x2 and x0 < x2:
                motifs[ch, t - 2] = 5

            # M6
            elif x0 < x1 and x1 > x2 and x0 < x2:
                motifs[ch, t - 2] = 6

    return motifs
