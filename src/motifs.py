import numpy as np

def trans_motifs(signal):
    """
    Transforma o sinal contínuo em motifs (ordem 3),
    conforme a implementação original de motif-synchronization.

    Parameters
    ----------
    signal : np.ndarray
        Array (n_channels, n_samples)

    Returns
    -------
    motifs : np.ndarray
        Array (n_channels, n_samples - 2)
        com valores inteiros {1,2,3,4,5}
    """

    n_channels, n_samples = signal.shape
    motifs = np.zeros((n_channels, n_samples - 2), dtype=np.int32)

    for ch in range(n_channels):
        for t in range(2, n_samples):
            last = signal[ch, t - 2]
            blast = signal[ch, t - 1]
            val = signal[ch, t]

            if last > blast and blast > val:
                motifs[ch, t - 2] = 1
            elif last > blast and blast < val:
                motifs[ch, t - 2] = 2
            elif last < blast and blast > val:
                motifs[ch, t - 2] = 3
            elif last < blast and blast < val:
                motifs[ch, t - 2] = 4
            else:
                motifs[ch, t - 2] = 5

    return motifs
