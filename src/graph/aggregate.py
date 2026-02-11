import numpy as np

def aggregate_static_network(tvg: np.ndarray) -> np.ndarray:
    """
    Rede estática agregada (REA):
    soma da conectividade ao longo do tempo.
    """
    return np.sum(tvg, axis=2)


def edges_per_time(tvg: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Número de arestas por janela temporal.
    """
    return np.sum(tvg > threshold, axis=(0, 1)) / 2


def degree_over_time(tvg: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Grau de cada nó ao longo do tempo.

    Returns
    -------
    degree_t : np.ndarray
        Array (n_nodes, n_time)
    """
    return np.sum(tvg > threshold, axis=1)


def hub_occurrence(tvg: np.ndarray, k_sigma: float = 2.0) -> np.ndarray:
    """
    Frequência (normalizada) com que cada nó atua como hub no tempo.
    """
    degree_t = degree_over_time(tvg)
    mean_k = np.mean(degree_t, axis=1, keepdims=True)
    std_k = np.std(degree_t, axis=1, keepdims=True)

    hubs = degree_t > (mean_k + k_sigma * std_k)
    hub_count = np.sum(hubs, axis=1)

    return hub_count / np.sum(hub_count)


def weighted_degree_static(tvg: np.ndarray) -> np.ndarray:
    """
    Grau ponderado dos nós na rede estática agregada.
    """
    rea = aggregate_static_network(tvg)
    return np.sum(rea, axis=1)


def temporal_statistics(x: np.ndarray) -> dict:
    """
    Estatísticas temporais básicas para séries dinâmicas.
    """
    return {
        "mean": np.mean(x, axis=-1),
        "std": np.std(x, axis=-1),
        "cv": np.std(x, axis=-1) / (np.mean(x, axis=-1) + 1e-10)
    }
