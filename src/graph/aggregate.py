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


def clustering_over_time(tvg: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Coeficiente de agrupamento binário de cada nó em cada janela.

    Nós com grau < 2 na janela não têm agrupamento definido e ficam NaN.

    Returns
    -------
    clustering_t : np.ndarray
        Array (n_nodes, n_time)
    """
    a = np.moveaxis(tvg > threshold, 2, 0).astype(np.float32)
    closed_walks = np.einsum("tij,tji->ti", a @ a, a)
    k = a.sum(axis=2)
    possible = k * (k - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        clustering = np.where(possible > 0, closed_walks / possible, np.nan)
    return clustering.T


def hub_occurrence(tvg: np.ndarray, k_sigma: float = 2.0) -> np.ndarray:
    """
    Fração das janelas em que cada nó é hub.

    Em cada janela, um nó é hub quando seu grau passa a média da rede
    naquela janela + k_sigma desvios padrão (calculados entre os nós).
    """
    degree_t = degree_over_time(tvg)
    mean_k = np.mean(degree_t, axis=0, keepdims=True)
    std_k = np.std(degree_t, axis=0, keepdims=True)

    hubs = degree_t > (mean_k + k_sigma * std_k)
    return hubs.mean(axis=1)


def window_node_features(tvg: np.ndarray) -> dict[str, np.ndarray]:
    """
    Resumos por nó das métricas binárias calculadas janela a janela.

    - sd_degree: desvio padrão no tempo do grau normalizado por (N − 1).
      A média desse grau é igual ao grau da REA normalizado.
    - bin_clustering: média no tempo do agrupamento binário (janelas com grau < 2 ignoradas).
    - hub_frequency: fração das janelas em que o nó é hub.
    """
    n_nodes = tvg.shape[0]
    degree_t = degree_over_time(tvg) / (n_nodes - 1)
    clustering_t = clustering_over_time(tvg)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(clustering_t)
        bin_clustering = np.where(
            valid.any(axis=1),
            np.nansum(clustering_t, axis=1) / np.maximum(valid.sum(axis=1), 1),
            np.nan,
        )
    return {
        "sd_degree": degree_t.std(axis=1),
        "bin_clustering": bin_clustering,
        "hub_frequency": hub_occurrence(tvg),
    }


def temporal_statistics(x: np.ndarray) -> dict:
    """
    Estatísticas temporais básicas para séries dinâmicas.
    """
    return {
        "mean": np.mean(x, axis=-1),
        "std": np.std(x, axis=-1),
        "cv": np.std(x, axis=-1) / (np.mean(x, axis=-1) + 1e-10)
    }
