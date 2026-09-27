import networkx as nx
import numpy as np


def graph_metrics(adj: np.ndarray, n_windows: int | None = None) -> dict:
    """
    Calcula métricas locais da rede estática agregada (REA) ponderada.

    A REA costuma ser completa (todo par de eletrodos sincroniza em pelo
    menos uma janela), então métricas binárias como grau binário e
    eficiência global/local valem o mesmo para todos os registros e não
    são calculadas.

    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência ponderada (n_nodes x n_nodes); o peso de cada
        aresta é o número de janelas em que o par esteve sincronizado.
    n_windows : int, optional
        Número de janelas do TVG. Se informado, o grau ponderado (força) é
        dividido por n_windows * (n_nodes - 1) e fica entre 0 e 1.

    Returns
    -------
    metrics : dict
        degree (grau ponderado / força), clustering (ponderado) e
        betweenness (com distância = 1 / peso).
    """
    adj = np.array(adj, dtype=float)
    np.fill_diagonal(adj, 0.0)
    n_nodes = adj.shape[0]

    G = nx.from_numpy_array(adj)
    metrics = {}

    strength = adj.sum(axis=1)
    if n_windows:
        strength = strength / (n_windows * (n_nodes - 1))
    metrics["degree"] = strength

    # networkx divide cada peso pelo maior peso da rede: mede a intensidade
    # relativa dos triângulos, não a presença deles
    metrics["clustering"] = np.array(
        list(nx.clustering(G, weight="weight").values())
    )

    # na intermediação o peso é comprimento do caminho: pares que
    # sincronizam mais vezes precisam ficar mais próximos
    for _, _, data in G.edges(data=True):
        data["distance"] = 1.0 / data["weight"]
    metrics["betweenness"] = np.array(
        list(nx.betweenness_centrality(G, weight="distance").values())
    )

    return metrics
