import networkx as nx
import numpy as np

def graph_metrics(adj: np.ndarray) -> dict:
    """
    Calcula métricas globais e locais de um grafo ponderado
    a partir de uma matriz de conectividade funcional.

    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência ponderada (n_nodes x n_nodes)

    Returns
    -------
    metrics : dict
        Dicionário contendo métricas de rede
    """

    G = nx.from_numpy_array(adj)

    metrics = {}

    # Grau ponderado
    metrics["degree"] = np.array(
        [d for _, d in G.degree(weight="weight")]
    )

    # Coeficiente de agrupamento (clustering)
    metrics["clustering"] = np.array(
        list(nx.clustering(G, weight="weight").values())
    )

    # Centralidade de intermediação (betweenness)
    metrics["betweenness"] = np.array(
        list(nx.betweenness_centrality(G, weight="weight").values())
    )

    # Eficiência global (escalar)
    metrics["global_efficiency"] = nx.global_efficiency(G)

    # Eficiência local (vetor)
    eff = []

    for node in G.nodes():
        neighbors = list(G.neighbors(node))

        if len(neighbors) < 2:
            eff.append(0.0)
        else:
            subgraph = G.subgraph(neighbors)
            eff.append(nx.global_efficiency(subgraph))
    metrics["local_efficiency"] = np.array(eff)

    return metrics
