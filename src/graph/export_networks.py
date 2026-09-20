"""Exportações de rede no estilo Rosário (nós, temporal, arestas)."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.graph.aggregate import edges_per_time


def _binary_adj(adj: np.ndarray) -> np.ndarray:
    a = (adj > 0).astype(float)
    np.fill_diagonal(a, 0.0)
    return a


def _average_clustering_binary(adj: np.ndarray) -> float:
    """Clustering médio em grafo não direcionado binário."""
    a = _binary_adj(adj)
    degrees = a.sum(axis=1)
    a3_diag = np.einsum("ij,jk,ki->i", a, a, a)
    possible = degrees * (degrees - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        local = np.where(possible > 0, a3_diag / possible, 0.0)
    return float(np.mean(local))


def build_nodes_table(rea: np.ndarray, ch_names: list[str]) -> pd.DataFrame:
    """Tabela por eletrodo: DegreeREA, ClusREA, WeiDegREA, ..."""
    binary = _binary_adj(rea)
    degree_rea = binary.sum(axis=1).astype(int)

    g = nx.from_numpy_array(binary)
    clus = nx.clustering(g)
    clus_rea = np.array([clus[i] for i in range(len(ch_names))])

    wei_deg = rea.sum(axis=1)
    wei_half = wei_deg / 2.0

    return pd.DataFrame(
        {
            "Elect": list(ch_names),
            "DegreeREA": degree_rea,
            "ClusREA": clus_rea,
            "WeiDegREA": wei_deg,
            "WeiDegIn": wei_half,
            "WeiDegOut": wei_half,
            "OutsideCon": 0,
            "InsideCon": 0,
            "Eindex": np.nan,
        }
    )


def build_temporal_table(tvg: np.ndarray) -> pd.DataFrame:
    """Série temporal: Time, Edges, CC por frame do TVG."""
    edges = edges_per_time(tvg)
    n_time = tvg.shape[2]
    cc = np.empty(n_time, dtype=float)
    for t in range(n_time):
        cc[t] = _average_clustering_binary(tvg[:, :, t])

    return pd.DataFrame(
        {
            "Time": np.arange(1, n_time + 1),
            "Edges": edges.astype(int),
            "CC": cc,
        }
    )


def build_edges_table(rea: np.ndarray) -> pd.DataFrame:
    """Lista de arestas da REA (índices 1-based, undirected)."""
    n = rea.shape[0]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(rea[i, j])
            if w > 0:
                rows.append(
                    {
                        "Source": i + 1,
                        "Target": j + 1,
                        "Weight": w,
                        "Type": "Undirected",
                    }
                )
    return pd.DataFrame(rows, columns=["Source", "Target", "Weight", "Type"])


def save_network_exports(
    tvg: np.ndarray,
    rea: np.ndarray,
    ch_names: list[str],
    networks_dir: Path,
    stem: str,
) -> dict[str, Path]:
    """
    Salva os 3 CSV em networks_dir:
    {stem}_nodes.csv, {stem}_temporal.csv, {stem}_edges.csv
    """
    networks_dir = Path(networks_dir)
    networks_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "nodes": networks_dir / f"{stem}_nodes.csv",
        "temporal": networks_dir / f"{stem}_temporal.csv",
        "edges": networks_dir / f"{stem}_edges.csv",
    }

    build_nodes_table(rea, ch_names).to_csv(paths["nodes"], index=False)
    build_temporal_table(tvg).to_csv(paths["temporal"], index=False)
    build_edges_table(rea).to_csv(paths["edges"], index=False)

    return paths
