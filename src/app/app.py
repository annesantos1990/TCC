import sys
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.motifs import trans_motifs
from src.graph.threshold import estimate_threshold
from src.graph.tvg import build_tvg_parallel
from src.graph.aggregate import (
    aggregate_static_network,
    edges_per_time,
    degree_over_time,
    hub_occurrence
)


st.set_page_config(layout="wide")
st.title("EEG Functional Connectivity – Exploratory Analysis")

if "sync_threshold" not in st.session_state:
    st.session_state.sync_threshold = 0.0
if "threshold_info" not in st.session_state:
    st.session_state.threshold_info = None


@st.cache_data(show_spinner=False)
def load_eeg_segment(eeg_file: str, n_sec: float):
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    raw.pick_types(eeg=True)
    eeg = raw.get_data()
    sfreq = raw.info["sfreq"]
    eeg = eeg[:, :int(n_sec * sfreq)]
    return eeg, sfreq, list(raw.info["ch_names"])

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Configurações")

eeg_file = st.sidebar.text_input(
    "Caminho do arquivo EEG (.set)",
        value="data/preprocessed/sub-010060_EO.set"
)

window_ms = st.sidebar.slider("Janela (ms)", 100, 500, 200, 50)
max_lag_ms = st.sidebar.slider("Lag máximo (ms)", 10, 100, 50, 10)
n_sec = st.sidebar.slider("Duração (s)", 1, 30, 5)

st.sidebar.header("Threshold")

n_surrogates = st.sidebar.slider("Surrogates", 50, 1000, 200, 50)
percentile = st.sidebar.slider("Percentil (%)", 90, 99, 95)

if st.sidebar.button("Estimar threshold (surrogate)"):
    with st.spinner("Estimando threshold por aleatorização..."):
        eeg_est, sfreq_est, _ = load_eeg_segment(eeg_file, n_sec)
        window_size_est = int(window_ms / 1000 * sfreq_est)
        max_lag_est = int(max_lag_ms / 1000 * sfreq_est)
        motifs_est = trans_motifs(eeg_est)
        info = estimate_threshold(
            motifs_est,
            window_size_est,
            max_lag_est,
            n_surrogates=n_surrogates,
            percentile=float(percentile),
        )
        st.session_state.sync_threshold = info["threshold"]
        st.session_state.threshold_info = info

if st.session_state.threshold_info:
    info = st.session_state.threshold_info
    st.sidebar.metric("Estimado (surrogate)", f"{info['threshold']:.4f}")
    st.sidebar.caption(
        f"Nulo: μ={info['null_mean']:.4f}, σ={info['null_std']:.4f} "
        f"(p{info['percentile']:.0f}, n={info['n_surrogates']})"
    )

sync_threshold = st.sidebar.number_input(
    "Threshold de sincronização",
    min_value=0.0,
    max_value=1.0,
    value=float(st.session_state.sync_threshold),
    step=0.01,
    format="%.4f",
    help="Valor usado na binarização do TVG. Ajuste manualmente ou estime acima.",
)
st.session_state.sync_threshold = sync_threshold

run_button = st.sidebar.button("Rodar análise")

# ==========================
# LOAD + PROCESS
# ==========================

if run_button:
    st.subheader("Carregando EEG")

    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=True)
    # st.write(raw)
    # st.write(raw.info["ch_names"])

    raw.pick_types(eeg=True)
    
    n_channels = 5
    n_points = 2000
    
    eeg = raw.get_data()
    sfreq = raw.info["sfreq"]
    eeg = eeg[:, :int(n_sec * sfreq)]
    eeg_uv = eeg * 1e6

    st.write(f"Shape EEG: {eeg.shape}")
    st.write(f"Frequência amostral: {sfreq} Hz")
    t = np.arange(n_points) / sfreq

    window_size = int(window_ms / 1000 * sfreq)
    max_lag = int(max_lag_ms / 1000 * sfreq)

    # ==========================
    # EEG RAW VIEW
    # ==========================

    st.subheader("EEG bruto (trecho curto)")

    fig = go.Figure()
    
    offset = 120 

    for ch in range(n_channels):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=eeg_uv[ch, :n_points] + ch * 50,
                mode="lines",
                name=raw.info["ch_names"][ch]
            )
        )

    fig.update_layout(
        title="EEG bruto (offset vertical)",
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude (offset)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # MOTIFS
    # ==========================

    st.subheader("Distribuição de Motifs")

    motifs = trans_motifs(eeg)
    motifs_flat = motifs.flatten()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=motifs_flat,
            xbins=dict(
                start=0.5,
                end=6.5,
                size=1
            )
        )
    )    
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[1, 2, 3, 4, 5, 6]
        ),
        xaxis_title="Motif",
        yaxis_title="Frequência",
        title="Distribuição de Motifs",
        bargap=0.3,
        bargroupgap=0.1
    )

    st.plotly_chart(fig, use_container_width=False)


    # ==========================
    # TVG
    # ==========================
    @st.cache_data(show_spinner=True)
    def compute_tvg(motifs, window_size, max_lag, threshold):
        return build_tvg_parallel(
            motifs, window_size, max_lag, threshold=threshold
        )

    tvg = compute_tvg(motifs, window_size, max_lag, sync_threshold)

    st.subheader("Time-Varying Graph (TVG)")

    st.write(f"TVG shape: {tvg.shape}")
    st.write(f"Threshold aplicado: **{sync_threshold:.4f}**")

    # Densidade de arestas no tempo
    edges_t = edges_per_time(tvg)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(edges_t))),
            y=edges_t,
            mode="lines",
            line=dict(width=2),
            name="Número de arestas"
        )
    )    
    fig.update_layout(
        title="Dinâmica da conectividade",
        xaxis_title="Janela temporal",
        yaxis_title="Número de arestas",
        template="plotly_dark",
        hovermode="x unified"
    )

    y_min = min(edges_t)
    y_max = max(edges_t)
    margin = 1 if y_min == y_max else (y_max - y_min) * 0.1

    print(min(edges_t), max(edges_t))
    print(len(set(edges_t)))


    fig.update_yaxes(
        range=[y_min - 1, y_max + 1]
    )


    st.plotly_chart(fig, use_container_width=False)

    st.stop()

    # ==========================
    # REDE ESTÁTICA AGREGADA
    # ==========================

    st.subheader("Rede Estática Agregada (REA)")

    rea = aggregate_static_network(tvg)

    fig = go.Figure(
        data=go.Heatmap(
            z=rea.values if hasattr(rea, "values") else rea,
            x=rea.columns if hasattr(rea, "columns") else None,
            y=rea.index if hasattr(rea, "index") else None,
            colorscale="Viridis",
            colorbar=dict(title="Conectividade")
        )
    )

    fig.update_layout(
        title="Matriz de conectividade agregada",
        xaxis_title="Nós",
        yaxis_title="Nós",
        template="plotly_dark",
        height=700,
        width=700
    )

    st.plotly_chart(fig, use_container_width=True)


    # ==========================
    # HUBS
    # ==========================

    st.subheader("Frequência de Hubs")

    hub_freq = hub_occurrence(tvg)

    fig = go.Figure(
        go.Bar(
            x=np.arange(len(hub_freq)),
            y=hub_freq
        )
    )    

    st.plotly_chart(fig, use_container_width=True)



    # Peso médio das arestas por janela temporal
    # considera apenas conexões existentes (> 0)

    mean_weight_t = [
        np.mean(tvg[:, :, t][tvg[:, :, t] > 0])
        if np.any(tvg[:, :, t] > 0) else 0
        for t in range(tvg.shape[2])
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(mean_weight_t))),
            y=mean_weight_t,
            mode="lines+markers",
            name="Peso médio"
        )
    )

    fig.update_layout(
        title="Peso médio das arestas no tempo",
        xaxis_title="Janela temporal",
        yaxis_title="Peso médio",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)


    # grau médio
    degree_t = degree_over_time(tvg)          # (n_nodes, n_time)
    mean_degree_t = np.mean(degree_t, axis=0) # média sobre nós

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(mean_degree_t))),
            y=mean_degree_t,
            mode="lines+markers",
            name="Grau médio"
        )
    )

    fig.update_layout(
        title="Grau médio no tempo",
        xaxis_title="Janela temporal",
        yaxis_title="Grau médio",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)



    # entropia

    from scipy.stats import entropy

    degree_t = degree_over_time(tvg)

    connectivity_entropy_t = [
        entropy(degree_t[:, t] / np.sum(degree_t[:, t]))
        if np.sum(degree_t[:, t]) > 0 else 0
        for t in range(degree_t.shape[1])
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(connectivity_entropy_t))),
            y=connectivity_entropy_t,
            mode="lines+markers",
            name="Entropia"
        )
    )

    fig.update_layout(
        title="Entropia da conectividade no tempo",
        xaxis_title="Janela temporal",
        yaxis_title="Entropia",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # conectividaade

    edges_t = edges_per_time(tvg)
    delta_edges = np.diff(edges_t)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(delta_edges))),
            y=delta_edges,
            mode="lines+markers",
            name="Δ número de arestas"
        )
    )

    fig.update_layout(
        title="Variação da conectividade entre janelas",
        xaxis_title="Janela temporal",
        yaxis_title="Δ arestas",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success("Exploração concluída ✔️")
