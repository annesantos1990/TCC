import streamlit as st
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns

from src.motifs import trans_motifs
from src.tvg import build_tvg
from src.aggregate import (
    aggregate_static_network,
    edges_per_time,
    hub_occurrence
)

st.set_page_config(layout="wide")
st.title("EEG Functional Connectivity – Exploratory Analysis")

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

run_button = st.sidebar.button("Rodar análise")

# ==========================
# LOAD + PROCESS
# ==========================

if run_button:

    st.subheader("Carregando EEG")

    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    st.write(raw)
    st.write(raw.info["ch_names"])

    raw.pick_types(eeg=True)
    fig = raw.plot(
        duration=5  ,
        n_channels=8,
        scalings='auto',
        show=False,
        block=False)
    
    if isinstance(fig, list):
        fig = fig[0]
    st.pyplot(fig, clear_figure=True)
    
    eeg = raw.get_data()
    sfreq = raw.info["sfreq"]

    st.write(f"Shape EEG: {eeg.shape}")
    st.write(f"Frequência amostral: {sfreq} Hz")

    window_size = int(window_ms / 1000 * sfreq)
    max_lag = int(max_lag_ms / 1000 * sfreq)

    # ==========================
    # EEG RAW VIEW
    # ==========================

    st.subheader("EEG bruto (trecho curto)")

    n_channels = min(5, eeg.shape[0])
    t = np.arange(1000) / sfreq

    fig, ax = plt.subplots(figsize=(10, 4))
    for ch in range(n_channels):
        ax.plot(t, eeg[ch, :1000] + ch * 50, label=f"Ch {ch}")

    ax.set_xlabel("Tempo (s)")
    ax.set_title("EEG bruto (offset vertical)")

    # ==========================
    # MOTIFS
    # ==========================

    st.subheader("Distribuição de Motifs")

    motifs = trans_motifs(eeg)
    motifs_flat = motifs.flatten()

    fig, ax = plt.subplots()
    ax.hist(motifs_flat, bins=np.arange(1, 7) - 0.5, rwidth=0.8)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Motif")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

    # ==========================
    # TVG
    # ==========================

    st.subheader("Time-Varying Graph (TVG)")

    tvg = build_tvg(
        motifs=motifs,
        window_size=window_size,
        max_lag=max_lag,
        threshold=None
    )

    st.write(f"TVG shape: {tvg.shape}")

    # Densidade de arestas no tempo
    edges_t = edges_per_time(tvg)

    fig, ax = plt.subplots()
    ax.plot(edges_t)
    ax.set_xlabel("Janela temporal")
    ax.set_ylabel("Número de arestas")
    ax.set_title("Dinâmica da conectividade")
    st.pyplot(fig)

    # ==========================
    # REDE ESTÁTICA AGREGADA
    # ==========================

    st.subheader("Rede Estática Agregada (REA)")

    rea = aggregate_static_network(tvg)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(rea, cmap="viridis", ax=ax)
    ax.set_title("Matriz de conectividade agregada")
    st.pyplot(fig)

    # Distribuição de pesos
    fig, ax = plt.subplots()
    ax.hist(rea[np.triu_indices_from(rea, k=1)], bins=30)
    ax.set_xlabel("Peso da aresta")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

    # ==========================
    # HUBS
    # ==========================

    st.subheader("Frequência de Hubs")

    hub_freq = hub_occurrence(tvg)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(hub_freq)), hub_freq)
    ax.set_xlabel("Canal")
    ax.set_ylabel("Frequência normalizada")
    st.pyplot(fig)

    st.success("Exploração concluída ✔️")
