"""Recorte de EEG em intervalo contínuo comum (sem atravessar boundary)."""

from __future__ import annotations

import numpy as np


def crop_continuous_segment(
    raw,
    t0: float,
    t1: float,
    n_sec: float,
    *,
    file_name: str = "",
) -> tuple[np.ndarray, float, int, int]:
    """
    Seleciona ``n_sec`` segundos de EEG estritamente dentro de ``[t0, t1]``.

    O início é a primeira amostra após ``t0`` (conservador em relação ao boundary).
    Garante que o recorte cabe em ``[t0, t1]``, não ultrapassa o registro e não
    contém eventos ``boundary``.

    Returns
    -------
    eeg : np.ndarray
        Shape (n_channels, n_amostras).
    sfreq : float
    start, stop : int
        Índices de amostra [start, stop) usados em ``raw.get_data``.
    """
    label = file_name or getattr(getattr(raw, "filenames", [None])[0], "name", "") or "raw"
    sfreq = float(raw.info["sfreq"])
    n_amostras = int(round(n_sec * sfreq))
    if n_amostras <= 0:
        raise ValueError(f"n_sec inválido ({n_sec}) em {label}")

    # Primeira amostra seguramente posterior a t0
    start = int(np.floor(t0 * sfreq)) + 1
    stop = start + n_amostras  # exclusivo

    if stop > raw.n_times:
        raise ValueError(
            f"Trecho de {n_sec:g} s não cabe no registro ({label}): "
            f"stop={stop} > n_times={raw.n_times}"
        )

    t_stop = stop / sfreq
    if t_stop > t1 + 1e-12:
        raise ValueError(
            f"Trecho de {n_sec:g} s não cabe no intervalo comum "
            f"[{t0:.6f}, {t1:.6f}] s ({label}): fim={t_stop:.6f} s"
        )

    boundaries_no_recorte = [
        float(ann["onset"])
        for ann in raw.annotations
        if str(ann["description"]).lower() == "boundary"
        and (start / sfreq) <= float(ann["onset"]) < t_stop
    ]
    if boundaries_no_recorte:
        raise ValueError(
            f"Recorte atravessa boundary em {label}: {boundaries_no_recorte[:5]}"
        )

    eeg = raw.get_data(start=start, stop=stop)
    if eeg.shape[1] != n_amostras:
        raise AssertionError(
            f"Esperado {n_amostras} amostras, obtido {eeg.shape[1]} ({label})"
        )
    return eeg, sfreq, start, stop
