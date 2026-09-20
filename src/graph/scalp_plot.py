"""Visualização de conectividade no couro cabeludo (Plotly)."""

from typing import Any


import mne
import numpy as np
import plotly.graph_objects as go
from mne.channels.layout import _find_topomap_coords
from mne.io import BaseRaw

AXIS_LIMIT = 0.55
HEAD_PADDING = 1.08


def prepare_info_from_raw(raw: BaseRaw) -> mne.Info:
    """Retorna Info com montagem aplicada quando possível."""
    info = raw.info.copy()
    if info.get_montage() is None:
        try:
            info.set_montage("standard_1020", on_missing="warn")
        except Exception:
            pass
    return info


def subsample_frame_indices(n_windows: int, max_frames: int) -> np.ndarray:
    """Subamostra índices de janela para animação mais leve."""
    if n_windows <= max_frames:
        return np.arange(n_windows, dtype=int)
    return np.linspace(0, n_windows - 1, max_frames, dtype=int)


def build_scalp_connectivity_figure(
    tvg: np.ndarray,
    info: mne.Info,
    sfreq: float,
    frame_indices: np.ndarray,
    *,
    subject_label: str = "",
    animate: bool = True,
    frame_duration_ms: int = 150,
) -> go.Figure:
    """Constrói figura de conectividade no couro cabeludo."""
    info = info.copy()
    if info.get_montage() is None:
        try:
            info.set_montage("standard_1020", on_missing="warn")
        except Exception:
            pass

    try:
        picks = mne.pick_types(info, meg=False, eeg=True, exclude=[])
        pos = _find_topomap_coords(info, picks=picks)
    except (ValueError, RuntimeError):
        layout = mne.channels.find_layout(info)
        if layout is None:
            pos = np.array([ch["loc"][:2] for ch in info["chs"]])
        else:
            name_to_pos = dict[Any, Any](zip(layout.names, layout.pos[:, :2]))
            raw_pos = np.array(
                [name_to_pos.get(name, np.nan) for name in info.ch_names],
                dtype=float,
            )
            valid = np.isfinite(raw_pos).all(axis=1)
            if not np.any(valid):
                pos = np.zeros((len(info.ch_names), 2))
            else:
                center = np.nanmean(raw_pos[valid], axis=0)
                pos = raw_pos - center
                pos[~valid] = 0.0

    pos = pos - np.mean(pos, axis=0)
    max_r = np.max(np.linalg.norm(pos, axis=1))
    if max_r > 0:
        pos = pos / max_r * (AXIS_LIMIT * 0.9)

    ch_names = list(info.ch_names)
    frame_indices = np.asarray(frame_indices, dtype=int)
    prefix = f"{subject_label} | " if subject_label else ""

    head_r = max(
        np.max(np.linalg.norm(pos, axis=1)) * HEAD_PADDING if len(pos) else 0.0,
        AXIS_LIMIT * 0.92,
    )
    theta = np.linspace(0, 2 * np.pi, 120)
    head_x = head_r * np.cos(theta)
    head_y = head_r * np.sin(theta)

    axis_layout = {
        "visible": False,
        "range": [-AXIS_LIMIT, AXIS_LIMIT],
        "constrain": "domain",
        "fixedrange": True,
    }

    plot_frames = []
    for t in frame_indices:
        t = int(t)
        adj = tvg[:, :, t]
        n = adj.shape[0]

        edge_x = []
        edge_y = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] > 0:
                    edge_x.extend([pos[i, 0], pos[j, 0], None])
                    edge_y.extend([pos[i, 1], pos[j, 1], None])

        t_sec = t / sfreq
        title = (
            f"{prefix}TVG — couro cabeludo | "
            f"t = {t_sec:.2f} s (janela {t})"
        )
        traces = [
            go.Scatter(
                x=head_x,
                y=head_y,
                mode="lines",
                line=dict(color="rgba(180,180,180,0.7)", width=1.5),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "rgba(100, 200, 255, 0.7)", "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(
                x=pos[:, 0],
                y=pos[:, 1],
                mode="markers",
                marker={"size": 11, "color": "white", "line": {"width": 1, "color": "gray"}},
                text=ch_names,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
        ]
        plot_frames.append((str(t), title, traces))

    first_title = plot_frames[0][1]
    fig = go.Figure(data=plot_frames[0][2])

    use_animation = animate and len(plot_frames) > 1
    if use_animation:
        fig.frames = [
            go.Frame(
                name=name,
                data=traces,
                layout=go.Layout(
                    title={"text": title},
                    xaxis=dict(**axis_layout, scaleanchor="y", scaleratio=1),
                    yaxis=axis_layout,
                ),
            )
            for name, title, traces in plot_frames
        ]
        fig.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "x": 0.05,
                    "y": -0.08,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration_ms,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "x": 0.1,
                    "len": 0.85,
                    "y": -0.12,
                    "pad": {"b": 10, "t": 50},
                    "currentvalue": {"prefix": "Janela: ", "visible": True},
                    "steps": [
                        {
                            "args": [
                                [name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f"{int(name) / sfreq:.1f}s",
                            "method": "animate",
                        }
                        for name, _, _ in plot_frames
                    ],
                }
            ],
        )

    fig.update_layout(
        title=first_title,
        template="plotly_dark",
        xaxis=dict(**axis_layout, scaleanchor="y", scaleratio=1),
        yaxis=axis_layout,
        height=650,
        margin={"l": 20, "r": 20, "t": 60, "b": 80 if use_animation else 40},
        showlegend=False,
        uirevision="scalp",
    )

    return fig
