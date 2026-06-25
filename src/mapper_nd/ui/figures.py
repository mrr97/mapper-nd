"""Turn a Mapper run into a Plotly figure (2D or 3D).

Colours follow the original thesis convention: vertices carry the chosen colouring, edges
are black, triangles (2-simplices) are blue and tetrahedra (3-simplices) are red — so the
higher-dimensional structure is visible at a glance, not just a graph.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from mapper_nd.ui.data import MapperRun

_EDGE = "#1f2937"
_TRI = "#2563eb"
_TET = "#dc2626"
_CATEGORY_COLORS = ["#0f9d8f", "#f59e0b", "#dc2626", "#6366f1", "#10b981", "#334155"]


def _positions_array(run: MapperRun) -> tuple[list[int], np.ndarray, dict[int, int]]:
    vids = sorted(run.positions)
    coords = (
        np.array([run.positions[v] for v in vids]) if vids else np.empty((0, run.embedding_dim))
    )
    index = {v: i for i, v in enumerate(vids)}
    return vids, coords, index


def _node_marker(values: list[float] | list[str], categorical: bool) -> dict:
    if categorical:
        categories = sorted(set(values))
        palette = {c: _CATEGORY_COLORS[i % len(_CATEGORY_COLORS)] for i, c in enumerate(categories)}
        return {"color": [palette[v] for v in values], "size": 9, "line": {"width": 0.5}}
    return {
        "color": values,
        "colorscale": "Viridis",
        "size": 9,
        "showscale": True,
        "colorbar": {"thickness": 12},
    }


def _edge_coords(run: MapperRun, coords: np.ndarray, index: dict[int, int], dim: int):
    axes: list[list[float]] = [[] for _ in range(dim)]
    for edge in run.complex.simplices.get(1, ()):
        a, b = index[edge[0]], index[edge[1]]
        for d in range(dim):
            axes[d] += [coords[a, d], coords[b, d], None]
    return axes


def build_figure(run: MapperRun, *, coloring: str) -> go.Figure:
    """Build the Plotly figure for the current Mapper run and chosen vertex colouring."""
    vids, coords, index = _positions_array(run)
    fig = go.Figure()
    if not vids:
        fig.add_annotation(text="No vertices — loosen the clustering or cover.", showarrow=False)
        return fig

    values, categorical = run.colorings[coloring]
    dim = run.embedding_dim

    if dim == 3:
        _build_3d(fig, run, coords, index, values, categorical, coloring)
    else:
        _build_2d(fig, run, coords, index, values, categorical, coloring)

    fig.update_layout(
        margin={"t": 30, "r": 10, "b": 10, "l": 10},
        showlegend=False,
        template="plotly_white",
        title=f"{run.stats['vertices']} vertices · coloured by {coloring.lower()}",
    )
    return fig


def _build_2d(fig, run, coords, index, values, categorical, coloring) -> None:
    ex, ey = _edge_coords(run, coords, index, 2)
    fig.add_scatter(x=ex, y=ey, mode="lines", line={"color": _EDGE, "width": 1}, hoverinfo="skip")
    for tri in run.complex.simplices.get(2, ()):
        pts = [index[v] for v in tri]
        loop = [*pts, pts[0]]  # close the triangle
        fig.add_scatter(
            x=coords[loop, 0],
            y=coords[loop, 1],
            mode="lines",
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line={"width": 0},
            hoverinfo="skip",
        )
    fig.add_scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=_node_marker(values, categorical),
        text=[f"vertex {v} ({coloring}: {values[i]})" for i, v in enumerate(index)],
        hoverinfo="text",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)


def _build_3d(fig, run, coords, index, values, categorical, coloring) -> None:
    triangles = [tuple(index[v] for v in tri) for tri in run.complex.simplices.get(2, ())]
    if triangles:
        tri = np.array(triangles)
        fig.add_mesh3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            i=tri[:, 0],
            j=tri[:, 1],
            k=tri[:, 2],
            color=_TRI,
            opacity=0.25,
            hoverinfo="skip",
        )
    tetra_faces = []
    for tet in run.complex.simplices.get(3, ()):
        p = [index[v] for v in tet]
        tetra_faces += [
            (p[0], p[1], p[2]),
            (p[0], p[1], p[3]),
            (p[0], p[2], p[3]),
            (p[1], p[2], p[3]),
        ]
    if tetra_faces:
        tf = np.array(tetra_faces)
        fig.add_mesh3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            i=tf[:, 0],
            j=tf[:, 1],
            k=tf[:, 2],
            color=_TET,
            opacity=0.18,
            hoverinfo="skip",
        )
    ex, ey, ez = _edge_coords(run, coords, index, 3)
    fig.add_scatter3d(
        x=ex, y=ey, z=ez, mode="lines", line={"color": _EDGE, "width": 2}, hoverinfo="skip"
    )
    fig.add_scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=_node_marker(values, categorical),
        text=[f"vertex {v} ({coloring}: {values[i]})" for i, v in enumerate(index)],
        hoverinfo="text",
    )
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
