"""Run Mapper for the dashboard: dataset registry, filter presets, and orchestration.

Kept free of any Dash imports so it can be unit-tested on its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from mapper_nd.config import Settings, get_settings
from mapper_nd.core import (
    Eccentricity,
    Filter,
    GraphLaplacian,
    HypercubeCover,
    Mapper,
    Projection,
    SimplicialComplex,
    Statistic,
    embed_vertices,
)
from mapper_nd.datasets import circle, load_diabetes, sphere, torus


@dataclass(frozen=True)
class DatasetSpec:
    """A selectable dataset plus the parameter preset that shows it off."""

    loader: Callable[[Settings], tuple[np.ndarray, list[str] | None]]
    filter_name: str
    n_intervals: int
    overlap: float
    eps: float
    min_samples: int
    embedding_dim: int
    render: str  # "skeleton", "surface" or "solid" — how to draw the complex by default


def _circle(s: Settings) -> tuple[np.ndarray, list[str] | None]:
    return circle(s.circle_points, noise=0.02, seed=s.cloud_seed), None


def _torus(s: Settings) -> tuple[np.ndarray, list[str] | None]:
    return torus(s.torus_points, seed=s.cloud_seed), None


def _sphere(s: Settings) -> tuple[np.ndarray, list[str] | None]:
    return sphere(s.sphere_points, seed=s.cloud_seed), None


def _diabetes(s: Settings) -> tuple[np.ndarray, list[str] | None]:
    data = load_diabetes(s.diabetes_path)
    return data.points, data.labels


# Presets tuned in F4 so each dataset opens at the parameters that recover its known
# topology (see mapper_nd.validation): circle -> one loop, sphere -> one void,
# torus -> two loops, diabetes -> a connected tree (core + two diabetic flares).
DATASETS: dict[str, DatasetSpec] = {
    "circle": DatasetSpec(_circle, "Projection (1D)", 10, 0.4, 0.35, 3, 2, "skeleton"),
    "torus": DatasetSpec(_torus, "Graph Laplacian (2D)", 5, 0.7, 0.8, 4, 3, "solid"),
    "sphere": DatasetSpec(_sphere, "Graph Laplacian (2D)", 4, 0.5, 0.55, 3, 3, "surface"),
    "diabetes": DatasetSpec(_diabetes, "Statistic (sum)", 8, 0.7, 1.4, 2, 3, "skeleton"),
}

FILTERS: dict[str, Callable[[], Filter]] = {
    "Projection (1D)": lambda: Projection(0),
    "Eccentricity": lambda: Eccentricity("euclidean", 2),
    "Statistic (sum)": lambda: Statistic("sum", "euclidean"),
    "Graph Laplacian (2D)": lambda: GraphLaplacian("rbf", n_components=2),
}


@dataclass
class MapperRun:
    """Everything the figure needs: positions, the complex, and per-vertex colourings."""

    positions: dict[int, np.ndarray]
    complex: SimplicialComplex
    embedding_dim: int
    colorings: dict[str, tuple[list[float] | list[str], bool]]  # name -> (values, categorical)
    stats: dict[str, int]


def _vertex_colorings(
    complex: SimplicialComplex, lens: np.ndarray, labels: list[str] | None
) -> dict[str, tuple[list[float] | list[str], bool]]:
    vids = sorted(complex.vertices)
    mean_lens = [float(lens[complex.vertices[v].members, 0].mean()) for v in vids]
    sizes = [float(complex.vertices[v].size) for v in vids]
    colorings: dict[str, tuple[list[float] | list[str], bool]] = {
        "Filter value": (mean_lens, False),
        "Cluster size": (sizes, False),
    }
    if labels is not None:
        label_arr = np.asarray(labels)
        dominant = [
            str(
                max(
                    set(label_arr[complex.vertices[v].members]),
                    key=list(label_arr[complex.vertices[v].members]).count,
                )
            )
            for v in vids
        ]
        colorings["Clinical label"] = (dominant, True)
    return colorings


def run_mapper(
    dataset: str,
    *,
    filter_name: str,
    n_intervals: int,
    overlap: float,
    eps: float,
    min_samples: int,
    embedding_dim: int,
    settings: Settings | None = None,
) -> MapperRun:
    """Run the full Mapper pipeline for one set of dashboard controls."""
    settings = settings or get_settings()
    spec = DATASETS[dataset]
    raw, labels = spec.loader(settings)
    X = StandardScaler().fit_transform(raw)

    filter_ = FILTERS[filter_name]()
    mapper = Mapper(
        filter=filter_,
        cover=HypercubeCover(n_intervals=n_intervals, overlap=overlap),
        clusterer=DBSCAN(eps=eps, min_samples=min_samples),
        min_cluster_size=1,
    )
    result = mapper.fit(X)

    positions = embed_vertices(
        result.complex, X, n_components=embedding_dim, seed=settings.mds_seed
    )
    colorings = _vertex_colorings(result.complex, result.lens, labels)
    stats = {
        "vertices": result.complex.n_vertices,
        "edges": len(result.complex.simplices.get(1, ())),
        "triangles": len(result.complex.simplices.get(2, ())),
        "components": result.complex.n_connected_components(),
        "dimension": max(result.complex.dimension, 0),
    }
    return MapperRun(
        positions=positions,
        complex=result.complex,
        embedding_dim=embedding_dim,
        colorings=colorings,
        stats=stats,
    )


def run_preset(dataset: str, settings: Settings | None = None) -> MapperRun:
    """Run a dataset at its tuned default preset (what the dashboard opens with)."""
    spec = DATASETS[dataset]
    return run_mapper(
        dataset,
        filter_name=spec.filter_name,
        n_intervals=spec.n_intervals,
        overlap=spec.overlap,
        eps=spec.eps,
        min_samples=spec.min_samples,
        embedding_dim=spec.embedding_dim,
        settings=settings,
    )
