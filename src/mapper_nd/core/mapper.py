"""The Mapper algorithm: filter -> cover -> cluster -> nerve.

``Mapper`` ties the pieces together. Given a point cloud it projects with the filter, covers
the lens with overlapping hypercubes, clusters the preimage of each hypercube into vertices,
and assembles the nerve. The result is a :class:`SimplicialComplex` plus the lens, ready to
visualise or analyse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.base import clone
from sklearn.cluster import DBSCAN

from mapper_nd.core.complex import SimplicialComplex, Vertex
from mapper_nd.core.cover import HypercubeCover
from mapper_nd.core.filters import Filter


@runtime_checkable
class Clusterer(Protocol):
    """Anything with scikit-learn's ``fit_predict`` works as the per-hypercube clusterer."""

    def fit_predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class MapperResult:
    """Output of :meth:`Mapper.fit`: the lens and the simplicial complex it produced."""

    lens: np.ndarray
    complex: SimplicialComplex

    @property
    def n_vertices(self) -> int:
        return self.complex.n_vertices


class Mapper:
    """Configurable Mapper pipeline.

    Args:
        filter: the lens projecting the cloud into R^n.
        cover: the overlapping hypercube cover of the lens space.
        clusterer: a scikit-learn-style clusterer applied per hypercube (default DBSCAN).
        min_cluster_size: clusters smaller than this are dropped (and tiny hypercubes skipped).
        max_dimension: cap on simplex dimension; defaults to ``2^n - 1`` for an n-dim lens,
            the most a cover of overlapping hypercubes in R^n can produce.
    """

    def __init__(
        self,
        filter: Filter,
        cover: HypercubeCover | None = None,
        clusterer: Clusterer | None = None,
        *,
        min_cluster_size: int = 1,
        max_dimension: int | None = None,
    ) -> None:
        self.filter = filter
        self.cover = cover if cover is not None else HypercubeCover()
        self.clusterer = clusterer if clusterer is not None else DBSCAN()
        self.min_cluster_size = max(1, min_cluster_size)
        self.max_dimension = max_dimension

    def fit(self, X: np.ndarray) -> MapperResult:
        """Run the full pipeline on cloud ``X`` and return the result."""
        X = np.asarray(X, dtype=float)
        lens = self.filter(X)
        hypercubes = self.cover.fit_transform(lens)

        vertices = self._build_vertices(X, hypercubes)

        lens_dim = lens.shape[1]
        max_dim = self.max_dimension if self.max_dimension is not None else 2**lens_dim - 1
        complex_ = SimplicialComplex.from_vertices(
            vertices, n_points=X.shape[0], max_dimension=max_dim
        )
        return MapperResult(lens=lens, complex=complex_)

    def _build_vertices(
        self, X: np.ndarray, hypercubes: dict[tuple[int, ...], np.ndarray]
    ) -> list[Vertex]:
        """Cluster each hypercube's preimage into vertices (in the original data space)."""
        vertices: list[Vertex] = []
        next_id = 0
        for hypercube, indices in hypercubes.items():
            if indices.size < self.min_cluster_size:
                continue
            labels = clone(self.clusterer).fit_predict(X[indices])
            for label in np.unique(labels):
                if label == -1:  # DBSCAN noise: not a cluster
                    continue
                members = np.sort(indices[labels == label])
                if members.size < self.min_cluster_size:
                    continue
                vertices.append(Vertex(id=next_id, hypercube=hypercube, members=members))
                next_id += 1
        return vertices
