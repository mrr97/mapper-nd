"""Geometric realisation: place the complex's vertices in R^2 or R^3 so it can be drawn.

A Mapper vertex is a *cluster* of points, not a point, so to draw the complex we first need
coordinates for each vertex. We measure how far apart two clusters are with a Hausdorff-like
distance and then embed those distances into the plane or space with multidimensional
scaling (MDS) — the approach from the original thesis.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from mapper_nd.core.complex import SimplicialComplex


def vertex_distance_matrix(
    complex: SimplicialComplex, X: np.ndarray, *, metric: str = "euclidean"
) -> tuple[list[int], np.ndarray]:
    """Symmetric distances between the clusters behind each vertex.

    Uses ``D(Vi, Vj) = max(mean_y min_x d(x, y), mean_x min_y d(x, y))`` for ``x in Ci``,
    ``y in Cj`` — an approximation to the Hausdorff distance between the two clusters.

    Returns the ordered vertex ids and the matching distance matrix.
    """
    X = np.asarray(X, dtype=float)
    vids = sorted(complex.vertices)
    members = [complex.vertices[v].members for v in vids]
    n = len(vids)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pair = pairwise_distances(X[members[i]], X[members[j]], metric=metric)
            dij = max(pair.min(axis=0).mean(), pair.min(axis=1).mean())
            matrix[i, j] = matrix[j, i] = dij
    return vids, matrix


def embed_vertices(
    complex: SimplicialComplex,
    X: np.ndarray,
    *,
    n_components: int = 2,
    metric: str = "euclidean",
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Embed vertices into ``n_components`` dimensions for plotting.

    Args:
        complex: the simplicial complex to realise.
        X: the original point cloud (to measure inter-cluster distances).
        n_components: embedding dimension (2 or 3).
        metric: distance used between data points.
        seed: MDS seed, so the layout is reproducible.

    Returns:
        Mapping from vertex id to its position (shape ``(n_components,)``).
    """
    vids, matrix = vertex_distance_matrix(complex, X, metric=metric)
    if not vids:
        return {}
    if len(vids) <= n_components:
        # Too few points for a meaningful MDS; lay them out on a simple axis.
        coords = np.zeros((len(vids), n_components))
        coords[:, 0] = np.arange(len(vids), dtype=float)
        return dict(zip(vids, coords, strict=True))

    mds = MDS(
        n_components=n_components,
        metric="precomputed",  # the distance matrix is already computed (sklearn >= 1.9 API)
        init="random",
        random_state=seed,
        normalized_stress=False,
    )
    coords = mds.fit_transform(matrix)
    return dict(zip(vids, coords, strict=True))
