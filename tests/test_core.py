"""Tests for the Mapper engine: filters, cover, nerve, and end-to-end on known shapes."""

import math

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from mapper_nd.core import (
    Eccentricity,
    GraphLaplacian,
    HypercubeCover,
    Mapper,
    Projection,
    SimplicialComplex,
    Statistic,
    Vertex,
)
from mapper_nd.datasets import circle, sphere

# --- filters --------------------------------------------------------------------------


def test_projection_selects_columns():
    X = np.arange(12.0).reshape(4, 3)
    lens = Projection([0, 2])(X)
    assert lens.shape == (4, 2)
    assert np.array_equal(lens, X[:, [0, 2]])


def test_eccentricity_is_high_for_outliers():
    X = np.vstack([np.zeros((10, 2)), [[10.0, 10.0]]])  # tight cluster + one far point
    lens = Eccentricity(metric="euclidean", exponent=2)(X)
    assert lens.shape == (11, 1)
    assert lens.argmax() == 10  # the outlier is the most eccentric


def test_eccentricity_infinite_exponent_is_the_max_distance():
    X = np.array([[0.0], [1.0], [5.0]])
    lens = Eccentricity(exponent=math.inf)(X)
    assert lens[2, 0] == pytest.approx(5.0)


def test_statistic_lens_shape_and_validation():
    X = np.random.default_rng(0).normal(size=(20, 3))
    assert Statistic("mean")(X).shape == (20, 1)
    with pytest.raises(ValueError, match="statistic"):
        Statistic("nope")


def test_graph_laplacian_projects_to_requested_dimension():
    X = sphere(60, seed=1)
    lens = GraphLaplacian(kernel="rbf", n_components=2)(X)
    assert lens.shape == (60, 2)


def test_filter_rejects_bad_params():
    with pytest.raises(ValueError, match="metric"):
        Eccentricity(metric="manhattan")
    with pytest.raises(ValueError, match="kernel"):
        GraphLaplacian(kernel="poly")


# --- cover ----------------------------------------------------------------------------


def test_cover_dimension_matches_lens_and_overlap_shares_points():
    lens = np.linspace(0, 1, 100).reshape(-1, 1)
    sparse = HypercubeCover(n_intervals=5, overlap=0.0).fit_transform(lens)
    dense = HypercubeCover(n_intervals=5, overlap=0.8).fit_transform(lens)
    # Overlap puts points into more hypercubes, so total memberships grow.
    assert sum(len(v) for v in dense.values()) > sum(len(v) for v in sparse.values())


def test_cover_is_arbitrary_dimensional():
    lens = np.random.default_rng(0).uniform(size=(200, 2))
    cover = HypercubeCover(n_intervals=4, overlap=0.5).fit(lens)
    assert cover.dimension == 2
    cubes = cover.transform(lens)
    assert all(len(key) == 2 for key in cubes)  # keys are 2D multi-indices


def test_cover_rejects_bad_overlap():
    with pytest.raises(ValueError, match="overlap"):
        HypercubeCover(overlap=1.5)


# --- nerve ----------------------------------------------------------------------------


def _vertex(vid: int, members: list[int]) -> Vertex:
    return Vertex(id=vid, hypercube=(vid,), members=np.array(members))


def test_nerve_builds_edge_when_clusters_share_a_point():
    # v0 and v1 share point 5; v2 is disjoint.
    vertices = [_vertex(0, [1, 5]), _vertex(1, [5, 9]), _vertex(2, [20, 21])]
    cx = SimplicialComplex.from_vertices(vertices, n_points=30, max_dimension=3)
    assert cx.simplices[0] == {(0,), (1,), (2,)}
    assert cx.simplices.get(1) == {(0, 1)}
    assert cx.n_connected_components() == 2


def test_nerve_builds_a_triangle_when_three_clusters_share_a_point():
    vertices = [_vertex(0, [7]), _vertex(1, [7]), _vertex(2, [7])]
    cx = SimplicialComplex.from_vertices(vertices, n_points=10, max_dimension=3)
    assert cx.simplices[2] == {(0, 1, 2)}  # the 2-simplex
    assert cx.simplices[1] == {(0, 1), (0, 2), (1, 2)}  # and all its faces
    assert cx.dimension == 2


def test_nerve_respects_max_dimension_cap():
    vertices = [_vertex(i, [0]) for i in range(5)]  # all share point 0
    cx = SimplicialComplex.from_vertices(vertices, n_points=1, max_dimension=1)
    assert cx.dimension == 1  # capped: only vertices and edges


# --- end to end -----------------------------------------------------------------------


def test_mapper_on_a_circle_is_connected_and_one_dimensional():
    X = circle(300, noise=0.02, seed=4)
    mapper = Mapper(
        filter=Projection([0]),  # project onto the x-axis
        cover=HypercubeCover(n_intervals=10, overlap=0.4),
        clusterer=DBSCAN(eps=0.3, min_samples=3),
    )
    result = mapper.fit(X)
    assert result.n_vertices > 0
    assert result.complex.dimension >= 1  # at least edges
    assert result.complex.n_connected_components() == 1  # the loop is connected


def test_mapper_separates_two_clusters_into_two_components():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=0.0, scale=0.1, size=(60, 2))
    blob_b = rng.normal(loc=5.0, scale=0.1, size=(60, 2))
    X = np.vstack([blob_a, blob_b])
    mapper = Mapper(
        filter=Projection([0]),
        cover=HypercubeCover(n_intervals=6, overlap=0.2),
        clusterer=DBSCAN(eps=0.5, min_samples=3),
    )
    result = mapper.fit(X)
    assert result.complex.n_connected_components() == 2


def test_mapper_builds_high_dimensional_simplices_with_2d_lens():
    # The differentiator: a 2D lens lets the nerve exceed a plain graph (dimension > 1).
    X = sphere(300, seed=2)
    mapper = Mapper(
        filter=GraphLaplacian(kernel="rbf", n_components=2),
        cover=HypercubeCover(n_intervals=5, overlap=0.5),
        clusterer=DBSCAN(eps=0.4, min_samples=3),
    )
    result = mapper.fit(X)
    assert result.complex.dimension >= 2  # triangles (or higher), not just a graph
