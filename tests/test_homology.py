"""Tests for Betti-number computation via GF(2) simplicial homology."""

import numpy as np

from mapper_nd.core import SimplicialComplex, Vertex, betti_numbers


def _complex(members_by_vertex: list[list[int]], *, max_dimension: int = 3) -> SimplicialComplex:
    vertices = [Vertex(i, (i,), np.array(m)) for i, m in enumerate(members_by_vertex)]
    return SimplicialComplex.from_vertices(vertices, n_points=200, max_dimension=max_dimension)


def test_filled_triangle_has_no_hole():
    # Three clusters sharing one common point -> a solid 2-simplex.
    betti = betti_numbers(_complex([[0], [0], [0]]))
    assert betti[0] == 1
    assert betti[1] == 0


def test_hollow_triangle_has_one_loop():
    # Pairwise overlaps but no common point -> the boundary of a triangle.
    betti = betti_numbers(_complex([[1, 2], [2, 3], [1, 3]]))
    assert betti[0] == 1
    assert betti[1] == 1


def test_square_cycle_has_one_loop():
    betti = betti_numbers(_complex([[1, 2], [2, 3], [3, 4], [4, 1]]))
    assert betti == {0: 1, 1: 1}


def test_two_disjoint_clusters_give_two_components():
    betti = betti_numbers(_complex([[1, 2], [3, 4]]))
    assert betti[0] == 2
    assert betti.get(1, 0) == 0


def test_hollow_tetrahedron_encloses_a_void():
    # Every triple of the four vertices shares a point, but no point is in all four:
    # the boundary of a tetrahedron, i.e. a 2-sphere with one enclosed void.
    members = [[10, 11, 12], [10, 11, 13], [10, 12, 13], [11, 12, 13]]
    betti = betti_numbers(_complex(members))
    assert betti[0] == 1
    assert betti[1] == 0
    assert betti[2] == 1
