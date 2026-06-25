"""Betti numbers of a simplicial complex, computed by simplicial homology over GF(2).

The Betti numbers count holes by dimension: ``b0`` connected components, ``b1`` independent
loops, ``b2`` enclosed voids. They are read off the ranks of the boundary maps,

    b_k = (#k-simplices - rank d_k) - rank d_{k+1},

with all linear algebra done over the field GF(2), where orientation drops out and a simplex
is just the set of its faces. This is genuine homology — a filled triangle has ``b1 = 0``
while a hollow one has ``b1 = 1`` — not a graph-cycle heuristic.
"""

from __future__ import annotations

from mapper_nd.core.complex import SimplicialComplex


def _gf2_rank(columns: list[int]) -> int:
    """Rank over GF(2) of vectors given as integer bitmasks (linear-basis reduction)."""
    basis: dict[int, int] = {}  # highest set bit -> reduced vector
    rank = 0
    for vector in columns:
        current = vector
        while current:
            high = current.bit_length() - 1
            if high in basis:
                current ^= basis[high]
            else:
                basis[high] = current
                rank += 1
                break
    return rank


def _boundary_rank(
    k_simplices: list[tuple[int, ...]], face_index: dict[tuple[int, ...], int]
) -> int:
    """Rank of the boundary map d_k: each k-simplex maps to the bitmask of its faces."""
    columns: list[int] = []
    for simplex in k_simplices:
        mask = 0
        for omit in range(len(simplex)):
            face = simplex[:omit] + simplex[omit + 1 :]
            position = face_index.get(face)
            if position is not None:
                mask |= 1 << position
        columns.append(mask)
    return _gf2_rank(columns)


def betti_numbers(complex: SimplicialComplex, max_dim: int | None = None) -> dict[int, int]:
    """Betti numbers ``b_0 .. b_maxdim`` of the complex.

    Args:
        complex: the simplicial complex.
        max_dim: highest dimension to report (defaults to the complex's dimension).

    Returns:
        Mapping ``dimension -> Betti number``.
    """
    simplices = {dim: sorted(simps) for dim, simps in complex.simplices.items()}
    top = max(simplices, default=0) if max_dim is None else max_dim

    counts = {dim: len(simplices.get(dim, [])) for dim in range(top + 2)}
    index = {
        dim: {simplex: i for i, simplex in enumerate(simplices.get(dim, []))}
        for dim in range(top + 1)
    }

    rank = {0: 0}  # d_0 is the zero map
    for dim in range(1, top + 2):
        rank[dim] = _boundary_rank(simplices.get(dim, []), index.get(dim - 1, {}))

    return {dim: counts[dim] - rank.get(dim, 0) - rank.get(dim + 1, 0) for dim in range(top + 1)}
