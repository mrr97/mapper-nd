"""The simplicial complex produced by Mapper, and the nerve construction that builds it.

Each vertex is a cluster of original data points. The **nerve** records which clusters
overlap: a set of ``k+1`` vertices spans a ``k``-simplex whenever their clusters share at
least one data point. Equivalently — and this is how it is computed here — every data point
is covered by some set of vertices, and that set spans a simplex. Taking all faces over all
points yields the nerve, naturally including simplices of any dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Vertex:
    """A Mapper node: a cluster of original data points within one hypercube."""

    id: int
    hypercube: tuple[int, ...]
    members: np.ndarray  # sorted original-point indices in this cluster

    @property
    def size(self) -> int:
        return int(self.members.size)


@dataclass
class SimplicialComplex:
    """Vertices plus simplices grouped by dimension (dimension 0 are the vertices)."""

    vertices: dict[int, Vertex]
    simplices: dict[int, set[tuple[int, ...]]] = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def dimension(self) -> int:
        """Highest simplex dimension present (-1 if empty)."""
        return max(self.simplices, default=-1)

    def simplex_counts(self) -> dict[int, int]:
        """Number of simplices per dimension."""
        return {dim: len(simps) for dim, simps in sorted(self.simplices.items())}

    def skeleton(self, dim: int) -> dict[int, set[tuple[int, ...]]]:
        """The k-skeleton: all simplices of dimension <= ``dim``."""
        return {d: s for d, s in self.simplices.items() if d <= dim}

    def n_connected_components(self) -> int:
        """Number of connected components (Betti-0), via union-find on the 1-skeleton."""
        parent = {vid: vid for vid in self.vertices}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for edge in self.simplices.get(1, ()):
            a, b = find(edge[0]), find(edge[1])
            if a != b:
                parent[a] = b
        return len({find(vid) for vid in self.vertices})

    @classmethod
    def from_vertices(
        cls, vertices: list[Vertex], *, n_points: int, max_dimension: int
    ) -> SimplicialComplex:
        """Build the nerve from clustered vertices.

        Args:
            vertices: the Mapper nodes (clusters of point indices).
            n_points: size of the original cloud, used to invert membership.
            max_dimension: largest simplex dimension to record (caps the combinatorics).

        Returns:
            The assembled simplicial complex.
        """
        by_id = {v.id: v for v in vertices}
        simplices: dict[int, set[tuple[int, ...]]] = {0: {(v.id,) for v in vertices}}

        # Invert membership: for each data point, which vertices cover it.
        covering: list[list[int]] = [[] for _ in range(n_points)]
        for v in vertices:
            for p in v.members:
                covering[int(p)].append(v.id)

        max_size = max_dimension + 1
        for cover in covering:
            if len(cover) < 2:
                continue
            cover_sorted = sorted(cover)
            top = min(len(cover_sorted), max_size)
            for size in range(2, top + 1):
                bucket = simplices.setdefault(size - 1, set())
                for face in combinations(cover_sorted, size):
                    bucket.add(face)
        return cls(vertices=by_id, simplices=simplices)
