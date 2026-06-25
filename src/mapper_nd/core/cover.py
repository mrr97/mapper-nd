"""Overlapping hypercube cover of the lens space, in arbitrary dimension.

Mapper covers the image of the filter with open sets and clusters each preimage. Here the
cover is a grid of ``n_intervals`` hypercubes per dimension that overlap by a fixed
fraction. In R^1 these are intervals, in R^2 squares, in R^3 cubes, and in R^n hypercubes —
the general case is what allows up to ``2^n`` sets to overlap and the nerve to reach
dimension ``2^n - 1``.
"""

from __future__ import annotations

from itertools import product

import numpy as np


class HypercubeCover:
    """A regular grid of overlapping hypercubes over the lens space."""

    def __init__(self, n_intervals: int = 10, overlap: float = 0.5) -> None:
        if n_intervals < 1:
            raise ValueError("n_intervals must be >= 1.")
        # overlap is the fraction of the base interval width that adjacent cubes share.
        if not 0.0 <= overlap <= 1.0:
            raise ValueError("overlap must be in [0, 1].")
        self.n_intervals = n_intervals
        self.overlap = overlap
        self._centers_per_dim: list[np.ndarray] | None = None
        self._radius: np.ndarray | None = None

    @property
    def dimension(self) -> int:
        """Dimension of the lens space the cover was fitted to."""
        if self._centers_per_dim is None:
            raise RuntimeError("Cover is not fitted; call fit() first.")
        return len(self._centers_per_dim)

    def fit(self, lens: np.ndarray) -> HypercubeCover:
        """Lay out hypercube centres and half-widths from the spread of the lens."""
        lens = np.atleast_2d(np.asarray(lens, dtype=float))
        lo = lens.min(axis=0)
        hi = lens.max(axis=0)
        width = (hi - lo) / self.n_intervals
        # A degenerate dimension (constant lens) gets a tiny non-zero width so it still covers.
        width = np.where(width > 0, width, 1.0)
        self._centers_per_dim = [
            np.linspace(lo[d] + width[d] / 2, hi[d] - width[d] / 2, self.n_intervals)
            for d in range(lens.shape[1])
        ]
        self._radius = (width / 2.0) * (1.0 + self.overlap)
        return self

    def transform(self, lens: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        """Map each non-empty hypercube (by multi-index) to the point indices inside it."""
        if self._centers_per_dim is None or self._radius is None:
            raise RuntimeError("Cover is not fitted; call fit() first.")
        lens = np.atleast_2d(np.asarray(lens, dtype=float))
        members: dict[tuple[int, ...], np.ndarray] = {}
        for multi_index in product(range(self.n_intervals), repeat=self.dimension):
            center = np.array([self._centers_per_dim[d][i] for d, i in enumerate(multi_index)])
            inside = np.all(np.abs(lens - center) <= self._radius, axis=1)
            idx = np.flatnonzero(inside)
            if idx.size:
                members[multi_index] = idx
        return members

    def fit_transform(self, lens: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
        """Fit the cover to the lens and return the hypercube-to-points mapping."""
        return self.fit(lens).transform(lens)
