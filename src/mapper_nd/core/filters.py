"""Filter functions (lenses): project a point cloud into R^n before covering it.

A filter ``f: X -> R^n`` is the first step of Mapper. The choice of filter decides which
geometric feature the final complex emphasises. Every filter returns a 2D array of shape
``(n_points, n_components)`` so the cover always sees a well-defined target dimension —
including ``n_components > 1``, which is what lets this implementation build
higher-dimensional simplices.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_kernels

_DISTANCE_METRICS = frozenset({"euclidean", "cityblock", "cosine"})


class Filter(ABC):
    """Base class for filter functions. Subclasses map a cloud to ``(n_points, n_comp)``."""

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Project ``X`` (shape ``(n_points, n_features)``) into the lens space."""

    @property
    @abstractmethod
    def n_components(self) -> int:
        """Dimension of the lens space this filter projects to."""


class Projection(Filter):
    """Keep selected coordinates of the data — the simplest possible lens."""

    def __init__(self, columns: int | Sequence[int] = 0) -> None:
        self.columns = [columns] if isinstance(columns, int) else list(columns)

    @property
    def n_components(self) -> int:
        return len(self.columns)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, self.columns]


class Eccentricity(Filter):
    """Eccentricity lens: how far each point sits from the rest of the cloud.

    ``E_p(x) = ( mean_y d(x, y)^p )^(1/p)``, with ``p = inf`` giving the farthest distance.
    Centres of dense regions score low, outliers score high. Projects to R^1.
    """

    def __init__(self, metric: str = "euclidean", exponent: float = 2.0) -> None:
        if metric not in _DISTANCE_METRICS:
            raise ValueError(
                f"Unsupported metric {metric!r}; choose from {sorted(_DISTANCE_METRICS)}."
            )
        if exponent < 1:
            raise ValueError("exponent must be >= 1 (or math.inf).")
        self.metric = metric
        self.exponent = exponent

    @property
    def n_components(self) -> int:
        return 1

    def __call__(self, X: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(X, metric=self.metric)
        if np.isinf(self.exponent):
            values = distances.max(axis=1)
        else:
            values = np.mean(distances**self.exponent, axis=1) ** (1.0 / self.exponent)
        return values.reshape(-1, 1)


class Statistic(Filter):
    """A simple statistic of each point's distance-to-others row. Projects to R^1."""

    _FUNCS: ClassVar[dict[str, Callable[..., np.ndarray]]] = {
        "sum": np.sum,
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min,
        "std": np.std,
    }

    def __init__(self, statistic: str = "mean", metric: str = "euclidean") -> None:
        if statistic not in self._FUNCS:
            raise ValueError(f"Unknown statistic {statistic!r}; choose from {sorted(self._FUNCS)}.")
        if metric not in _DISTANCE_METRICS:
            raise ValueError(
                f"Unsupported metric {metric!r}; choose from {sorted(_DISTANCE_METRICS)}."
            )
        self.statistic = statistic
        self.metric = metric

    @property
    def n_components(self) -> int:
        return 1

    def __call__(self, X: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(X, metric=self.metric)
        values = self._FUNCS[self.statistic](distances, axis=0)
        return np.asarray(values).reshape(-1, 1)


class GraphLaplacian(Filter):
    """Normalised graph-Laplacian eigenvector embedding. Projects to R^n (n >= 1).

    Builds a kernel weight matrix over the cloud, forms the normalised adjacency
    ``M = D^{-1/2} W D^{-1/2}``, and embeds each point using its top eigenvectors scaled by
    the eigenvalues — skipping the trivial leading one. This is the lens that makes the
    arbitrary-dimensional cover useful: pick ``n_components = 2`` or ``3`` and the nerve can
    contain triangles and tetrahedra.
    """

    def __init__(
        self, kernel: str = "rbf", n_components: int = 2, gamma: float | None = None
    ) -> None:
        if kernel not in {"rbf", "laplacian"}:
            raise ValueError(f"Unsupported kernel {kernel!r}; choose 'rbf' or 'laplacian'.")
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        self.kernel = kernel
        self._n_components = n_components
        self.gamma = gamma

    @property
    def n_components(self) -> int:
        return self._n_components

    def __call__(self, X: np.ndarray) -> np.ndarray:
        weights = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma)
        degree = weights.sum(axis=1)
        inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        normalised = inv_sqrt[:, None] * weights * inv_sqrt[None, :]
        normalised = 0.5 * (normalised + normalised.T)  # enforce symmetry numerically

        eigenvalues, eigenvectors = np.linalg.eigh(normalised)
        order = np.argsort(eigenvalues)[::-1]  # descending
        # Skip the leading (trivial) eigenvector; scale the rest by their eigenvalues.
        chosen = order[1 : 1 + self._n_components]
        return eigenvectors[:, chosen] * eigenvalues[chosen]
