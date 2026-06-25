"""Reproducible synthetic point clouds with known topology.

These manifolds are the ground truth for validating Mapper: a circle should reduce to a
single loop, a torus to a surface with two independent loops, and a sphere to a hollow
shell. Every cloud is deterministic given a seed, so tests can assert on it byte-for-byte.
"""

from __future__ import annotations

import numpy as np


def circle(n_points: int = 400, *, noise: float = 0.0, seed: int = 42) -> np.ndarray:
    """Points on the unit circle S^1 in R^2.

    Args:
        n_points: Number of points to sample.
        noise: Standard deviation of radial Gaussian noise (0 = exact circle).
        seed: Seed for reproducibility.

    Returns:
        Array of shape (n_points, 2).
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    radius = 1.0 + noise * rng.standard_normal(n_points)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def torus(
    n_points: int = 600,
    *,
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    noise: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Points on the surface of a torus S^1 x S^1 in R^3.

    Args:
        n_points: Number of points to sample.
        major_radius: Distance from the tube centre to the torus centre (R).
        minor_radius: Tube radius (r), with r < R.
        noise: Standard deviation of Gaussian noise added in R^3.
        seed: Seed for reproducibility.

    Returns:
        Array of shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)  # around the torus centre
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_points)  # around the tube
    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)
    cloud = np.column_stack([x, y, z])
    if noise:
        cloud += noise * rng.standard_normal(cloud.shape)
    return cloud


def sphere(
    n_points: int = 400, *, radius: float = 1.0, noise: float = 0.0, seed: int = 42
) -> np.ndarray:
    """Points (almost) uniformly distributed on the sphere S^2 in R^3.

    Uses the standard trick of normalising Gaussian vectors, which is uniform on the sphere.

    Args:
        n_points: Number of points to sample.
        radius: Sphere radius.
        noise: Standard deviation of radial Gaussian noise.
        seed: Seed for reproducibility.

    Returns:
        Array of shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scale = radius + noise * rng.standard_normal((n_points, 1))
    return directions * scale
