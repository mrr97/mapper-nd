"""Tests for the bundled datasets: reproducibility, geometry, and integrity."""

import numpy as np
import pytest

from mapper_nd.datasets import circle, load_diabetes, sphere, torus


def test_clouds_are_reproducible_byte_for_byte():
    assert np.array_equal(circle(200, seed=7), circle(200, seed=7))
    assert np.array_equal(torus(150, seed=7), torus(150, seed=7))
    assert np.array_equal(sphere(150, seed=7), sphere(150, seed=7))


def test_different_seed_changes_the_cloud():
    assert not np.array_equal(sphere(150, seed=1), sphere(150, seed=2))


def test_clouds_have_expected_shapes():
    assert circle(300).shape == (300, 2)
    assert torus(300).shape == (300, 3)
    assert sphere(300).shape == (300, 3)


def test_circle_points_lie_on_the_unit_circle():
    cloud = circle(500, noise=0.0, seed=3)
    radii = np.linalg.norm(cloud, axis=1)
    assert np.allclose(radii, 1.0)


def test_sphere_points_lie_on_the_sphere():
    cloud = sphere(500, radius=2.0, noise=0.0, seed=3)
    radii = np.linalg.norm(cloud, axis=1)
    assert np.allclose(radii, 2.0)


def test_torus_points_satisfy_the_torus_equation():
    r_major, r_minor = 3.0, 1.0
    cloud = torus(500, major_radius=r_major, minor_radius=r_minor, noise=0.0, seed=3)
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    # (sqrt(x^2 + y^2) - R)^2 + z^2 = r^2 holds on the torus surface.
    residual = (np.sqrt(x**2 + y**2) - r_major) ** 2 + z**2
    assert np.allclose(residual, r_minor**2)


def test_noise_perturbs_the_cloud_off_the_manifold():
    clean = circle(400, noise=0.0, seed=5)
    noisy = circle(400, noise=0.1, seed=5)
    assert not np.allclose(np.linalg.norm(noisy, axis=1), 1.0)
    assert clean.shape == noisy.shape


def test_diabetes_dataset_loads_with_expected_integrity():
    data = load_diabetes()
    assert data.n_points == 145
    assert data.points.shape == (145, 5)
    assert data.feature_names == ["relwt", "glufast", "glutest", "instest", "sspg"]
    assert len(data.labels) == 145
    # The three clinical groups Reaven & Miller identified.
    assert set(data.labels) == {"Normal", "Chemical_Diabetic", "Overt_Diabetic"}
    assert not np.isnan(data.points).any()


def test_diabetes_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_diabetes(tmp_path / "absent.csv")
