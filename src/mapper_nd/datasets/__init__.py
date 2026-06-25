"""Datasets for Mapper: reproducible synthetic point clouds and the diabetes study."""

from mapper_nd.datasets.diabetes import load_diabetes
from mapper_nd.datasets.synthetic import circle, sphere, torus

__all__ = ["circle", "load_diabetes", "sphere", "torus"]
