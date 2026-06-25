"""Tests for the visualisation layer: embedding, data orchestration, figures, app build."""

import numpy as np
import plotly.graph_objects as go

from mapper_nd.core import HypercubeCover, Mapper, Projection, embed_vertices
from mapper_nd.datasets import circle
from mapper_nd.ui import app as ui_app
from mapper_nd.ui.data import DATASETS, MapperRun, run_mapper
from mapper_nd.ui.figures import build_figure


def _small_run() -> MapperRun:
    return run_mapper(
        "circle",
        filter_name="Projection (1D)",
        n_intervals=8,
        overlap=0.4,
        eps=0.35,
        min_samples=3,
        embedding_dim=2,
    )


# --- embedding ------------------------------------------------------------------------


def test_embedding_returns_a_position_per_vertex_and_is_reproducible():
    X = circle(200, noise=0.02, seed=1)
    mapper = Mapper(filter=Projection(0), cover=HypercubeCover(8, 0.4))
    cx = mapper.fit(X).complex
    first = embed_vertices(cx, X, n_components=2, seed=7)
    second = embed_vertices(cx, X, n_components=2, seed=7)
    assert set(first) == set(cx.vertices)
    assert all(first[v].shape == (2,) for v in first)
    assert all(np.allclose(first[v], second[v]) for v in first)


def test_embedding_can_target_three_dimensions():
    X = circle(200, seed=2)
    cx = Mapper(filter=Projection(0), cover=HypercubeCover(8, 0.4)).fit(X).complex
    coords = embed_vertices(cx, X, n_components=3, seed=1)
    assert all(coords[v].shape == (3,) for v in coords)


# --- data orchestration ---------------------------------------------------------------


def test_run_mapper_produces_a_consistent_run():
    run = _small_run()
    assert run.complex.n_vertices == len(run.positions)
    assert run.stats["vertices"] == run.complex.n_vertices
    assert "Filter value" in run.colorings
    assert run.embedding_dim == 2


def test_diabetes_run_exposes_clinical_labels():
    run = run_mapper(
        "diabetes",
        filter_name="Statistic (sum)",
        n_intervals=8,
        overlap=0.6,
        eps=0.8,
        min_samples=2,
        embedding_dim=2,
    )
    assert "Clinical label" in run.colorings
    values, categorical = run.colorings["Clinical label"]
    assert categorical is True
    assert set(values) <= {"Normal", "Chemical_Diabetic", "Overt_Diabetic"}


def test_every_dataset_has_a_valid_preset():
    for name, spec in DATASETS.items():
        assert spec.embedding_dim in (2, 3)
        assert 0.0 <= spec.overlap <= 1.0
        assert name  # non-empty key


# --- figures --------------------------------------------------------------------------


def test_build_figure_2d_returns_a_plotly_figure():
    fig = build_figure(_small_run(), coloring="Filter value")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_build_figure_3d_with_graph_laplacian_lens():
    run = run_mapper(
        "sphere",
        filter_name="Graph Laplacian (2D)",
        n_intervals=5,
        overlap=0.5,
        eps=0.45,
        min_samples=3,
        embedding_dim=3,
    )
    fig = build_figure(run, coloring="Filter value")
    assert isinstance(fig, go.Figure)


# --- app build ------------------------------------------------------------------------


def test_create_app_builds_without_error():
    app = ui_app.create_app()
    assert app.layout is not None
    assert "mapper-nd" in str(app.layout)
