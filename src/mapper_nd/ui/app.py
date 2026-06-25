"""Dash app: pick a dataset, tune the Mapper controls, and watch the complex form."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from mapper_nd.config import Settings, get_settings
from mapper_nd.ui.data import DATASETS, FILTERS, run_mapper
from mapper_nd.ui.figures import build_figure

_SLATE = "#334155"
_BASE_COLORINGS = ["Filter value", "Cluster size"]


def _coloring_options(dataset: str) -> list[str]:
    return [*_BASE_COLORINGS, "Clinical label"] if dataset == "diabetes" else list(_BASE_COLORINGS)


def _control(label: str, component) -> html.Div:
    return html.Div(
        style={"marginBottom": "14px"},
        children=[
            html.Label(label, style={"fontSize": "13px", "fontWeight": 600, "color": _SLATE}),
            component,
        ],
    )


def _controls(default: str) -> html.Div:
    spec = DATASETS[default]
    return html.Div(
        style={"width": "300px", "padding": "8px 16px"},
        children=[
            _control(
                "Dataset",
                dcc.Dropdown(
                    id="dataset",
                    options=[{"label": k, "value": k} for k in DATASETS],
                    value=default,
                    clearable=False,
                ),
            ),
            _control(
                "Filter (lens)",
                dcc.Dropdown(
                    id="filter-name",
                    options=[{"label": k, "value": k} for k in FILTERS],
                    value=spec.filter_name,
                    clearable=False,
                ),
            ),
            _control(
                "Hypercubes per dimension",
                dcc.Slider(id="n-intervals", min=2, max=15, step=1, value=spec.n_intervals),
            ),
            _control(
                "Overlap",
                dcc.Slider(id="overlap", min=0.0, max=0.95, step=0.05, value=spec.overlap),
            ),
            _control(
                "DBSCAN eps", dcc.Slider(id="eps", min=0.1, max=2.0, step=0.05, value=spec.eps)
            ),
            _control(
                "DBSCAN min samples",
                dcc.Slider(id="min-samples", min=1, max=10, step=1, value=spec.min_samples),
            ),
            _control(
                "Embedding dimension",
                dcc.RadioItems(
                    id="embedding-dim",
                    options=[{"label": "2D", "value": 2}, {"label": "3D", "value": 3}],
                    value=spec.embedding_dim,
                    inline=True,
                ),
            ),
            _control(
                "Colour vertices by",
                dcc.Dropdown(
                    id="coloring",
                    options=[{"label": c, "value": c} for c in _coloring_options(default)],
                    value="Clinical label" if default == "diabetes" else "Filter value",
                    clearable=False,
                ),
            ),
        ],
    )


def create_app(settings: Settings | None = None) -> Dash:
    """Build the Dash app. The Mapper run happens inside the callback on every change."""
    settings = settings or get_settings()
    default = "diabetes"
    app = Dash(__name__, title="mapper-nd")

    app.layout = html.Div(
        style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "system-ui"},
        children=[
            html.H2("mapper-nd — Topological Data Analysis", style={"color": _SLATE}),
            html.P(
                "Mapper reduces a point cloud to a simplicial complex that captures its shape. "
                "Tune the controls and watch it rebuild.",
                style={"color": _SLATE},
            ),
            html.Div(
                style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
                children=[
                    _controls(default),
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Div(
                                id="stats",
                                style={"color": _SLATE, "fontWeight": 600, "minHeight": "22px"},
                            ),
                            dcc.Loading(dcc.Graph(id="complex-graph", style={"height": "560px"})),
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("filter-name", "value"),
        Output("n-intervals", "value"),
        Output("overlap", "value"),
        Output("eps", "value"),
        Output("min-samples", "value"),
        Output("embedding-dim", "value"),
        Output("coloring", "options"),
        Output("coloring", "value"),
        Input("dataset", "value"),
        prevent_initial_call=True,
    )
    def _apply_preset(dataset: str):
        spec = DATASETS[dataset]
        options = [{"label": c, "value": c} for c in _coloring_options(dataset)]
        coloring = "Clinical label" if dataset == "diabetes" else "Filter value"
        return (
            spec.filter_name,
            spec.n_intervals,
            spec.overlap,
            spec.eps,
            spec.min_samples,
            spec.embedding_dim,
            options,
            coloring,
        )

    @app.callback(
        Output("complex-graph", "figure"),
        Output("stats", "children"),
        Input("dataset", "value"),
        Input("filter-name", "value"),
        Input("n-intervals", "value"),
        Input("overlap", "value"),
        Input("eps", "value"),
        Input("min-samples", "value"),
        Input("embedding-dim", "value"),
        Input("coloring", "value"),
    )
    def _update(
        dataset, filter_name, n_intervals, overlap, eps, min_samples, embedding_dim, coloring
    ):
        run = run_mapper(
            dataset,
            filter_name=filter_name,
            n_intervals=int(n_intervals),
            overlap=float(overlap),
            eps=float(eps),
            min_samples=int(min_samples),
            embedding_dim=int(embedding_dim),
            settings=settings,
        )
        if coloring not in run.colorings:
            coloring = "Filter value"
        s = run.stats
        stats = (
            f"{s['vertices']} vertices · {s['edges']} edges · {s['triangles']} triangles · "
            f"{s['components']} components · complex dimension {s['dimension']}"
        )
        return build_figure(run, coloring=coloring), stats

    return app


def main() -> None:
    """Entry point for the `dashboard` CLI command."""
    settings = get_settings()
    app = create_app(settings)
    app.run(host=settings.ui_host, port=settings.ui_port, debug=False)
