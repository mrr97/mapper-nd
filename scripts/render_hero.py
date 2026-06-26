"""Render the README hero image: the diabetes Mapper complex, in 3D.

Runs the bundled Reaven-Miller diabetes dataset at its validated preset and exports the
interactive app's own Plotly figure (coloured by clinical group) to a static PNG. This is
the headline demo — a central core with two flares, one per diabetes subtype. Regenerate
with:

    uv run python scripts/render_hero.py
"""

from __future__ import annotations

from pathlib import Path

from mapper_nd.ui.data import DATASETS, run_mapper
from mapper_nd.ui.figures import build_figure

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "diabetes.png"


def main() -> None:
    dataset = "diabetes"
    spec = DATASETS[dataset]
    # The app opens this dataset in 3D, but a 2D embedding makes a tighter, clearer
    # static hero (the complex is a tree, so depth adds nothing here) and fills the frame.
    run = run_mapper(
        dataset,
        filter_name=spec.filter_name,
        n_intervals=spec.n_intervals,
        overlap=spec.overlap,
        eps=spec.eps,
        min_samples=spec.min_samples,
        embedding_dim=2,
    )
    fig = build_figure(run, coloring="Clinical label", render=spec.render)
    fig.update_layout(
        title={
            "text": "Mapper on the Reaven-Miller diabetes study: a core with two diabetic flares",
            "x": 0.5,
            "font": {"size": 16, "color": "#334155"},
        },
        margin={"l": 20, "r": 20, "t": 44, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(OUT_PATH), width=1100, height=560, scale=2)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}  ({run.stats['vertices']} vertices)")


if __name__ == "__main__":
    main()
