"""Command-line interface for mapper-nd."""

import typer

from mapper_nd.config import get_settings
from mapper_nd.log import configure_logging

app = typer.Typer(
    name="mapper-nd",
    help="Mapper algorithm (Topological Data Analysis), implemented from scratch.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """mapper-nd command-line interface."""
    # Present so Typer keeps named subcommands even when only one exists yet.


@app.command()
def datasets() -> None:
    """List the bundled datasets and their shapes (point clouds + the diabetes study)."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from mapper_nd.datasets import circle, load_diabetes, sphere, torus

    clouds = {
        "circle": circle(settings.circle_points, seed=settings.cloud_seed),
        "torus": torus(settings.torus_points, seed=settings.cloud_seed),
        "sphere": sphere(settings.sphere_points, seed=settings.cloud_seed),
    }

    typer.echo("")
    typer.echo(f"{'dataset':<12}{'points':>8}{'dim':>6}  description")
    descriptions = {
        "circle": "S^1 loop in R^2",
        "torus": "S^1 x S^1 surface in R^3",
        "sphere": "S^2 shell in R^3",
    }
    for name, cloud in clouds.items():
        typer.echo(f"{name:<12}{cloud.shape[0]:>8}{cloud.shape[1]:>6}  {descriptions[name]}")

    diabetes = load_diabetes(settings.diabetes_path)
    typer.echo(
        f"{'diabetes':<12}{diabetes.n_points:>8}{diabetes.points.shape[1]:>6}  "
        "Reaven-Miller study (labelled Normal/Chemical/Overt)"
    )
    typer.echo("")


@app.command()
def dashboard() -> None:
    """Launch the interactive Dash app: build a Mapper complex and view it in 2D/3D."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    from mapper_nd.ui.app import create_app

    app_ = create_app(settings)
    typer.echo("")
    typer.echo(f"Dashboard running at http://{settings.ui_host}:{settings.ui_port}")
    typer.echo("Press Ctrl+C to stop.")
    app_.run(host=settings.ui_host, port=settings.ui_port, debug=False)


if __name__ == "__main__":
    app()
