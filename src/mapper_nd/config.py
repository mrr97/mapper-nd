"""Application settings loaded from environment variables / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Every value can be overridden via env vars or `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Where datasets live (synthetic point clouds + the bundled diabetes study).
    data_dir: Path = Path("data")
    diabetes_path: Path = Path("data/diabetes.csv")

    # Reproducible synthetic point clouds. A single seed makes every cloud deterministic.
    cloud_seed: int = 42
    circle_points: int = 400
    torus_points: int = 600
    sphere_points: int = 400

    # Geometric realisation: MDS seed so the drawn layout is reproducible.
    mds_seed: int = 42

    # Dash UI
    ui_host: str = "127.0.0.1"
    ui_port: int = 8050

    # Logging
    log_level: str = "INFO"
    json_logs: bool = False


def get_settings() -> Settings:
    """Build a fresh Settings instance (reads `.env` if present)."""
    return Settings()
