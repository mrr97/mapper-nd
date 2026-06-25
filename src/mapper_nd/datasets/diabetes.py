"""The Reaven & Miller (1979) diabetes study — the headline real-world dataset.

145 non-obese adults, five blood-chemistry measurements each, clinically labelled as
Normal, Chemical_Diabetic or Overt_Diabetic. Reaven and Miller found the cloud forms a
central core with two "flares" branching off it — one per diabetes subtype. Mapper is
meant to recover exactly that shape, which is the demo this project is built around.

Source: heplots::Diabetes (R), mirrored as CSV by the Rdatasets project. Bundled under
``data/diabetes.csv`` so the project is self-contained and reproducible offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = ["relwt", "glufast", "glutest", "instest", "sspg"]
LABEL_COLUMN = "group"

_DEFAULT_PATH = Path("data/diabetes.csv")


@dataclass(frozen=True)
class DiabetesData:
    """The diabetes point cloud plus its clinical labels."""

    points: np.ndarray  # (145, 5) blood-chemistry measurements
    labels: list[str]  # clinical group per patient
    feature_names: list[str]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]


def load_diabetes(path: str | Path | None = None) -> DiabetesData:
    """Load the bundled diabetes study as a point cloud in R^5.

    Args:
        path: Optional override for the CSV location (defaults to ``data/diabetes.csv``).

    Returns:
        A :class:`DiabetesData` with the numeric measurements and clinical labels.
    """
    csv_path = Path(path) if path is not None else _DEFAULT_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Diabetes dataset not found at {csv_path}.")

    frame = pd.read_csv(csv_path)
    points = frame[FEATURE_NAMES].to_numpy(dtype=float)
    labels = frame[LABEL_COLUMN].astype(str).tolist()
    return DiabetesData(points=points, labels=labels, feature_names=list(FEATURE_NAMES))
