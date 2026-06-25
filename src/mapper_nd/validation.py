"""Validate that Mapper recovers the known topology of each bundled dataset.

Each dataset has a topological *signature* — the Betti numbers its true shape must have.
Running Mapper at the tuned default preset and computing the homology of the result lets us
check, automatically and honestly, that the pipeline reconstructs the right shape:

- circle   -> b0 = 1, b1 = 1   (one connected loop)
- sphere   -> b0 = 1, b2 = 1   (a hollow shell, one enclosed void)
- torus    -> b0 = 1, b1 = 2   (one connected surface with two independent loops)
- diabetes -> b0 = 1, b1 = 0   (a connected tree: a core with two diabetic flares)

Mapper is an approximation, so the torus is checked on its signature (connected, two loops)
rather than its full Betti vector — recovering ``b2 = 1`` as well needs denser sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

from mapper_nd.config import Settings, get_settings
from mapper_nd.core import betti_numbers
from mapper_nd.ui.data import run_preset


@dataclass(frozen=True)
class ValidationCase:
    """The topological signature a dataset's reconstruction must match."""

    dataset: str
    description: str
    expected: dict[int, int]  # Betti entries that must hold (a subset of dimensions)


@dataclass
class ValidationResult:
    """Outcome of validating one dataset."""

    dataset: str
    description: str
    expected: dict[int, int]
    betti: dict[int, int]
    passed: bool


CASES: tuple[ValidationCase, ...] = (
    ValidationCase("circle", "one connected loop", {0: 1, 1: 1}),
    ValidationCase("sphere", "a hollow shell (one void)", {0: 1, 1: 0, 2: 1}),
    ValidationCase("torus", "connected, two independent loops", {0: 1, 1: 2}),
    ValidationCase("diabetes", "a connected tree (core + two flares)", {0: 1, 1: 0}),
)


def validate_dataset(case: ValidationCase, settings: Settings | None = None) -> ValidationResult:
    """Run a dataset at its preset and check its Betti signature."""
    run = run_preset(case.dataset, settings)
    betti = betti_numbers(run.complex)
    passed = all(betti.get(dim, 0) == value for dim, value in case.expected.items())
    return ValidationResult(
        dataset=case.dataset,
        description=case.description,
        expected=case.expected,
        betti=betti,
        passed=passed,
    )


def validate_all(settings: Settings | None = None) -> list[ValidationResult]:
    """Validate every dataset against its topological signature."""
    settings = settings or get_settings()
    return [validate_dataset(case, settings) for case in CASES]
