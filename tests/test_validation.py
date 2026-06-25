"""Tests that each dataset's tuned preset recovers its known topology."""

from mapper_nd.validation import CASES, validate_all, validate_dataset


def test_circle_preset_recovers_a_single_loop():
    case = next(c for c in CASES if c.dataset == "circle")
    result = validate_dataset(case)
    assert result.passed
    assert result.betti[0] == 1
    assert result.betti[1] == 1


def test_diabetes_preset_is_a_connected_tree():
    case = next(c for c in CASES if c.dataset == "diabetes")
    result = validate_dataset(case)
    assert result.passed
    assert result.betti[0] == 1  # one connected shape
    assert result.betti[1] == 0  # tree-like: the core with its flares, no loops


def test_all_datasets_recover_their_signature():
    results = validate_all()
    assert {r.dataset for r in results} == {c.dataset for c in CASES}
    assert all(r.passed for r in results), {r.dataset: r.betti for r in results if not r.passed}
