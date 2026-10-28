import numpy as np
import pytest

from lab02.divergence import THRESHOLD
from lab02.experiment import (
    Design,
    Plan,
    WeakSettings,
    critical_fraction,
    decision_tests,
    eliminate,
    final_keep,
    rule,
    sweep,
)
from lab02.intervention import attenuation
from lab02.synthetic import simulate, with_noise

SMALL_PLAN = Plan(ks=(3.0, 10.0), qs=(0.05, 1.0), seeds=2, placebo_seeds=1)


def _design(rng, theta=(1.0, 0.2, 0.0), n_windows=3000, noise=1.0):
    component = np.tile(np.arange(3), n_windows)
    window = np.repeat(np.arange(n_windows), 3)
    X = rng.standard_normal((3 * n_windows, 3))
    X[:, 2] += 0.6 * X[:, 1]  # the control is correlated with V, as on the data
    y = X @ np.array(theta) + noise * rng.standard_normal(3 * n_windows)
    return Design("test", y, X, window // 20, window, component)


def test_rule_uses_the_shift_threshold_and_the_sign():
    assert rule("V", 1.0, THRESHOLD * 1.01)["keep"]
    assert not rule("V", 1.0, THRESHOLD)["keep"]  # strictly above the threshold
    wrong = rule("V", -1.0, 100.0)
    assert wrong["significant"] and not wrong["admissible"] and not wrong["keep"]


def test_known_structure_is_recovered(rng):
    steps = eliminate(_design(rng), Plan(), key=1)
    assert [s["removed"] for s in steps] == ["D", None]
    assert final_keep(steps) == {"A": True, "V": True, "D": False}


def test_significant_parameter_of_the_wrong_sign_is_removed(rng):
    steps = eliminate(_design(rng, theta=(1.0, -0.3, 0.0)), Plan(), key=1)
    tests = steps[0]["tests"]
    assert tests["V"]["significant"] and not tests["V"]["admissible"]
    assert final_keep(steps) == {"A": True, "V": False, "D": False}


def test_one_element_at_a_time_weakest_first(rng):
    design = _design(rng, theta=(1.0, 0.0, 0.0))
    steps = eliminate(design, Plan(), key=1)
    assert [len(s["model"]) for s in steps] == [3, 2, 1]
    first = steps[0]["tests"]
    weakest = min(("V", "D"), key=lambda e: first[e]["kl_shift"])
    assert steps[0]["removed"] == weakest


def test_full_strength_shift_follows_the_attenuation_theory(rng):
    design = _design(rng, theta=(1.0, 0.2, 0.0))
    tests = decision_tests(design, Plan(), key=1)
    expected = attenuation(10.0, 1.0) ** 2 * tests["V"]["removal_kl"]
    assert abs(tests["V"]["kl_shift"] / expected - 1) < 0.1
    # The null control has nothing to lose: its centre moves by noise only.
    assert tests["D"]["kl_shift"] < THRESHOLD


def test_critical_fraction_interpolates_in_logs():
    curve = {"x": [1e-3, 1e-2, 1e-1, 1.0], "median": [0.1, 1.0, 10.0, 20.0]}
    found = critical_fraction(curve)
    assert np.isclose(found["value"], 10 ** (-2 + np.log10(THRESHOLD)))
    assert found["upper_bound"] is False
    never = critical_fraction({"x": [0.1, 1.0], "median": [0.1, 1.0]})
    assert never == {"value": None, "upper_bound": False}
    # Above the threshold from the first point: only a bound, and it must say so.
    edge = critical_fraction({"x": [0.1, 1.0], "median": [5.0, 9.0]})
    assert edge == {"value": 0.1, "upper_bound": True}


def test_plan_is_validated():
    with pytest.raises(ValueError):
        Plan(ks=(2.0, 10.0))
    with pytest.raises(ValueError):
        Plan(qs=(0.05, 0.5))  # no full-strength point
    with pytest.raises(ValueError):
        Plan(k_sweep_q=0.3)  # the sweeps must cross


def test_unidentifiable_column_is_rejected(rng):
    design = _design(rng)
    X = design.X.copy()
    X[:, 1] = 0.0
    with pytest.raises(ValueError):
        Design("bad", design.y, X, design.clusters, design.window, design.component)


def test_sweep_is_reproducible_and_keyed(rng):
    design = _design(rng, n_windows=600)

    def thetas(key):
        return [r.dists["l2", "godambe"].mean for r in sweep(design, SMALL_PLAN, key)]

    first, again, other = thetas(1), thetas(1), thetas(2)
    assert all(np.array_equal(a, b) for a, b in zip(first, again))
    # Full strength moves every window: the same for any key; partial ones differ.
    differs = [not np.array_equal(a, b) for a, b in zip(first, other)]
    assert any(differs)


@pytest.mark.parametrize("alpha,control_kept", [(0.0, False), (0.05, True)])
def test_dns_pipeline_recovers_the_structure(alpha, control_kept):
    # Integration: DNS -> weak form -> distributions -> interventions -> decisions.
    data = with_noise(simulate(n=32, nu=0.06, n_frames=41, spinup=0.3, alpha=alpha), 0.05, 3)
    design = Design.build(data, WeakSettings(mx=6, mt=6, stride_x=2, stride_t=4))
    assert design.meta["pressure_column_max"] == 0.0  # the pressure branch is invisible
    keep = final_keep(eliminate(design, Plan(), key=1))
    assert keep == {"A": True, "V": True, "D": control_kept}
