import numpy as np
import pytest

from lab02.estimation import fit_l2
from lab02.intervention import (
    attenuation,
    fit_field_conditional,
    fit_window_conditional,
    placebo_windows,
    select_windows,
    three_sigma_field,
    three_sigma_windows,
)
from tests.helpers import periodic_field


def _design(rng, n_windows=4000, theta=(1.0, 1.0, 0.0), noise=0.5):
    """Rows grouped in windows of three components; component-specific spreads."""
    component = np.tile(np.arange(3), n_windows)
    window = np.repeat(np.arange(n_windows), 3)
    spread = np.array([1.0, 2.0, 0.5])[component]
    x0 = spread * rng.standard_normal(len(window))
    x1 = 0.5 * x0 + spread * rng.standard_normal(len(window))
    x2 = 0.3 * x1 + spread * rng.standard_normal(len(window))
    X = np.stack([x0, x1, x2], axis=1)
    y = X @ np.array(theta) + noise * rng.standard_normal(len(window))
    return X, y, window, component


def test_moved_windows_sit_k_sigma_out_on_their_own_side(rng):
    X, _, window, component = _design(rng)
    cond = fit_window_conditional(X, 1, component)
    k, q = 4.0, 0.1
    out = three_sigma_windows(X, cond, window, k, q, rng)
    np.testing.assert_array_equal(out[:, [0, 2]], X[:, [0, 2]])
    moved = out[:, 1] != X[:, 1]
    # Whole windows move: all three component rows together, exactly ceil(q N) of them.
    per_window = moved.reshape(-1, 3)
    assert np.all(per_window.all(axis=1) | ~per_window.any(axis=1))
    assert per_window.all(axis=1).sum() == int(np.ceil(q * 4000))
    z_before, z_after = cond.z(X[:, 1]), cond.z(out[:, 1])
    np.testing.assert_allclose(np.abs(z_after[moved]), k, rtol=1e-12)
    assert np.all(np.sign(z_after[moved]) == np.sign(z_before[moved]))


def test_conditional_law_is_per_component_and_standardizes(rng):
    X, _, _, component = _design(rng)
    cond = fit_window_conditional(X, 1, component)
    # x1 | x0 has the component spread; x2 = 0.3 x1 + noise sharpens it by sqrt(1 + 0.3^2).
    np.testing.assert_allclose(cond.sigma, np.array([1.0, 2.0, 0.5]) / np.sqrt(1.09), rtol=0.05)
    z = cond.z(X[:, 1])
    for c in range(3):
        assert abs(z[component == c].std() - 1) < 0.02


def test_placebo_redraws_from_the_conditional_law(rng):
    X, _, window, component = _design(rng)
    cond = fit_window_conditional(X, 1, component)
    out = placebo_windows(X, cond, window, 0.5, rng)
    redrawn = out[:, 1] != X[:, 1]
    z = cond.z(out[:, 1])[redrawn]
    assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.05


def test_invalid_hyperparameters_fail_loudly(rng):
    X, _, window, component = _design(rng, n_windows=100)
    cond = fit_window_conditional(X, 0, component)
    with pytest.raises(ValueError):
        three_sigma_windows(X, cond, window, 2.5, 0.1, rng)
    with pytest.raises(ValueError):
        select_windows(10, 0.0, rng)
    with pytest.raises(ValueError):
        select_windows(10, 1.5, rng)
    assert select_windows(10, 1.0, rng).all()
    # At least one window for any q > 0: ceil, not floor.
    assert select_windows(1000, 1e-4, rng).sum() == 1


@pytest.mark.parametrize("k,q", [(3.0, 0.05), (10.0, 0.05), (10.0, 0.5), (10.0, 1.0)])
def test_attenuation_of_a_real_parameter_matches_the_theory(k, q):
    rng = np.random.default_rng(int(10 * k + 100 * q))
    ratios = []
    for _ in range(5):
        X, y, window, component = _design(rng)
        cond = fit_window_conditional(X, 1, component)
        out = three_sigma_windows(X, cond, window, k, q, rng)
        ratios.append(fit_l2(out, y, "abc").theta[1])
    assert abs(np.mean(ratios) - (1 - attenuation(k, q))) < 0.03


def test_null_parameter_keeps_its_centre_and_narrows(rng):
    X, y, window, component = _design(rng, theta=(1.0, 1.0, 0.0))
    cond = fit_window_conditional(X, 2, component)
    before = fit_l2(X, y, "abc")
    out = three_sigma_windows(X, cond, window, 10.0, 0.5, rng)
    after = fit_l2(out, y, "abc")
    se_before = before.scale / np.sqrt(np.sum((X[:, 2] - cond.mean) ** 2))
    assert abs(after.theta[2]) < 3 * se_before
    se_after = after.scale / np.sqrt(np.sum((out[:, 2] - cond.mean) ** 2))
    assert se_after < 0.2 * se_before  # the moved values are high-leverage points


def test_attenuation_limits():
    assert attenuation(10.0, 1e-9) < 1e-6
    assert np.isclose(attenuation(10.0, 1.0), 1 - np.sqrt(2 / np.pi) / 10)
    assert attenuation(30.0, 0.5) > attenuation(10.0, 0.5) > attenuation(3.0, 0.5)


@pytest.fixture(scope="module")
def frames():
    return np.stack([periodic_field(18, seed=s, smooth=3.0) for s in range(4)])


def test_field_conditional_is_sharper_than_the_marginal(frames):
    model = fit_field_conditional(frames)
    assert np.all(model.sigma < 0.5 * model.marginal_std)
    # Boundary values have no full neighbourhood and keep mu = u.
    np.testing.assert_array_equal(model.mean[:, :, 0], frames[:, :, 0])


def test_field_values_move_k_sigma_out(frames, rng):
    model = fit_field_conditional(frames)
    k = 5.0
    out = three_sigma_field(frames, model, k, 0.1, rng)
    moved = out != frames
    z_after = (out - model.mean) / model.sigma[None, :, None, None, None]
    z_before = (frames - model.mean) / model.sigma[None, :, None, None, None]
    np.testing.assert_allclose(np.abs(z_after[moved]), k, rtol=1e-9)
    assert np.all(np.sign(z_after[moved]) == np.sign(z_before[moved]))
    assert abs(moved.mean() - 0.1 * (16 / 18) ** 3) < 0.005  # q of the interior values
