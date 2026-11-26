"""Verify exact equivalence to the dense kernel catalogue and its random stream."""
import numpy as np
import pytest

from sster_data import catalog


def dense_reference(rng, knots, duration):
    dist = np.abs(knots[:, None] - knots)
    unit = np.clip(knots / (duration - 1), 0, 1)
    result = None
    for _ in range(rng.integers(1, 5)):
        kind = rng.integers(7)
        scale = catalog.logu(rng, 2, duration / 2)
        if kind == 0:
            kernel = np.exp(-.5 * (dist / scale)**2)
        elif kind == 1:
            period = catalog.logu(rng, 2, duration / 2)
            kernel = np.exp(-2 * (np.sin(np.pi * dist / period) / scale)**2)
        elif kind == 2:
            alpha = rng.uniform(.25, 4)
            kernel = (1 + (dist / scale)**2 / (2 * alpha))**(-alpha)
        elif kind == 3:
            x = np.sqrt(3) * dist / scale
            kernel = (1 + x) * np.exp(-x)
        elif kind == 4:
            kernel = np.ones_like(dist)
        elif kind == 5:
            kernel = np.eye(len(knots))
        else:
            kernel = 1 + np.outer(unit, unit)
        kernel /= max(kernel.diagonal().max(), 1e-12)
        result = (kernel if result is None else
                  result + kernel if rng.integers(2) else result * kernel)
        result /= max(result.diagonal().max(), 1e-12)
    return (result + result.T) * .5


@pytest.mark.parametrize("knots", [
    np.linspace(-83., 1142., 256), np.linspace(-2., 20., 9),
    np.array([2.7, -5., 1.1, 2.7, -1., 8.2]),
    np.array([4., 4., 4.]), np.array([-4.]),
])
def test_exact_covariance_and_rng_state(knots):
    grid = catalog._KernelGrid(knots, 47)
    for seed in range(128):
        original, revised = [np.random.default_rng(seed) for _ in range(2)]
        expected = dense_reference(original, knots, 47)
        np.testing.assert_array_equal(grid.covariance(revised), expected)
        assert original.bit_generator.state == revised.bit_generator.state


def test_gp_fields_preserves_draws_and_interpolation(monkeypatch):
    monkeypatch.setenv("SSTER_GP_DEVICE", "cpu")
    warm, duration, count = 17, 106, 5
    knots = np.linspace(-warm, duration - 1, warm + duration)
    time = np.arange(-warm, duration + 23)
    original, revised = [np.random.default_rng(418) for _ in range(2)]
    matrices = [dense_reference(r, knots, duration) for r in catalog.streams(original, count)]
    normal = original.standard_normal((count, len(knots), 1))
    expected = np.stack([np.interp(time, knots, catalog.factor(c) @ z[:, 0])
                         for c, z in zip(matrices, normal)], axis=1)
    result = catalog.gp_fields(revised, count, warm, duration, time)
    np.testing.assert_array_equal(result, expected)
    assert original.bit_generator.state == revised.bit_generator.state
