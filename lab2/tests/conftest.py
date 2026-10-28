import numpy as np
import pytest

from lab02.synthetic import simulate


@pytest.fixture(scope="session")
def small_dns():
    """Short, well-resolved DNS shared by the integration tests (about 1 s)."""
    return simulate(n=32, nu=0.06, n_frames=41, spinup=0.3)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
