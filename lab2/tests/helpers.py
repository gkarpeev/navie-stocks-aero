import numpy as np


def periodic_field(n: int, seed: int, smooth: float = 2.0) -> np.ndarray:
    """Smooth random periodic vector field shaped (3, n, n, n)."""
    k = np.fft.fftfreq(n, 1.0 / n)
    kk = np.sqrt(sum(g**2 for g in np.meshgrid(k, k, k, indexing="ij")))
    noise = np.random.default_rng(seed).standard_normal((3, n, n, n))
    return np.real(
        np.fft.ifftn(
            np.fft.fftn(noise, axes=(1, 2, 3)) * np.exp(-((kk / smooth) ** 2)), axes=(1, 2, 3)
        )
    )
