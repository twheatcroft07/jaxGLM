"""pytest configuration for the CPU test tier.

Forces JAX onto the CPU backend so the fast suite runs anywhere (CI, a login node) without a GPU,
and registers the `slow` marker for the GPU-scale tests (skipped by default; run with `-m slow`).
"""
import os

# Must be set before any test module imports jax. conftest is imported first by pytest.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running / GPU-scale test (skipped by default)")
