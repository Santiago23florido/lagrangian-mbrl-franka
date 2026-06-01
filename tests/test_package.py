"""Smoke tests that the package imports and exposes its version.

These run without Isaac Sim / a GPU (env imports are lazy), so they're safe in CI.
"""

import lagrangian_mbrl


def test_version_exposed():
    assert isinstance(lagrangian_mbrl.__version__, str)
    assert lagrangian_mbrl.__version__


def test_subpackages_importable():
    # Importing should not require Isaac Sim or a GPU.
    import lagrangian_mbrl.eval  # noqa: F401
    import lagrangian_mbrl.models  # noqa: F401
    import lagrangian_mbrl.utils  # noqa: F401
