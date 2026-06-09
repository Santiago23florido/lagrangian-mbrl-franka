"""Phase-0 exit-criterion test: DeLaN beats the MLP on held-out accel MSE.

This is the headline deliverable check. It is heavier than the other unit tests
(it trains two models), so it is kept small and marked ``slow``.
"""

import pytest

from lagrangian_mbrl.offline import OfflineFitConfig, run_phase0_comparison


@pytest.mark.slow
def test_delan_beats_mlp_on_offline_acceleration_mse():
    cfg = OfflineFitConfig(
        system="two_link",
        n_train=256,
        n_val=1024,
        epochs=400,
        batch_size=64,
        lr=3e-3,
        hidden_sizes=(128, 128),
        mlp_hidden_sizes=(256, 256, 256),
        seed=0,
    )
    results = run_phase0_comparison(cfg)
    assert results["delan_wins"], (
        f"DeLaN ({results['delan_val_accel_mse']:.3e}) did not beat "
        f"MLP ({results['mlp_val_accel_mse']:.3e})"
    )
    # Structure should help by a large margin in this A1-satisfying regime.
    assert results["improvement_ratio"] > 3.0
