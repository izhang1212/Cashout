import numpy as np

from parlay_follower.shared.stern import SternModel


def test_win_prob_basic_sanity():
    m = SternModel(sigma_per_min=1.7, pregame_spread=0.0)
    assert m.home_win_prob(0, 48.0) == 0.5            # even game, no drift
    assert m.home_win_prob(20, 5.0) > 0.99            # big late lead
    assert m.home_win_prob(-20, 5.0) < 0.01
    assert m.home_win_prob(3, 0.0) == 1.0             # buzzer, leading


def test_favorite_drift_direction():
    fav = SternModel(pregame_spread=-6.5)             # home favored
    dog = SternModel(pregame_spread=+6.5)
    assert fav.home_win_prob(0, 48.0) > 0.5 > dog.home_win_prob(0, 48.0)


def test_transition_matrix_rows_sum_to_one():
    m = SternModel()
    grid = np.arange(-45, 46, dtype=float)
    P = m.transition_matrix(grid, dt_min=0.5)
    assert np.allclose(P.sum(axis=1), 1.0)
    # mass should center near the diagonal for small dt
    assert P[45, 45] > P[45, 0]


def test_simulated_paths_match_closed_form():
    m = SternModel(sigma_per_min=1.7, pregame_spread=-4.0)
    rng = np.random.default_rng(0)
    paths = m.simulate_paths(0.0, 48.0, n_paths=20000, n_steps=48, rng=rng)
    mc_p = (paths[:, -1] > 0).mean()
    cf_p = m.home_win_prob(0.0, 48.0)
    assert abs(mc_p - cf_p) < 0.01
