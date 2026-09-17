import pytest

from edge_msd.icc_coupled.fixed_modes import ConstantMode


def test_constant_mode_matches_physical_ms_discounting_and_terminal_interval():
    policy = ConstantMode(0)
    # Core deployment cost exists before the first action, as in PPO rollouts.
    states = [
        {"cost":200, "violations":0, "report_bytes":0, "time_ms":0},
        {"cost":300, "violations":1, "report_bytes":32, "time_ms":5},
        {"cost":330, "violations":3, "report_bytes":32, "time_ms":8},
    ]
    for accounting in states:
        assert policy.act({"accounting":accounting}) == (.25, 2)
    policy.finish({"accounting":{"cost":350, "violations":3,
                                 "report_bytes":64, "time_ms":10}})
    assert policy.raw_return == pytest.approx(-3.01-4.3-.21)
    assert policy.discounted_return == pytest.approx(-3.01-.995**5*4.3-.995**8*.21)
    assert len(policy.intervals) == 3
