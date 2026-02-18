import pymc as pm

from pymc_extras.statespace.core.statespace import PyMCStateSpace


def test_two_statespace_models_can_coexist_with_names(monkeypatch):
    monkeypatch.setattr(PyMCStateSpace, "make_symbolic_graph", lambda self: None)

    with pm.Model():
        ssm_a = PyMCStateSpace(k_endog=1, k_states=1, k_posdef=1, name="a")
        ssm_b = PyMCStateSpace(k_endog=1, k_states=1, k_posdef=1, name="b")

        assert ssm_a.graph_name("data") != ssm_b.graph_name("data")
