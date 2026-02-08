import numpy as np
import pymc as pm
import pytest

from pymc_extras.model.transforms.autoreparam import vip_reparametrize


@pytest.fixture
def model_c():
    # TODO: Restructure tests so they check one dist at a time
    with pm.Model(coords=dict(a=range(5))) as mod:
        m = pm.Normal("m")
        s = pm.LogNormal("s")
        pm.Normal("g", m, s, dims="a")
        pm.Exponential("e", scale=s, shape=7)
    return mod


@pytest.fixture
def model_nc():
    with pm.Model(coords=dict(a=range(5))) as mod:
        m = pm.Normal("m")
        s = pm.LogNormal("s")
        pm.Deterministic("g", pm.Normal("z", dims="a") * s + m)
        pm.Deterministic("e", pm.Exponential("z_e", 1, shape=7) * s)
    return mod


@pytest.mark.parametrize("var", ["g", "e"])
def test_reparametrize_created(model_c: pm.Model, var):
    model_reparam, vip = vip_reparametrize(model_c, [var])
    assert f"{var}" in vip.get_lambda()
    assert f"{var}::lam_logit__" in model_reparam.named_vars
    assert f"{var}::tau_" in model_reparam.named_vars
    vip.set_all_lambda(1)
    assert ~np.isfinite(model_reparam[f"{var}::lam_logit__"].get_value()).any()


@pytest.mark.parametrize("var", ["g", "e"])
def test_random_draw(model_c: pm.Model, model_nc, var):
    model_c = pm.do(model_c, {"m": 3, "s": 2})
    model_nc = pm.do(model_nc, {"m": 3, "s": 2})
    model_v, vip = vip_reparametrize(model_c, [var])
    assert var in [v.name for v in model_v.deterministics]
    c = pm.draw(model_c[var], random_seed=42, draws=1000)
    nc = pm.draw(model_nc[var], random_seed=42, draws=1000)
    vip.set_all_lambda(1)
    v_1 = pm.draw(model_v[var], random_seed=42, draws=1000)
    vip.set_all_lambda(0)
    v_0 = pm.draw(model_v[var], random_seed=42, draws=1000)
    vip.set_all_lambda(0.5)
    v_05 = pm.draw(model_v[var], random_seed=42, draws=1000)
    np.testing.assert_allclose(c.mean(), nc.mean())
    np.testing.assert_allclose(c.mean(), v_0.mean())
    np.testing.assert_allclose(v_05.mean(), v_1.mean())
    np.testing.assert_allclose(v_1.mean(), nc.mean())

    np.testing.assert_allclose(c.std(), nc.std())
    np.testing.assert_allclose(c.std(), v_0.std())
    np.testing.assert_allclose(v_05.std(), v_1.std())
    np.testing.assert_allclose(v_1.std(), nc.std())


def test_reparam_fit(model_c):
    vars = ["g", "e"]
    model_v, vip = vip_reparametrize(model_c, ["g", "e"])
    with model_v:
        vip.fit(50000, random_seed=42)
    for var in vars:
        np.testing.assert_allclose(vip.get_lambda()[var], 0, atol=0.01)


def test_multilevel():
    with pm.Model(
        coords=dict(level=["Basement", "Floor"], county=[1, 2]),
    ) as model:
        # multilevel modelling
        a = pm.Normal("a")
        s = pm.HalfNormal("s")
        a_g = pm.Normal("a_g", a, s, shape=(2,), dims="level")
        s_g = pm.HalfNormal("s_g")
        a_ig = pm.Normal("a_ig", a_g, s_g, shape=(2, 2), dims=("county", "level"))

    model_r, vip = vip_reparametrize(model, ["a_g", "a_ig"])
    assert "a_g" in vip.get_lambda()
    assert "a_ig" in vip.get_lambda()
    assert {v.name for v in model_r.free_RVs} == {
        "a",
        "s",
        "a_g::tau_",
        "s_g",
        "a_ig::tau_",
    }
    assert "a_g" in [v.name for v in model_r.deterministics]


def test_set_truncate(model_c: pm.Model):
    model_v, vip = vip_reparametrize(model_c, ["m", "g"])
    vip.set_all_lambda(0.93)
    np.testing.assert_allclose(vip.get_lambda()["g"], 0.93)
    np.testing.assert_allclose(vip.get_lambda()["m"], 0.93)
    vip.truncate_all_lambda(0.1)
    np.testing.assert_allclose(vip.get_lambda()["g"], 1)
    np.testing.assert_allclose(vip.get_lambda()["m"], 1)

    vip.set_lambda(g=0.93, m=0.9)
    np.testing.assert_allclose(vip.get_lambda()["g"], 0.93)
    np.testing.assert_allclose(vip.get_lambda()["m"], 0.9)
    vip.truncate_lambda(g=0.2)
    np.testing.assert_allclose(vip.get_lambda()["g"], 1)
    np.testing.assert_allclose(vip.get_lambda()["m"], 0.9)


@pytest.mark.xfail(reason="FIX shape computation for lambda")
def test_lambda_shape():
    with pm.Model(coords=dict(a=[1, 2])) as model:
        b1 = pm.Normal("b1", dims="a")
        b2 = pm.Normal("b2", shape=2)
        b3 = pm.Normal("b3", size=2)
        b4 = pm.Normal("b4", np.asarray([1, 2]))
    model_v, vip = vip_reparametrize(model, ["b1", "b2", "b3", "b4"])
    lams = vip.get_lambda()
    for v in ["b1", "b2", "b3", "b4"]:
        assert lams[v].shape == (2,), v


@pytest.mark.xfail(reason="FIX shape computation for lambda")
def test_lambda_shape_transformed_1d():
    with pm.Model(coords=dict(a=[1, 2])) as model:
        b1 = pm.Exponential("b1", 1, dims="a")
        b2 = pm.Exponential("b2", 1, shape=2)
        b3 = pm.Exponential("b3", 1, size=2)
        b4 = pm.Exponential("b4", np.asarray([1, 2]))
    model_v, vip = vip_reparametrize(model, ["b1", "b2", "b3", "b4"])
    lams = vip.get_lambda()
    for v in ["b1", "b2", "b3", "b4"]:
        assert lams[v].shape == (2,), v
