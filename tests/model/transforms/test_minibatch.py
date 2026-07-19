import numpy as np
import pymc as pm
import pytest

from pymc_extras.model.transforms.minibatch import minibatch
from pymc_extras.utils.model_equivalence import equivalent_models

# MinibatchRandomVariable is an identity/view Op with no numba dispatch, so drawing it
# under the numba linker harmlessly falls back to object mode.
_ignore_numba_object_mode = pytest.mark.filterwarnings(
    "ignore:Numba will use object mode to run MinibatchRandomVariable:UserWarning"
)


@_ignore_numba_object_mode
def test_matches_handwritten_minibatch():
    coords = {"obs": range(5000), "feature": range(3)}
    with pm.Model(coords=coords) as model:
        x = pm.Data("x", np.ones((5000, 3)), dims=("obs", "feature"))
        y_obs = pm.Data("y_obs", np.ones(5000), dims=("obs",))
        beta = pm.Normal("beta", dims="feature")
        noise = pm.HalfNormal("noise")
        # `obs` is moved off the leading axis by the matmul; the transform finds it by
        # tracing the graph, not by matching dim names.
        pm.Normal("y", beta @ x.T, noise, observed=y_obs, dims=("obs",))

    with pm.Model(coords=coords) as reference:
        reference.add_coord("obs_minibatch", length=10)
        x = pm.Data("x", np.ones((5000, 3)), dims=("obs", "feature"))
        y_obs = pm.Data("y_obs", np.ones(5000), dims=("obs",))
        x_mb, y_obs_mb = pm.Minibatch(x, y_obs, batch_size=10)
        beta = pm.Normal("beta", dims="feature")
        noise = pm.HalfNormal("noise")
        pm.Normal(
            "y",
            beta @ x_mb.T,
            noise,
            observed=y_obs_mb,
            dims=("obs_minibatch",),
            total_size=[x.shape[0]],
        )

    mb = minibatch(model, batch_size=10)

    # Structurally equivalent to the hand-written minibatch model, and draws batches.
    assert equivalent_models(mb, reference)
    assert pm.draw(mb["y"]).shape == (10,)

    # The original model is left untouched.
    assert model["y"].eval().shape == (5000,)

    # Full-size data variables are preserved under their original names and dims.
    assert mb["x"].eval().shape == (5000, 3)
    assert tuple(mb.named_vars_to_dims["x"]) == ("obs", "feature")
    assert len(mb.coords["obs"]) == 5000

    # The observed value is batch-sized, on a relabeled minibatch dimension.
    assert tuple(mb.rvs_to_values[mb["y"]].shape.eval()) == (10,)
    assert tuple(mb.named_vars_to_dims["y"]) == ("obs_minibatch",)
    assert int(mb.dim_lengths["obs_minibatch"].eval()) == 10

    # Explicit observed/data selection matches the default.
    explicit = minibatch(model, observed="y", data=["x", "y_obs"], batch_size=10)
    assert equivalent_models(mb, explicit)


def test_relabels_client_deterministics():
    # Deterministics that are clients of the minibatch (of the data or of the resized
    # observed) are resized and relabeled; an unconnected one is left alone.
    with pm.Model(coords={"obs": range(100), "feature": range(2)}) as model:
        x = pm.Data("x", np.ones((100, 2)), dims=("obs", "feature"))
        y_obs = pm.Data("y_obs", np.ones(100), dims=("obs",))
        beta = pm.Normal("beta", dims="feature")
        mu = pm.Deterministic("mu", beta @ x.T, dims=("obs",))  # client of the data
        y = pm.Normal("y", mu, 1, observed=y_obs, dims=("obs",))
        pm.Deterministic("z", y * 2, dims=("obs",))  # client of the resized observed
        pm.Deterministic("total", beta.sum())  # unconnected

    mb = minibatch(model, batch_size=10)

    assert tuple(mb.named_vars_to_dims["mu"]) == ("obs_minibatch",)
    assert tuple(mb.named_vars_to_dims["z"]) == ("obs_minibatch",)
    assert tuple(mb["mu"].shape.eval()) == (10,)
    assert tuple(mb["z"].shape.eval()) == (10,)
    assert mb.named_vars_to_dims.get("total") is None


def test_relabels_client_of_value_only_observed():
    # The observed's distribution does not depend on data (only its value does), so a
    # downstream Deterministic is unreachable from the data; seeding the observed RV as a
    # forward source is what lets the client be relabeled and resized.
    with pm.Model(coords={"obs": range(100)}) as model:
        y_obs = pm.Data("y_obs", np.ones(100), dims="obs")
        sigma = pm.HalfNormal("sigma")  # global scale, off the minibatch axis
        y = pm.Normal("y", 0, sigma, observed=y_obs, dims="obs")
        pm.Deterministic("z", y * 2, dims="obs")

    mb = minibatch(model, batch_size=10)

    assert tuple(mb.named_vars_to_dims["z"]) == ("obs_minibatch",)
    assert tuple(mb["z"].shape.eval()) == (10,)
    assert mb.named_vars_to_dims.get("sigma") is None


def test_auto_slices_only_on_axis_data():
    # Auto-selection slices the data that shares the observed's subsampled axis and leaves
    # everything else full-size: `col_w` is on the feature axis, and `a` is a group-level
    # effect reached through an index array (only the index `group_idx` is on the axis).
    with pm.Model(coords={"obs": range(100), "feature": range(3), "group": range(5)}) as model:
        x = pm.Data("x", np.ones((100, 3)), dims=("obs", "feature"))
        col_w = pm.Data("col_w", np.ones(3), dims="feature")
        group_idx = pm.Data("group_idx", np.zeros(100, dtype="int64"), dims="obs")
        y_obs = pm.Data("y_obs", np.ones(100), dims="obs")
        beta = pm.Normal("beta", dims="feature")
        a = pm.Normal("a", dims="group")
        pm.Normal("y", (beta * col_w) @ x.T + a[group_idx], 1, observed=y_obs, dims="obs")

    mb = minibatch(model, batch_size=10)

    # The observed (and its on-axis data) is minibatched.
    assert tuple(mb.named_vars_to_dims["y"]) == ("obs_minibatch",)
    assert tuple(mb.rvs_to_values[mb["y"]].shape.eval()) == (10,)
    # The off-axis covariate and the group-level effect are untouched and not rejected.
    assert tuple(mb.named_vars_to_dims["col_w"]) == ("feature",)
    assert mb["col_w"].eval().shape == (3,)
    assert tuple(mb.named_vars_to_dims["a"]) == ("group",)
    assert tuple(mb["a"].shape.eval()) == (5,)


def test_allows_unrelated_rv_sharing_dim_name():
    # A free RV that shares the minibatched dim name but is neither a client of the data
    # nor an ancestor of the observed keeps its full-size dim and is not rejected.
    with pm.Model(coords={"obs": range(200), "feature": range(2)}) as model:
        x = pm.Data("x", np.ones((200, 2)), dims=("obs", "feature"))
        y_obs = pm.Data("y_obs", np.ones(200), dims=("obs",))
        beta = pm.Normal("beta", dims="feature")
        unrelated = pm.Normal("unrelated", dims="obs")
        pm.Deterministic("u_sum", unrelated.sum())
        pm.Normal("y", beta @ x.T, 1, observed=y_obs, dims=("obs",))

    mb = minibatch(model, batch_size=10)

    assert tuple(mb.named_vars_to_dims["unrelated"]) == ("obs",)
    assert tuple(mb["unrelated"].shape.eval()) == (200,)
    assert len(mb.coords["obs"]) == 200


def test_rejects_free_rv_depending_on_minibatched_data():
    # A free RV computed from the minibatched data (here `d` is both the observed value
    # and an input to the global prior `mu`); its logp is not rescaled.
    with pm.Model(coords={"obs": range(50)}) as model:
        d = pm.Data("d", np.ones(50), dims="obs")
        mu = pm.Normal("mu", d.sum())
        pm.Normal("o", mu, observed=d, dims="obs")

    with pytest.raises(ValueError, match="free RV or Potential"):
        minibatch(model, batch_size=5)
    assert isinstance(minibatch(model, batch_size=5, validate=False), pm.Model)


def test_rejects_potential_on_resized_observed():
    # A Potential that is a client of a minibatched observed would be evaluated on the
    # subsample without rescaling.
    with pm.Model(coords={"obs": range(50)}) as model:
        d = pm.Data("d", np.ones(50), dims="obs")
        y = pm.Normal("y", 0, 1, observed=d, dims="obs")
        pm.Potential("pot", y.sum())

    with pytest.raises(ValueError, match="free RV or Potential"):
        minibatch(model, batch_size=5)
    assert isinstance(minibatch(model, batch_size=5, validate=False), pm.Model)


def test_rejects_per_observation_effect():
    # A per-observation random effect the observed depends on is an ancestor on the
    # subsampled axis. It is caught by the backward trace even though it does not depend on
    # the data and carries a different dim name.
    with pm.Model(coords={"obs": range(200), "re_dim": range(200)}) as model:
        x = pm.Data("x", np.ones(200), dims="obs")
        y_obs = pm.Data("y_obs", np.ones(200), dims="obs")
        re = pm.Normal("re", dims="re_dim")
        pm.Normal("y", x + re, 1, observed=y_obs, dims="obs")

    with pytest.raises(ValueError, match="free RV or Potential"):
        minibatch(model, batch_size=10)
    assert isinstance(minibatch(model, batch_size=10, validate=False), pm.Model)


def test_rejects_minibatching_support_dim():
    # A single MvNormal's components are a support dimension, not an independent batch
    # dimension, so slicing the leading axis is invalid.
    with pm.Model(coords={"obs": range(50)}) as model:
        d = pm.Data("d", np.ones(50), dims="obs")
        pm.MvNormal("y", mu=d, cov=np.eye(50), observed=np.ones(50), dims="obs")

    with pytest.raises(ValueError, match="independent batch dimension"):
        minibatch(model, batch_size=5)
    assert isinstance(minibatch(model, batch_size=5, validate=False), pm.Model)


def test_rejects_data_feeding_non_selected_observed():
    # `x` feeds a second observed that the user did not select; it would be resized without
    # the matching rescaling.
    with pm.Model(coords={"obs": range(200), "feature": range(2)}) as model:
        x = pm.Data("x", np.ones((200, 2)), dims=("obs", "feature"))
        y1 = pm.Data("y1", np.ones(200), dims="obs")
        y2 = pm.Data("y2", np.ones(200), dims="obs")
        beta = pm.Normal("beta", dims="feature")
        pm.Normal("a", beta @ x.T, 1, observed=y1, dims="obs")
        pm.Normal("b", beta @ x.T, 1, observed=y2, dims="obs")

    with pytest.raises(ValueError, match="non-selected observed"):
        minibatch(model, observed="a", data=["x", "y1"], batch_size=10)
    assert isinstance(
        minibatch(model, observed="a", data=["x", "y1"], batch_size=10, validate=False), pm.Model
    )


def test_rejects_partial_data_missing_covariate():
    # The observed depends on both `x` and `z` on the obs axis; omitting `z` would leave it
    # full-size while the observed is sliced.
    with pm.Model(coords={"obs": range(200), "feature": range(2)}) as model:
        x = pm.Data("x", np.ones((200, 2)), dims=("obs", "feature"))
        z = pm.Data("z", np.ones(200), dims="obs")
        y_obs = pm.Data("y_obs", np.ones(200), dims="obs")
        beta = pm.Normal("beta", dims="feature")
        gamma = pm.Normal("gamma")
        pm.Normal("y", beta @ x.T + gamma * z, 1, observed=y_obs, dims="obs")

    with pytest.raises(ValueError, match="along the minibatched axis"):
        minibatch(model, observed="y", data=["x", "y_obs"], batch_size=10)
    assert isinstance(
        minibatch(model, observed="y", data=["x", "z", "y_obs"], batch_size=10), pm.Model
    )
    assert isinstance(
        minibatch(model, observed="y", data=["x", "y_obs"], batch_size=10, validate=False), pm.Model
    )


def test_rejects_partial_data_missing_observed_value():
    # Omitting the observed value leaves it full-size while the RV is sliced.
    with pm.Model(coords={"obs": range(200), "feature": range(2)}) as model:
        x = pm.Data("x", np.ones((200, 2)), dims=("obs", "feature"))
        y_obs = pm.Data("y_obs", np.ones(200), dims="obs")
        beta = pm.Normal("beta", dims="feature")
        pm.Normal("y", beta @ x.T, 1, observed=y_obs, dims="obs")

    with pytest.raises(ValueError, match="value of observed"):
        minibatch(model, observed="y", data=["x"], batch_size=10)


def test_batch_size_must_be_int():
    with pm.Model(coords={"obs": range(10)}) as model:
        pm.Normal("y", observed=np.ones(10), dims="obs", shape=10)
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        minibatch(model, batch_size=10.0)


def test_returns_copy_without_minibatchable_observed():
    with pm.Model() as model:
        pm.Normal("x")
        pm.Normal("y", observed=0.0)  # scalar observed, nothing to batch
    result = minibatch(model, batch_size=10)
    assert result is not model
    assert equivalent_models(result, model)
