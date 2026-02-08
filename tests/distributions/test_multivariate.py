import numpy as np
import pymc as pm
import pytensor
import pytest

import pymc_extras as pmx


class TestR2D2M2CP:
    @pytest.fixture(autouse=True)
    def fast_compile(self):
        with pytensor.config.change_flags(
            mode="FAST_COMPILE", exception_verbosity="high"
        ):
            yield

    @pytest.fixture(autouse=True)
    def model(self):
        # every method is within a model
        with pm.Model() as model:
            yield model

    @pytest.fixture(params=[True, False], ids=["centered", "non-centered"])
    def centered(self, request):
        return request.param

    @pytest.fixture(params=[["a"], ["a", "b"], ["one"]])
    def dims(self, model: pm.Model, request):
        for i, c in enumerate(request.param):
            if c == "one":
                model.add_coord(c, range(1))
            else:
                model.add_coord(c, range((i + 2) ** 2))
        return request.param

    @pytest.fixture
    def input_shape(self, dims, model):
        return [int(model.dim_lengths[d].eval()) for d in dims]

    @pytest.fixture
    def output_shape(self, dims, model):
        *hierarchy, _ = dims
        return [int(model.dim_lengths[d].eval()) for d in hierarchy]

    @pytest.fixture
    def input_std(self, input_shape):
        return np.ones(input_shape)

    @pytest.fixture
    def output_std(self, output_shape):
        return np.ones(output_shape)

    @pytest.fixture
    def r2(self):
        return 0.8

    @pytest.fixture(params=[None, 0.1], ids=["r2-std", "no-r2-std"])
    def r2_std(self, request):
        return request.param

    @pytest.fixture(params=["true", "false", "limit-1", "limit-0", "limit-all"])
    def positive_probs(self, input_std, request):
        if request.param == "true":
            return np.full_like(input_std, 0.5)
        elif request.param == "false":
            return 0.5
        elif request.param == "limit-1":
            ret = np.full_like(input_std, 0.5)
            ret[..., 0] = 1
            return ret
        elif request.param == "limit-0":
            ret = np.full_like(input_std, 0.5)
            ret[..., 0] = 0
            return ret
        elif request.param == "limit-all":
            return np.full_like(input_std, 0)

    @pytest.fixture(params=[True, False], ids=["probs-std", "no-probs-std"])
    def positive_probs_std(self, positive_probs, request):
        if request.param:
            std = np.full_like(positive_probs, 0.1)
            std[positive_probs == 0] = 0
            std[positive_probs == 1] = 0
            return std
        else:
            return None

    @pytest.fixture(params=[None, "importance", "explained"])
    def phi_args_base(self, request, input_shape):
        if input_shape[-1] < 2 and request.param is not None:
            pytest.skip("not compatible")
        elif request.param is None:
            return {}
        elif request.param == "importance":
            return {"variables_importance": np.full(input_shape, 2)}
        else:
            val = np.full(input_shape, 2)
            return {"variance_explained": val / val.sum(-1, keepdims=True)}

    @pytest.fixture(params=["concentration", "no-concentration"])
    def phi_args(self, request, phi_args_base):
        if request.param == "concentration":
            phi_args_base["importance_concentration"] = 10
        return phi_args_base

    def test_init_r2(
        self,
        dims,
        input_std,
        output_std,
        r2,
        r2_std,
        model: pm.Model,
    ):
        eps, beta = pmx.distributions.R2D2M2CP(
            "beta",
            output_std,
            input_std,
            dims=dims,
            r2=r2,
            r2_std=r2_std,
        )
        assert not np.isnan(beta.eval()).any()
        assert eps.eval().shape == output_std.shape
        assert beta.eval().shape == input_std.shape
        # r2 rv is only created if r2 std is not None
        assert "beta" in model.named_vars
        assert ("beta::r2" in model.named_vars) == (r2_std is not None), set(
            model.named_vars
        )
        # phi is only created if variable importance is not None and there is more than one var
        assert np.isfinite(model.compile_logp()(model.initial_point()))

    def test_init_importance(
        self,
        dims,
        centered,
        input_std,
        output_std,
        phi_args,
        model: pm.Model,
    ):
        eps, beta = pmx.distributions.R2D2M2CP(
            "beta",
            output_std,
            input_std,
            dims=dims,
            r2=1,
            centered=centered,
            **phi_args,
        )
        assert not np.isnan(beta.eval()).any()
        assert eps.eval().shape == output_std.shape
        assert beta.eval().shape == input_std.shape
        # r2 rv is only created if r2 std is not None
        assert "beta" in model.named_vars
        # phi is only created if variable importance is not None and there is more than one var
        assert ("beta::phi" in model.named_vars) == (
            "variables_importance" in phi_args or "importance_concentration" in phi_args
        ), set(model.named_vars)
        assert np.isfinite(model.compile_logp()(model.initial_point()))

    def test_init_positive_probs(
        self,
        dims,
        centered,
        input_std,
        output_std,
        positive_probs,
        positive_probs_std,
        model: pm.Model,
    ):
        eps, beta = pmx.distributions.R2D2M2CP(
            "beta",
            output_std,
            input_std,
            dims=dims,
            r2=1.0,
            centered=centered,
            positive_probs_std=positive_probs_std,
            positive_probs=positive_probs,
        )
        assert not np.isnan(beta.eval()).any()
        assert eps.eval().shape == output_std.shape
        assert beta.eval().shape == input_std.shape
        # r2 rv is only created if r2 std is not None
        assert "beta" in model.named_vars
        # phi is only created if variable importance is not None and there is more than one var
        assert ("beta::psi" in model.named_vars) == (
            positive_probs_std is not None and positive_probs_std.any()
        ), set(model.named_vars)
        assert np.isfinite(sum(model.point_logps().values()))

    def test_failing_importance(self, dims, input_shape, output_std, input_std):
        if input_shape[-1] < 2:
            with pytest.raises(TypeError, match="less than two variables"):
                pmx.distributions.R2D2M2CP(
                    "beta",
                    output_std,
                    input_std,
                    dims=dims,
                    r2=0.8,
                    variables_importance=abs(input_std),
                )
        else:
            pmx.distributions.R2D2M2CP(
                "beta",
                output_std,
                input_std,
                dims=dims,
                r2=0.8,
                variables_importance=abs(input_std),
            )

    def test_failing_variance_explained(self, dims, input_shape, output_std, input_std):
        if input_shape[-1] < 2:
            with pytest.raises(TypeError, match="less than two variables"):
                pmx.distributions.R2D2M2CP(
                    "beta",
                    output_std,
                    input_std,
                    dims=dims,
                    r2=0.8,
                    variance_explained=abs(input_std),
                )
        else:
            pmx.distributions.R2D2M2CP(
                "beta",
                output_std,
                input_std,
                dims=dims,
                r2=0.8,
                variance_explained=abs(input_std),
            )

    def test_failing_mutual_exclusive(self, model: pm.Model):
        with pytest.raises(
            TypeError, match="variable importance with variance explained"
        ):
            with model:
                model.add_coord("a", range(2))
            pmx.distributions.R2D2M2CP(
                "beta",
                1,
                [1, 1],
                dims="a",
                r2=0.8,
                variance_explained=[0.5, 0.5],
                variables_importance=[1, 1],
            )

    def test_limit_case_requires_std_0(self, model: pm.Model):
        model.add_coord("a", range(2))
        with pytest.raises(ValueError, match="Can't have both positive_probs"):
            pmx.distributions.R2D2M2CP(
                "beta",
                1,
                [1, 1],
                dims="a",
                r2=0.8,
                positive_probs=[0.5, 0],
                positive_probs_std=[0.3, 0.1],
            )
        with pytest.raises(ValueError, match="Can't have both positive_probs"):
            pmx.distributions.R2D2M2CP(
                "beta",
                1,
                [1, 1],
                dims="a",
                r2=0.8,
                positive_probs=[0.5, 1],
                positive_probs_std=[0.3, 0.1],
            )

    def test_limit_case_creates_masked_vars(self, model: pm.Model, centered: bool):
        model.add_coord("a", range(2))
        pmx.distributions.R2D2M2CP(
            "beta0",
            1,
            [1, 1],
            dims="a",
            r2=0.8,
            positive_probs=[0.5, 1],
            positive_probs_std=[0.3, 0],
            centered=centered,
        )
        pmx.distributions.R2D2M2CP(
            "beta1",
            1,
            [1, 1],
            dims="a",
            r2=0.8,
            positive_probs=[0.5, 0],
            positive_probs_std=[0.3, 0],
            centered=centered,
        )
        if not centered:
            assert "beta0::raw::masked" in model.named_vars, model.named_vars
            assert "beta1::raw::masked" in model.named_vars, model.named_vars
        else:
            assert "beta0::masked" in model.named_vars, model.named_vars
            assert "beta1::masked" in model.named_vars, model.named_vars
        assert "beta1::psi::masked" in model.named_vars
        assert "beta0::psi::masked" in model.named_vars

    def test_zero_length_rvs_not_created(self, model: pm.Model):
        model.add_coord("a", range(2))
        # deterministic case which should not have any new variables
        b = pmx.distributions.R2D2M2CP(
            "b1", 1, [1, 1], r2=0.5, positive_probs=[1, 1], dims="a"
        )
        assert not model.free_RVs, model.free_RVs

        b = pmx.distributions.R2D2M2CP(
            "b2",
            1,
            [1, 1],
            r2=0.5,
            positive_probs=[1, 1],
            positive_probs_std=[0, 0],
            dims="a",
        )
        assert not model.free_RVs, model.free_RVs
