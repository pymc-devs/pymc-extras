import numpy as np
import pymc as pm
import pytest

import pymc_experimental as pmx


class TestR2D2M2CP:
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

    @pytest.fixture(params=[True, False], ids=["probs", "no-probs"])
    def positive_probs(self, input_std, request):
        if request.param:
            return np.full_like(input_std, 0.5)
        else:
            return 0.5

    @pytest.fixture(params=[True, False], ids=["probs-std", "no-probs-std"])
    def positive_probs_std(self, positive_probs, request):
        if request.param:
            return np.full_like(positive_probs, 0.1)
        else:
            return None

    @pytest.fixture(params=[None, "importance", "explained"])
    def phi_args(self, request, input_shape):
        if input_shape[-1] < 2 and request.param is not None:
            pytest.skip("not compatible")
        elif request.param is None:
            return {}
        elif request.param == "importance":
            return {"variables_importance": np.full(input_shape, 2)}
        else:
            val = np.full(input_shape, 2)
            return {"variance_explained": val / val.sum(-1, keepdims=True)}

    def test_init(
        self,
        dims,
        centered,
        input_std,
        output_std,
        r2,
        r2_std,
        positive_probs,
        positive_probs_std,
        phi_args,
        model: pm.Model,
    ):
        eps, beta = pmx.distributions.R2D2M2CP(
            "beta",
            output_std,
            input_std,
            dims=dims,
            r2=r2,
            r2_std=r2_std,
            centered=centered,
            positive_probs_std=positive_probs_std,
            positive_probs=positive_probs,
            **phi_args
        )
        assert eps.eval().shape == output_std.shape
        assert beta.eval().shape == input_std.shape
        # r2 rv is only created if r2 std is not None
        assert ("beta::r2" in model.named_vars) == (r2_std is not None), set(model.named_vars)
        # phi is only created if variable importances is not None and there is more than one var
        assert ("beta::phi" in model.named_vars) == ("variables_importance" in phi_args), set(
            model.named_vars
        )
        assert ("beta::psi" in model.named_vars) == (positive_probs_std is not None), set(
            model.named_vars
        )

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
                "beta", output_std, input_std, dims=dims, r2=0.8, variance_explained=abs(input_std)
            )

    def test_failing_mutual_exclusive(self, model: pm.Model):
        with pytest.raises(TypeError, match="variable importance with variance explained"):
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