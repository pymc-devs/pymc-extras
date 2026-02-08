import unittest

import numpy as np
import pytensor
import pytensor.tensor as pt

from numpy.testing import assert_allclose

from pymc_extras.statespace.core.representation import PytensorRepresentation
from tests.statespace.shared_fixtures import TEST_SEED
from tests.statespace.test_utilities import fast_eval, make_test_inputs

floatX = pytensor.config.floatX
atol = 1e-12 if floatX == "float64" else 1e-6


def unpack_ssm_dims(ssm):
    p = ssm.k_endog
    m = ssm.k_states
    r = ssm.k_posdef

    return p, m, r


class BasicFunctionality(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(TEST_SEED)

    def test_numpy_to_pytensor(self):
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=1)
        X = np.eye(5)
        X_pt = ssm._numpy_to_pytensor("transition", X)
        self.assertTrue(isinstance(X_pt, pt.TensorVariable))
        assert_allclose(ssm["transition"].type.shape, X.shape)

        assert ssm["transition"].name == "transition"

    def test_default_shapes_full_rank(self):
        ssm = PytensorRepresentation(k_endog=5, k_states=5, k_posdef=5)
        p, m, r = unpack_ssm_dims(ssm)

        assert_allclose(ssm["design"].type.shape, (p, m))
        assert_allclose(ssm["transition"].type.shape, (m, m))
        assert_allclose(ssm["selection"].type.shape, (m, r))
        assert_allclose(ssm["state_cov"].type.shape, (r, r))
        assert_allclose(ssm["obs_cov"].type.shape, (p, p))

    def test_default_shapes_low_rank(self):
        ssm = PytensorRepresentation(k_endog=5, k_states=5, k_posdef=2)
        p, m, r = unpack_ssm_dims(ssm)

        assert_allclose(ssm["design"].type.shape, (p, m))
        assert_allclose(ssm["transition"].type.shape, (m, m))
        assert_allclose(ssm["selection"].type.shape, (m, r))
        assert_allclose(ssm["state_cov"].type.shape, (r, r))
        assert_allclose(ssm["obs_cov"].type.shape, (p, p))

    def test_matrix_assignment(self):
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=2)

        ssm["design", 0, 0] = 3.0
        ssm["transition", 0, :] = 2.7
        ssm["selection", -1, -1] = 9.9

        assert_allclose(fast_eval(ssm["design"][0, 0]), 3.0, atol=atol)
        assert_allclose(fast_eval(ssm["transition"][0, :]), 2.7, atol=atol)
        assert_allclose(fast_eval(ssm["selection"][-1, -1]), 9.9, atol=atol)

        assert ssm["design"].name == "design"
        assert ssm["transition"].name == "transition"
        assert ssm["selection"].name == "selection"

    def test_build_representation_from_data(self):
        p, m, r, n = 3, 6, 1, 10
        inputs = [data, a0, P0, c, d, T, Z, R, H, Q] = make_test_inputs(
            p, m, r, n, self.rng, missing_data=0
        )

        ssm = PytensorRepresentation(
            k_endog=p,
            k_states=m,
            k_posdef=r,
            design=Z,
            transition=T,
            selection=R,
            state_cov=Q,
            obs_cov=H,
            initial_state=a0,
            initial_state_cov=P0,
            state_intercept=c,
            obs_intercept=d,
        )

        names = [
            "initial_state",
            "initial_state_cov",
            "state_intercept",
            "obs_intercept",
            "transition",
            "design",
            "selection",
            "obs_cov",
            "state_cov",
        ]

        for name, X in zip(names, inputs[1:]):
            assert_allclose(X, fast_eval(ssm[name]), err_msg=name)

        for name, X in zip(names, inputs[1:]):
            assert ssm[name].name == name
            assert_allclose(ssm[name].type.shape, X.shape, err_msg=f"{name} shape test")

    def test_assign_time_varying_matrices(self):
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=2)
        n = 10

        ssm["design", 0, 0] = 3.0
        ssm["transition", 0, :] = 2.7
        ssm["selection", -1, -1] = 9.9

        ssm["state_intercept"] = np.zeros((n, 5))
        ssm["state_intercept", :, 0] = np.arange(n)

        assert_allclose(fast_eval(ssm["design"][0, 0]), 3.0, atol=atol)
        assert_allclose(fast_eval(ssm["transition"][0, :]), 2.7, atol=atol)
        assert_allclose(fast_eval(ssm["selection"][-1, -1]), 9.9, atol=atol)
        assert_allclose(fast_eval(ssm["state_intercept"][:, 0]), np.arange(n), atol=atol)

    def test_invalid_key_name_raises(self):
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=1)
        with self.assertRaises(IndexError) as e:
            X = ssm["invalid_key"]
        msg = str(e.exception)
        self.assertEqual(msg, "invalid_key is an invalid state space matrix name")

    def test_non_string_key_raises(self):
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=1)
        with self.assertRaises(IndexError) as e:
            X = ssm[0]
        msg = str(e.exception)
        self.assertEqual(msg, "First index must the name of a valid state space matrix.")

    def test_invalid_key_tuple_raises(self):
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=1)
        with self.assertRaises(IndexError) as e:
            X = ssm[0, 1, 1]
        msg = str(e.exception)
        self.assertEqual(msg, "First index must the name of a valid state space matrix.")

    def test_slice_statespace_matrix(self):
        T = np.eye(5)
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=1, transition=T)
        T_out = ssm["transition", :3, :]
        assert_allclose(T[:3], fast_eval(T_out))

    def test_update_matrix_via_key(self):
        T = np.eye(5)
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=1)
        ssm["transition"] = T

        assert_allclose(T, fast_eval(ssm["transition"]))

    def test_update_matrix_with_invalid_shape_raises(self):
        T = np.eye(10)
        ssm = PytensorRepresentation(k_endog=3, k_states=5, k_posdef=1)
        with self.assertRaises(ValueError) as e:
            ssm["transition"] = T
        msg = str(e.exception)
        self.assertEqual(
            msg, "The last two dimensions of transition must be (5, 5), found (10, 10)"
        )


if __name__ == "__main__":
    unittest.main()
